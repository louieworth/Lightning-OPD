"""CPU regression for health probes arriving between cache flush and offload."""

import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests


ROOT = Path(__file__).resolve().parents[1]


def load_engine():
    # Exercise the actual HTTP methods without importing CUDA/Ray on the CPU.
    path = ROOT / "slime/backends/sglang_utils/sglang_engine.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SGLangEngine")
    clock = SimpleNamespace(monotonic=Mock(return_value=0.0), sleep=Mock())
    namespace = dict(RayActor=object, requests=requests, time=clock, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[cls.name], clock


class OffloadTests(unittest.TestCase):
    def setUp(self):
        engine_type, self.clock = load_engine()
        self.state = state = SimpleNamespace(paused=False, offloaded=False, pending_health=False,
                                            busy_flushes=0, fail_release=False, fail_continue=False,
                                            health_probes=0, crashed=False)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status=200, payload=None):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def health_probe(self):
                state.health_probes += 1
                if not state.paused and not state.offloaded:
                    state.pending_health = True

            def do_GET(self):
                if self.path != "/flush_cache":
                    self.respond(404)
                    return
                if state.busy_flushes:
                    state.busy_flushes -= 1
                    self.respond(400)
                    return
                state.pending_health = False
                # Deterministically put a router health probe in the exact
                # flush/release window which killed job 636940.
                self.health_probe()
                self.respond()

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/pause_generation":
                    state.paused = True
                    state.pending_health = False
                elif self.path == "/continue_generation":
                    if state.fail_continue:
                        self.respond(503, "continue failed")
                        return
                    state.paused = False
                elif self.path == "/release_memory_occupation":
                    if state.pending_health:
                        state.crashed = True
                        self.respond(500, "release_memory_occupation requires no ongoing request")
                        return
                    if state.fail_release:
                        self.respond(500, "release failed")
                        return
                    state.offloaded = True
                elif self.path == "/resume_memory_occupation":
                    state.offloaded = False
                else:
                    self.respond(404)
                    return
                self.respond()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(close)
        self.engine = engine_type(SimpleNamespace(), 0)
        self.engine.node_rank = 0
        self.engine.server_host, self.engine.server_port = server.server_address

    def test_unprotected_flush_then_release_reproduces_scheduler_failure(self):
        self.engine.flush_cache()
        with self.assertRaises(requests.HTTPError):
            self.engine._make_request("release_memory_occupation")
        self.assertTrue(self.state.crashed)

    def test_offload_blocks_probe_in_the_flush_release_window_and_can_repeat(self):
        for _ in range(3):
            self.engine.release_memory_occupation()
            self.assertTrue(self.state.offloaded)
            self.assertFalse(self.state.paused)
            self.assertFalse(self.state.crashed)
            self.assertFalse(self.state.pending_health)
            self.engine.resume_memory_occupation()
            self.assertFalse(self.state.offloaded)
        self.assertEqual(self.state.health_probes, 3)

    def test_offload_waits_for_scheduler_to_drain(self):
        self.state.busy_flushes = 2
        self.engine.release_memory_occupation()
        self.assertTrue(self.state.offloaded)
        self.assertEqual(self.clock.sleep.call_count, 2)
        self.assertFalse(self.state.paused)

    def test_failed_release_unpauses_generation(self):
        self.state.fail_release = True
        with self.assertRaisesRegex(requests.HTTPError, "/release_memory_occupation"):
            self.engine.release_memory_occupation()
        self.assertFalse(self.state.paused)

    def test_failed_flush_unpauses_generation_without_releasing_memory(self):
        with patch.object(self.engine, "flush_cache", side_effect=TimeoutError("busy")):
            with self.assertRaisesRegex(TimeoutError, "busy"):
                self.engine.release_memory_occupation()
        self.assertFalse(self.state.paused)
        self.assertFalse(self.state.offloaded)

    def test_busy_flush_has_a_deadline_and_unpauses_on_timeout(self):
        self.state.busy_flushes = 1
        self.clock.monotonic.side_effect = [0.0, 0.0, 60.0, 60.0]
        with self.assertRaisesRegex(TimeoutError, "flushing cache"):
            self.engine.release_memory_occupation()
        self.assertFalse(self.state.paused)
        self.assertFalse(self.state.offloaded)

    def test_flush_retries_transport_timeout_before_offloading(self):
        get = requests.get

        def flaky_get(*args, **kwargs):
            if mocked.call_count == 1:
                raise requests.Timeout("slow")
            return get(*args, **kwargs)

        with patch.object(requests, "get", side_effect=flaky_get) as mocked:
            self.engine.release_memory_occupation()
        self.assertTrue(self.state.offloaded)
        self.assertFalse(self.state.paused)
        self.assertEqual(self.clock.sleep.call_count, 1)

    def test_cleanup_failure_preserves_original_error(self):
        self.state.fail_release = self.state.fail_continue = True
        with self.assertLogs(__name__, level="ERROR"):
            with self.assertRaisesRegex(requests.HTTPError, "/release_memory_occupation"):
                self.engine.release_memory_occupation()

    def test_nonzero_node_rank_does_not_call_the_http_server(self):
        self.engine.node_rank = 1
        with patch.object(requests, "get") as get, patch.object(requests, "post") as post:
            self.engine.release_memory_occupation()
        get.assert_not_called()
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
