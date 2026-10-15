"""CPU regressions for rollout placement with reserved actor/teacher GPUs."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

from slime.utils.http_utils import _wrap_ipv6


ROOT = Path(__file__).resolve().parents[1]


def load_allocator():
    # Importing rollout.py initializes GPU training dependencies. Execute its
    # actual allocator with synchronous Ray handles for this CPU placement test.
    tree = ast.parse((ROOT / "slime/ray/rollout.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_allocate_rollout_engine_addr_and_ports_normal")
    namespace = dict(ray=SimpleNamespace(get=lambda value: value),
                     logger=logging.getLogger(__name__), _wrap_ipv6=_wrap_ipv6)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "rollout.py", "exec"), namespace)
    return namespace[function.name]


class Engine:
    def __init__(self, host, occupied=()):
        self.host = host
        self.occupied = set(occupied)
        self.queries = []
        self._get_current_node_ip_and_free_port = SimpleNamespace(remote=self.free_port)

    def free_port(self, start_port=10000, consecutive=1):
        port = start_port
        while self.occupied.intersection(range(port, port + consecutive)):
            port += 1
        self.queries.append((port, consecutive))
        return self.host, port


class RolloutAddressTests(unittest.TestCase):
    def setUp(self):
        self.allocate = load_allocator()
        self.args = SimpleNamespace(num_gpus_per_node=4, rollout_num_gpus=4,
                                    rollout_num_gpus_per_engine=1, prefill_num_servers=None,
                                    sglang_dp_size=1)

    def allocate_hosts(self, hosts, ranks=None, occupied=()):
        ranks = range(len(hosts)) if ranks is None else ranks
        engines = [(rank, Engine(hosts[rank], occupied)) for rank in ranks]
        result = self.allocate(args=self.args, num_engines=len(hosts), rollout_engines=engines)
        for rank, engine in engines:
            self.assertEqual(result[rank]["host"], engine.host)
        # Check complete reserved port ranges, not only their starting ports.
        reserved = {}
        for _, engine in engines:
            used = reserved.setdefault(engine.host, set())
            for port, size in engine.queries:
                if port < 15000:  # The initial IP-only lookup does not reserve a port.
                    continue
                requested = set(range(port, port + size))
                self.assertFalse(used.intersection(requested))
                used.update(requested)
        return result

    def test_two_actor_and_two_teacher_gpus_leave_two_engines_on_each_node(self):
        hosts = ["10.224.3.20"] * 2 + ["10.224.3.44"] * 2
        result = self.allocate_hosts(hosts)
        for rank, host in enumerate(hosts):
            self.assertTrue(result[rank]["dist_init_addr"].startswith(host + ":"))

    def test_partially_used_first_and_last_nodes(self):
        hosts = ["10.0.0.1"] + ["10.0.0.2"] * 4 + ["10.0.0.3"]
        self.args.rollout_num_gpus = len(hosts)
        result = self.allocate_hosts(hosts, occupied=range(15000, 15005))
        for rank, host in enumerate(hosts):
            self.assertTrue(result[rank]["dist_init_addr"].startswith(host + ":"))
            self.assertGreaterEqual(result[rank]["port"], 15005)

    def test_colocated_full_nodes(self):
        hosts = ["10.0.0.1"] * 4 + ["10.0.0.2"] * 4
        self.args.rollout_num_gpus = len(hosts)
        result = self.allocate_hosts(hosts)
        for rank, host in enumerate(hosts):
            self.assertTrue(result[rank]["dist_init_addr"].startswith(host + ":"))

    def test_restart_allocates_only_requested_engines(self):
        hosts = ["10.0.0.1"] * 2 + ["10.0.0.2"] * 2
        result = self.allocate_hosts(hosts, ranks=[1, 3])
        self.assertEqual(result[0], {})
        self.assertEqual(result[2], {})
        self.assertTrue(result[1]["dist_init_addr"].startswith(hosts[1] + ":"))
        self.assertTrue(result[3]["dist_init_addr"].startswith(hosts[3] + ":"))

    def test_multi_node_tp_shares_only_its_first_node_rendezvous(self):
        hosts = [f"10.0.0.{n}" for n in range(1, 5)]
        self.args.rollout_num_gpus = 16
        self.args.rollout_num_gpus_per_engine = 8
        result = self.allocate_hosts(hosts)
        for first in (0, 2):
            address = result[first]["dist_init_addr"]
            self.assertTrue(address.startswith(hosts[first] + ":"))
            self.assertEqual(result[first + 1]["dist_init_addr"], address)
        self.assertNotEqual(result[0]["dist_init_addr"], result[2]["dist_init_addr"])

    def test_prefill_bootstrap_ports_and_ipv6_hosts(self):
        hosts = ["2001:db8::1"] * 2 + ["2001:db8::2"] * 2
        self.args.prefill_num_servers = 2
        result = self.allocate_hosts(hosts)
        for rank, host in enumerate(hosts):
            self.assertTrue(result[rank]["dist_init_addr"].startswith(f"[{host}]:"))
            self.assertEqual("disaggregation_bootstrap_port" in result[rank], rank < 2)


if __name__ == "__main__":
    unittest.main()
