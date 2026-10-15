"""Teacher scoring service which caches response-aligned final hidden states.

    python -m slime.rollout.opd_teacher --model-path ... --cache-dir ... --port ...

The frozen output head + hidden states reconstruct every next-token logit without
writing an L x vocabulary matrix. Artifacts must live on the shared training filesystem.
Only teacher scoring uses this service; student generation continues to use SGLang.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import uuid

import torch
from aiohttp import web


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def model_identity(model_path):
    path = Path(model_path).resolve()
    files = sorted(p for pattern in ("*.json", "*.safetensors", "*.bin") for p in path.glob(pattern))
    signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps([str(path), signature]).encode()).hexdigest()[:20]


class TeacherScorer:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("The managed distribution teacher needs a CUDA GPU.")
        self.chunk_size = args.chunk_size
        self.context_length = args.context_length
        self.root = Path(args.cache_dir).resolve() / model_identity(args.model_path)
        self.root.mkdir(parents=True, exist_ok=True)
        max_memory = {
            index: int(torch.cuda.get_device_properties(index).total_memory * args.memory_fraction)
            for index in range(torch.cuda.device_count())
        }
        # Prevent auto placement from silently moving decoder layers to CPU/disk.
        max_memory["cpu"] = 0
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto", max_memory=max_memory,
            local_files_only=True, trust_remote_code=True, attn_implementation="sdpa",
        ).eval()
        self.model.requires_grad_(False)
        self.head = self.model.get_output_embeddings()
        self.head_path = self.root / "output_head.pt"
        if not self.head_path.exists():
            atomic_save({"weight": self.head.weight.detach().cpu(),
                         "bias": None if self.head.bias is None else self.head.bias.detach().cpu()}, self.head_path)
        print(f"Distribution teacher ready; shared feature cache: {self.root}", flush=True)

    @torch.inference_mode()
    def score(self, payload):
        ids = payload["input_ids"]
        response_length = payload.get("opd_response_length")
        if type(response_length) is not int or not 1 <= response_length < len(ids):
            raise ValueError("opd_response_length must identify a nonempty response after a nonempty prompt.")
        if len(ids) > self.context_length:
            raise ValueError("Teacher request exceeds teacher_context_length.")
        if payload.get("sampling_params", {}).get("max_new_tokens", 0) != 0 or payload.get("image_data"):
            raise ValueError("The feature teacher supports text-only scoring, not generation.")
        key = hashlib.sha256(json.dumps([ids, response_length]).encode()).hexdigest()
        feature_path = self.root / key[:2] / f"{key}.pt"
        if feature_path.exists():
            artifact = torch.load(feature_path, map_location="cpu", weights_only=True, mmap=True)
            response_logprobs = artifact["sample_log_probs"].tolist()
        else:
            input_device = self.model.get_input_embeddings().weight.device
            tokens = torch.tensor([ids], device=input_device, dtype=torch.long)
            hidden = self.model.base_model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
            # Position before y_t predicts y_t; do not use the distribution after y_t.
            response_hidden = hidden[0, len(ids) - response_length - 1 : len(ids) - 1]
            values = []
            for start in range(0, response_length, self.chunk_size):
                chunk = response_hidden[start : start + self.chunk_size].to(self.head.weight.device)
                logits = self.head(chunk).float()
                targets = tokens[0, len(ids) - response_length + start : len(ids) - response_length + start + len(chunk)]
                values.append(logits.log_softmax(-1).gather(-1, targets.to(logits.device).unsqueeze(-1)).squeeze(-1).cpu())
            sample_log_probs = torch.cat(values)
            if not torch.isfinite(sample_log_probs).all():
                raise ValueError("Teacher returned non-finite sampled log probabilities.")
            atomic_save({"hidden_states": response_hidden.detach().cpu().contiguous(),
                         "response_tokens": torch.tensor(ids[-response_length:], dtype=torch.long),
                         "sample_log_probs": sample_log_probs}, feature_path)
            response_logprobs = sample_log_probs.tolist()
        prefix = [[None, token, ""] for token in ids[:-response_length]]
        scored = [[lp, token, ""] for lp, token in zip(response_logprobs, ids[-response_length:], strict=True)]
        return {"meta_info": {"input_token_logprobs": prefix + scored, "opd_teacher_features": {
            "path": str(feature_path), "head_path": str(self.head_path),
            "vocab_size": self.head.weight.shape[0], "response_length": response_length,
        }}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    args = parser.parse_args()
    scorer = TeacherScorer(args)
    lock = asyncio.Lock()

    async def health(request):
        return web.json_response({"status": "ready"})

    async def score(request):
        payload = await request.json()
        async with lock:
            try:
                return web.json_response(await asyncio.to_thread(scorer.score, payload))
            except (ValueError, KeyError) as error:
                raise web.HTTPBadRequest(text=str(error)) from error

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health_generate", health)
    app.router.add_get("/health", health)
    app.router.add_post("/generate", score)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
