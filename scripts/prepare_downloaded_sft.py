"""Download a pinned teacher dataset, verify its hash, and audit every row."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import pyarrow.parquet as pq
import yaml

from qwen3_sft import write_json, sft_dataset_identity


def prepare(cfg):
    output = Path(cfg["train_files"])
    manifest = output.with_suffix(".manifest.json")
    if output.exists() and manifest.exists():
        sft_dataset_identity(cfg)
        print(f"Reusing validated SFT dataset: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{cfg['sft_dataset_repo']}/resolve/"
           f"{cfg['sft_dataset_revision']}/{cfg['sft_dataset_filename']}")
    expected_size = cfg["sft_dataset_size_bytes"]
    partial = output.with_name(output.name + ".partial")
    if not output.exists():
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == expected_size:
                break
            if offset > expected_size:
                raise ValueError("Partial download is larger than the pinned dataset")
            headers = {"User-Agent": "Lightning-OPD-SFT-data-preparation"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urlopen(Request(url, headers=headers), timeout=120) as source:
                    append = offset and source.status == 206
                    if append and not source.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected download range")
                    done = offset if append else 0
                    last_report = 0
                    with partial.open("ab" if append else "wb") as target:
                        while chunk := source.read(8 * 1024 * 1024):
                            target.write(chunk)
                            done += len(chunk)
                            if time.monotonic() - last_report >= 10:
                                print(f"Download: {done / 1e9:.2f}/{expected_size / 1e9:.2f} GB "
                                      f"({done / expected_size:.1%})", flush=True)
                                last_report = time.monotonic()
                if partial.stat().st_size != expected_size:
                    raise ValueError("Incomplete dataset download")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        candidate = partial
    else:
        candidate = output
    if candidate.stat().st_size != expected_size:
        raise ValueError("Dataset size differs from the pinned Hugging Face file")
    digest = hashlib.sha256()
    with candidate.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != cfg["sft_dataset_sha256"]:
        raise ValueError("Dataset SHA-256 differs from the pinned Hugging Face LFS hash")
    print("Dataset SHA-256 verified; auditing Parquet rows", flush=True)
    parquet = pq.ParquetFile(candidate)
    if parquet.metadata.num_rows != cfg["num_sft_prompts"]:
        raise ValueError("Unexpected dataset row count")
    if set(parquet.schema_arrow.names) != {"messages", "tokens"}:
        raise ValueError("Unexpected dataset columns")
    rows = token_sum = capped = missing_think_close = 0
    last_report = time.monotonic()
    for batch in parquet.iter_batches(batch_size=256, columns=["messages", "tokens"]):
        for item in batch.to_pylist():
            messages, tokens = item["messages"], item["tokens"]
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"Invalid conversation at row {rows}")
            if [message.get("role") for message in messages] != ["user", "assistant"]:
                raise ValueError(f"Invalid roles at row {rows}")
            if any(not isinstance(message.get("content"), str) or not message["content"].strip()
                   for message in messages):
                raise ValueError(f"Empty or invalid message at row {rows}")
            if not isinstance(tokens, int) or not 0 < tokens <= cfg["generation_max_tokens"]:
                raise ValueError(f"Invalid token count at row {rows}")
            rows += 1
            token_sum += tokens
            capped += tokens == cfg["generation_max_tokens"]
            missing_think_close += "</think>" not in messages[-1]["content"]
        if time.monotonic() - last_report >= 10:
            print(f"Audit: {rows}/{parquet.metadata.num_rows} rows", flush=True)
            last_report = time.monotonic()
    audit = {"all_rows_valid": True, "total_assistant_tokens": token_sum,
             "mean_assistant_tokens": token_sum / rows, "capped_responses": capped,
             "responses_without_think_close": missing_think_close}
    identity = {"source": "huggingface", "dataset_repo": cfg["sft_dataset_repo"],
                "dataset_revision": cfg["sft_dataset_revision"],
                "filename": cfg["sft_dataset_filename"], "size_bytes": expected_size,
                "sha256": digest.hexdigest(), "num_rows": rows,
                "teacher_model_path": cfg["teacher_model_path"],
                "teacher_revision": None, "teacher_revision_disclosed": False,
                "audit": audit}
    if candidate != output:
        candidate.replace(output)
    write_json(manifest, identity)
    print(f"Prepared {rows} SFT rows: {output}", flush=True)
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("sft_data_source") != "huggingface":
        raise ValueError("This preparation script requires sft_data_source: huggingface")
    prepare(cfg)


if __name__ == "__main__":
    main()
