"""Prepare the README's seeded 300K OpenThoughts3 prompts as Parquet."""

import argparse
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import time
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from prepare_sft_prompts import extract_prompt

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download_file(cfg, entry):
    destination = Path(cfg["source_data_dir"]) / entry["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = entry.get("lfs", {}).get("oid")
    if destination.exists() and destination.stat().st_size == entry["size"]:
        return destination
    temporary = destination.with_name(destination.name + ".partial")
    url = (f'https://huggingface.co/datasets/{cfg["source_dataset"]}/resolve/'
           f'{cfg["source_revision"]}/{entry["path"]}')
    for attempt in range(5):
        try:
            digest = hashlib.sha256()
            # A retry replaces only this incomplete download, never a ready file.
            with urlopen(url, timeout=180) as source, temporary.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
            if temporary.stat().st_size != entry["size"]:
                raise ValueError(f"Incomplete download: {entry['path']}")
            if expected_hash and digest.hexdigest() != expected_hash:
                raise ValueError(f"SHA256 mismatch: {entry['path']}")
            temporary.replace(destination)
            return destination
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def extract_file(work):
    path, file_offset, selected_indices = work
    rows = []
    offset = file_offset
    cursor = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["conversations"]):
        end = bisect_left(selected_indices, offset + batch.num_rows, lo=cursor)
        chosen = selected_indices[cursor:end]
        if chosen:
            # Discard unselected rows before converting large reasoning answers
            # to Python objects. Retain only the prompt from selected records.
            selected = batch.take(pa.array([index - offset for index in chosen], type=pa.int64()))
            for global_index, sample in zip(chosen, selected.to_pylist()):
                prompt = extract_prompt(sample)
                if not prompt or not prompt["prompt"]:
                    raise ValueError(f"Missing prompt at source row {global_index}")
                rows.append({"sample_id": global_index, **prompt})
        cursor = end
        offset += batch.num_rows
    return rows


def prepare(cfg):
    output = Path(cfg["prompts_file"])
    manifest = output.with_suffix(".manifest.json")
    identity = {key: cfg[key] for key in (
        "source_dataset", "source_revision", "num_sft_prompts", "seed")}
    if output.exists() and manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Prompt provenance changed; choose a new prompts_file: {output}")
        if pq.ParquetFile(output).metadata.num_rows == cfg["num_sft_prompts"]:
            print(f"Reusing {cfg['num_sft_prompts']} prompts: {output}", flush=True)
            return
        raise ValueError(f"Incomplete prompt Parquet: {output}")
    source_dir = Path(cfg["source_data_dir"])
    source_dir.mkdir(parents=True, exist_ok=True)
    listing_path = source_dir / "source_files.json"
    if listing_path.exists():
        listing = json.loads(listing_path.read_text())
        if listing["revision"] != cfg["source_revision"]:
            raise ValueError("Source revision changed; choose a new source_data_dir")
        entries = listing["files"]
    else:
        url = (f'https://huggingface.co/api/datasets/{cfg["source_dataset"]}/tree/'
               f'{cfg["source_revision"]}/data?recursive=false&limit=1000')
        with urlopen(url, timeout=120) as response:
            entries = sorted((item for item in json.load(response)
                              if item["path"].endswith(".parquet")), key=lambda item: item["path"])
        write_json(listing_path, {"revision": cfg["source_revision"], "files": entries})
    print(f"Preparing {len(entries)} source Parquet files, "
          f"{sum(item['size'] for item in entries) / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as pool:
        futures = [pool.submit(download_file, cfg, entry) for entry in entries]
        for done, future in enumerate(as_completed(futures), 1):
            path = future.result()
            print(f"Source ready {done}/{len(entries)}: {path.name}", flush=True)
    files = [source_dir / entry["path"] for entry in entries]
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if cfg["num_sft_prompts"] > total:
        raise ValueError("Not enough source samples")
    # Exactly the selection algorithm used in scripts/prepare_sft_prompts.py.
    indices = sorted(random.Random(cfg["seed"]).sample(range(total), cfg["num_sft_prompts"]))
    message_type = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
    schema = pa.schema([("sample_id", pa.int64()), ("prompt", message_type)])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    selected = 0
    offset = 0
    cursor = 0
    work = []
    for path in files:
        count = pq.ParquetFile(path).metadata.num_rows
        end = bisect_left(indices, offset + count, lo=cursor)
        work.append((path, offset, indices[cursor:end]))
        cursor = end
        offset += count
    with ProcessPoolExecutor(max_workers=cfg["prompt_preparation_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            # map preserves source-file order, matching the original recipe.
            for rows in pool.map(extract_file, work, chunksize=1):
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                selected += len(rows)
                print(f"Extracted {selected}/{len(indices)} prompts", flush=True)
    if selected != cfg["num_sft_prompts"]:
        raise ValueError(f"Expected {cfg['num_sft_prompts']} prompts, got {selected}")
    temporary.replace(output)
    identity.update({"source_rows": total, "num_rows": selected,
                     "prompts_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                     "original_assistant_responses": "discarded"})
    write_json(manifest, identity)
    print(f"Ready: {output} ({selected} prompts)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config/qwen3_4b_sft.yaml")
    args = parser.parse_args()
    prepare(yaml.safe_load(args.config.read_text()))
