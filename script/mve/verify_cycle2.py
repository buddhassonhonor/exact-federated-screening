"""Verify frozen policies, native streams, recorded ranks and TCP frame totals."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import zlib

import numpy as np

from exact_tracking import independent_rank
from fetch_native import SOURCES
from run_exact_study import RUNS, ROOT, choose_bin
from screen_protocol import HEADER
from stream_data import digest


def lookup(run, filename):
    return RUNS / run / "artifacts" / filename


def check_native(dev, native, data_root):
    source = lookup(dev, "policy.json")
    target = lookup(native, "frozen_policy.json")
    assert digest(source) == digest(target)
    fixed = json.loads(source.read_text(encoding="utf-8"))
    assert fixed["development_metrics_sha256"] == digest(RUNS / dev / "metrics.jsonl")
    given = json.loads(lookup(native, "sha256.json").read_text(encoding="utf-8"))
    total = 0
    for name in ("beijing", "load"):
        provenance = json.loads(lookup(native, f"source__{name}.json").read_text(encoding="utf-8"))
        meta, baseline = provenance, provenance["source"]
        x = ROOT / "data" / "derived" / "native_v1" / f"{name}.npz"
        assert digest(x) == meta["derived_sha256"]
        with np.load(x, allow_pickle=False) as files:
            features, labels, dates = files["features"], files["labels"], files["timestamp"]
            assert features.shape == (*labels.shape, meta["d"])
            assert len(dates) == meta["hours"] and features.shape[1] == meta["clients"]
            assert np.all(np.diff(dates) == np.timedelta64(1, "h"))
            expected_start = meta["calibration_hours"]
            blocks = []
            for block, span in enumerate(meta["periods"]):
                assert expected_start <= span["start"] < span["stop"]
                expected_start = span["stop"]
                blocks.append(labels[span["start"]:span["stop"]])
                filename = f"{name}__block{block}.npz"
                frozen = lookup(native, filename)
                assert given[filename] == digest(frozen)
                with np.load(frozen, allow_pickle=False) as scored:
                    assert scored["start"] == span["start"] and scored["stop"] == span["stop"]
                    for scores, rank in zip(scored["scores"], scored["exact_rank"], strict=True):
                        assert np.array_equal(independent_rank(scores, len(rank)), rank)
                        total += 1
            combined = np.concatenate(blocks)
            assert np.sum(combined >= 0) == meta["rows_with_label"]
            assert np.isclose(np.mean(combined[combined >= 0] == 1), meta["positive_fraction"])
        assert baseline["sha256"] == digest(data_root / SOURCES[name][0] / baseline["filename"])
        assert fixed["selection"][choose_bin(meta["d"])] == "hybrid-25"
    return total


def check_network(run):
    reference = {}
    frame_count = 0
    with gzip.open(lookup(run, "frame_traces.jsonl.gz"), "rt", encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            field = (entry["case"], entry["method"], entry["round"])
            expected = entry["reference"]
            counts = Counter()
            waves = set()
            for frame in entry["frames"]:
                buffer = bytes.fromhex(frame["bytes"])
                _, flag, length = HEADER.unpack_from(buffer)
                assert len(buffer) - HEADER.size == length
                if flag:
                    zlib.decompress(buffer[HEADER.size:])
                counts[frame["direction"]] += len(buffer)
                waves.add((frame["direction"], frame["phase"]))
                frame_count += 1
            assert counts["up"] == expected["up"]
            assert counts["down"] == expected["down"] + expected["selection_bytes"]
            assert len(waves) == expected["phases_with_selection"]
            reference[field] = sum(counts.values()), len(waves)
    observed = 0
    with (RUNS / run / "metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            field = (row["case"], row["method"], row["round"])
            assert reference[field] == (row["app_bytes"], row["waves"])
            assert row["elapsed_ms"] > 0
            observed += 1
    return observed, frame_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--tcp", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    args = parser.parse_args()
    source = "220 native rank rounds"
    native = check_native(args.dev, args.native, args.data_root)
    result = {"verified_native_rank_rounds": native, "verified_policy_sha256": digest(lookup(args.dev, "policy.json")),
              "verification_scope": source, "tcp": {run: check_network(run) for run in args.tcp}}
    print(json.dumps(result, indent=2))
    (ROOT / "result" / "cycle2_verified.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
