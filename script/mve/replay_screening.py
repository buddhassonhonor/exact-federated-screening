"""Verify frozen scores, rankings and byte counts in an isolated process."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import difflib
import json
from pathlib import Path
import shlex
import shutil
import statistics
import sys
import time

import numpy as np

from run_screening import EXACT, ROOT, settings
from screen_protocol import Protocol
from stream_data import digest


def edge_cases():
    rng = np.random.default_rng(20261124)
    comparisons = 0
    for clients in (1, 3, 16):
        for d in (2, 9, 257):
            for k in sorted({1, d - 1}):
                for group in (1, 4, 8):
                    kinds = ["lossless", "delta", "dense-best", "tput", "bound", "hybrid", "absolute"]
                    protocols = [Protocol(kind, clients, d, k, min(5, d), group) for kind in kinds]
                    for t in range(24):
                        limit = (1, 2, 3, 2049)[t % 4]
                        x = rng.integers(0, limit, (clients, d), dtype=np.int16)
                        expected = sorted(range(d), key=lambda j: (-sum(int(v) for v in x[:, j]), j))[:k]
                        for protocol in protocols:
                            actual = protocol.step(x)
                            assert actual.selected.tolist() == sorted(expected)
                            if protocol.kind == "hybrid":
                                dense = Protocol("lossless", clients, d, k).step(x)
                                if not actual.extra:
                                    assert actual.up <= dense.up
                                else:
                                    request_max = clients * (7 + (d + 7) // 8)
                                    assert actual.up + actual.down <= 1.75 * dense.up + request_max
                            comparisons += 1
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()
    runs = ROOT / "result" / "runs"
    source, output = runs / args.source, runs / args.slug
    output.mkdir()
    (output / "artifacts").mkdir()
    (output / "source").mkdir()
    (output / "plots").mkdir()
    command = shlex.join([sys.executable.replace("\\", "/"), *sys.argv])
    (output / "commands.sh").write_text("#!/usr/bin/env bash\n" + command + "\n", encoding="utf-8")
    (output / "commands.json").write_text(json.dumps([sys.executable, *sys.argv]), encoding="utf-8")
    (output / "config.yaml").write_text(json.dumps(vars(args) | {"workers": 1, "edge_case_seed": 20261124}, indent=2), encoding="utf-8")
    for name in ("screen_protocol.py", "run_screening.py", "stream_data.py", "fetch_streams.py", "replay_screening.py"):
        p = Path(__file__).with_name(name)
        shutil.copyfile(p, output / "source" / name)
    env = json.loads((source / "artifacts" / "environment.json").read_text(encoding="utf-8"))
    env.update(executable=sys.executable, python=sys.version, workers=1)
    (output / "artifacts" / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    script = Path(__file__)
    changes = difflib.unified_diff([], script.read_text(encoding="utf-8-sig").splitlines(keepends=True),
                                   fromfile="/dev/null", tofile="b/script/mve/replay_screening.py")
    (output / "diff.patch").write_text((source / "diff.patch").read_text(encoding="utf-8") + "".join(changes),
                                      encoding="utf-8")
    data = defaultdict(list)
    fields = ("up", "down", "extra", "candidates", "fallback", "gap", "dense_clients")
    with (source / "metrics.jsonl").open(encoding="utf-8") as metrics:
        for line in metrics:
            row = json.loads(line)
            if row["method"] not in EXACT:
                continue
            key = "__".join((row["dataset"], row["scenario"], row["variant"], "s" + str(row["seed"])))
            data[key, row["method"]].append(tuple(row[x] for x in fields))
    checks = 0
    procedure = []
    fingerprints = json.loads((source / "artifacts" / "sha256.json").read_text(encoding="utf-8"))
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as metrics, (
        output / "stdout_stderr.log"
    ).open("w", encoding="utf-8") as log:
        print(datetime.now(timezone.utc).isoformat(), "Started single-process replay", file=log, flush=True)
        for relative, sha in fingerprints.items():
            path = source / relative
            assert digest(path) == sha
            with np.load(path, allow_pickle=False) as frozen:
                scores, ranks = frozen["scores"], frozen["exact_rank"]
            _, c, d = scores.shape
            k = ranks.shape[1]
            seed = int(path.stem.rsplit("__s", 1)[-1])
            for name, kind, param, groups in settings(d, k, 2, 4, followup=True, adaptive=True):
                if (path.stem, name) not in data:
                    continue
                protocol = Protocol(kind, c, d, k, param, groups, seed)
                elapsed = []
                for row, (current, expected, prior) in enumerate(zip(scores, ranks, data[path.stem, name], strict=True)):
                    now = time.perf_counter_ns()
                    result = protocol.step(current)
                    duration = time.perf_counter_ns() - now
                    assert np.array_equal(result.selected, expected)
                    assert tuple(getattr(result, key) for key in fields) == prior
                    checks += 1
                    if row >= 6:
                        elapsed.append(duration)
                record = {"source": path.stem, "method": name, "rounds": len(scores),
                          "median_screen_ms": statistics.median(elapsed) / 1e6}
                procedure.append(record)
                print(json.dumps(record), file=metrics)
        trials = edge_cases()
        summary = {"source": args.source, "hashes_verified": len(fingerprints), "replayed_protocol_rounds": checks,
                   "boundary_stress_checks": trials, "screen_time_unit": "ms", "processes": 1,
                   "median_of_source_seed_medians_ms": {
                       method: statistics.median(row["median_screen_ms"] for row in procedure if row["method"] == method)
                       for method in sorted({row["method"] for row in procedure})}}
        print(json.dumps(summary), file=log)
        print(json.dumps(summary, indent=2))
    (output / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "artifacts" / "timing.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(procedure[0]))
        writer.writeheader()
        writer.writerows(procedure)


if __name__ == "__main__":
    main()
