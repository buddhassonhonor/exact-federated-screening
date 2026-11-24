"""Evaluate the adaptive first-frame threshold on frozen streams."""

import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd

from screen_protocol import Protocol
from stream_data import digest

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (25, 50, 75, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()
    runs = ROOT / "result" / "runs"
    reference = runs / args.from_run
    output = runs / args.slug
    output.mkdir()
    for subfolder in ("artifacts", "plots", "source"):
        (output / subfolder).mkdir()
    script = Path(__file__)
    files = [script, script.with_name("screen_protocol.py"), script.with_name("stream_data.py"),
             script.with_name("fetch_streams.py")]
    patches = []
    for path in files:
        shutil.copy2(path, output / "source" / path.name)
        patches.extend(difflib.unified_diff([], path.read_text(encoding="utf-8-sig").splitlines(keepends=True),
                                           fromfile="/dev/null", tofile="b/script/mve/" + path.name))
    (output / "diff.patch").write_text("".join(patches), encoding="utf-8")
    (output / "commands.sh").write_text("#!/usr/bin/env bash\n" + shlex.join([sys.executable.replace("\\", "/"), *sys.argv]) + "\n",
                                        encoding="utf-8")
    (output / "commands.json").write_text(json.dumps([sys.executable, *sys.argv]), encoding="utf-8")
    (output / "config.yaml").write_text(json.dumps(vars(args) | {"thresholds": THRESHOLDS}, indent=2), encoding="utf-8")
    shutil.copy2(reference / "artifacts" / "environment.json", output / "artifacts" / "environment.json")
    rows = []
    log = output / "stdout_stderr.log"
    log.write_text(f"{datetime.now(timezone.utc).isoformat()} start\n", encoding="utf-8")
    fingerprints = json.loads((reference / "artifacts" / "sha256.json").read_text(encoding="utf-8"))
    for relative, expected_hash in fingerprints.items():
        path = reference / relative
        assert digest(path) == expected_hash
        dataset, case, _, seed = path.stem.split("__")
        with np.load(path, allow_pickle=False) as src:
            x, expected = src["scores"], src["exact_rank"]
        _, c, d = x.shape
        k = expected.shape[1]
        dense = Protocol("lossless", c, d, k)
        baseline = [dense.step(scores).up for scores in x]
        for threshold in THRESHOLDS:
            transport = Protocol("hybrid", c, d, k, setting=2, groups=4, dense_percent=threshold)
            n_refine, cost, sent_dense = 0, 0, 0
            for t, scores in enumerate(x):
                result = transport.step(scores)
                assert np.array_equal(result.selected, expected[t])
                assert result.up + result.down <= (1 + threshold / 100) * baseline[t] + c * (7 + (d + 7) // 8)
                if t >= 6:
                    n_refine += result.extra
                    cost += result.up + result.down
                    sent_dense += result.dense_clients
            past_warm = len(x) - 6
            rows.append(dict(dataset=dataset, case=case, seed=int(seed[1:]), threshold=threshold, rounds=len(x),
                             ratio=cost / sum(baseline[6:]), extra=n_refine / past_warm, dense=sent_dense / (past_warm * c)))
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            print(json.dumps(row), file=out)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output / "artifacts" / "seed_summary.csv", index=False)
    result = per_seed.groupby(["threshold", "case", "dataset"]).mean(numeric_only=True)
    grid = []
    for threshold in THRESHOLDS:
        r = result.loc[threshold]
        grid.append([threshold, r.loc["perturbed"].ratio.median(), r.loc["perturbed"].extra.median(),
                     r.loc[("natural-groups", "Airlines"), "ratio"], r.loc[("chronological", "Occupancy"), "ratio"],
                     r.loc[("synthetic", "ties"), "ratio"]])
    pd.DataFrame(grid, columns=["threshold", "public_ratio", "public_extra", "airports", "occupancy", "ties"]).to_csv(
        output / "artifacts" / "grid.csv", index=False)
    checks = int(per_seed.rounds.sum())
    with log.open("a", encoding="utf-8") as out:
        print(f"{checks} exact decisions and byte-bound checks verified", file=out)
    print(checks, grid)


if __name__ == "__main__":
    main()
