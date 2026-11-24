"""Pilot, develop and evaluate exact transports with frozen policies."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import difflib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from time import perf_counter_ns
import traceback

from run_screening import evaluate, fit, local_histogram, score
import numpy as np

from exact_tracking import AnchoredMonitor, BatchedNRA, TraceWire, announce, independent_rank
from native_streams import get_block, prepare_native
from screen_protocol import Protocol
from stream_data import digest

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "result" / "runs"
MAIN = "20261124-1310__screen_revised"
SCALE = "20261124-1303__screen_scale"
CHOICES = ("dense-best", "hybrid-25", "hybrid-50", "hybrid-75", "anchor")
METHODS = ("lossless", "dense-best", "tput", "nra-8", "absolute", "bound", "hybrid-25",
           "hybrid-50", "hybrid-75", "safe-zone", "anchor")
SOURCES = ("beijing", "load")


def make_method(method, clients, d, k, seed=7):
    if method in ("safe-zone", "anchor"):
        return AnchoredMonitor(clients, d, k, safe_zone=method == "safe-zone")
    if method.startswith("nra-"):
        return BatchedNRA(k, block=int(method.split("-")[1]))
    if method.startswith("hybrid-"):
        return Protocol("hybrid", clients, d, k, 2, 4, seed, int(method.split("-")[1]))
    return Protocol(method, clients, d, k, k if method == "absolute" else 2,
                    1 if method == "absolute" else 4, seed)


def choose_bin(d):
    return "small" if d <= 32 else "medium" if d <= 128 else "wide"


def run_trace(name, case, variant, seed, scores, k, methods):
    methods = list(dict.fromkeys(methods))
    c, d = scores.shape[1:]
    protocols = {m: make_method(m, c, d, k, seed) for m in methods}
    previous = None
    output = []
    ranks = []
    for t, raw in enumerate(scores):
        expected = independent_rank(raw, k)
        ranks.append(expected)
        for m, method in protocols.items():
            row, chosen, _ = measure(method, raw, previous, capture=False)
            if not np.array_equal(chosen, expected):
                raise AssertionError((name, case, variant, seed, m, t, chosen, expected))
            output.append(dict(source=name, case=case, variant=variant, unit=seed, round=t, method=m,
                               clients=c, d=d, k=k, **row))
        previous = expected
    return output, np.asarray(ranks)


def measure(method, scores, previous, capture):
    wire = TraceWire(capture=capture)
    start = perf_counter_ns()
    certificate, violations = False, 0
    if isinstance(method, AnchoredMonitor):
        selected, certificate, violations = method.step(scores, wire)
    elif isinstance(method, BatchedNRA):
        selected, _ = method.step(scores, wire)
    else:
        selected = method.step(scores, wire=wire).selected
    cpu = perf_counter_ns() - start
    down = wire.down
    phases = len(wire.waves)
    announce(wire, selected, previous, len(scores), scores.shape[1])
    return dict(up=wire.up, down=down, selection_bytes=wire.down - down,
                phases=phases, phases_with_selection=len(wire.waves),
                messages=wire.messages, ns=cpu, certificate=int(certificate),
                violations=violations), selected, wire.records


def pilot(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for clients in (1, 4, 12):
        for d in (2, 19, 72):
            for k in sorted({1, min(4, d - 1), d - 1}):
                for regime in ("tied", "variable", "oscillating"):
                    limit = {"tied": 3, "variable": 2049, "oscillating": 62}[regime]
                    data = rng.integers(0, limit, (12, clients, d), dtype=np.int16)
                    if regime == "oscillating":
                        data[0::2] = data[0]
                        data[1::4] = limit - 1 - data[0]
                    bunch, _ = run_trace(regime, "fixture", f"c{clients}-d{d}-k{k}", seed, data, k,
                                         (*METHODS, "nra-1", "nra-32"))
                    rows.extend(bunch)
    return rows


def archive(path_text):
    path = Path(path_text)
    source, case, variant, seed_text = path.stem.split("__")
    seed = int(seed_text[1:])
    with np.load(path, allow_pickle=False) as frozen:
        scores, ranks = frozen["scores"], frozen["exact_rank"]
    output, actual = run_trace(source, case, variant, seed, scores, ranks.shape[1], METHODS)
    assert np.array_equal(actual, ranks)
    return output


def native(job):
    name, path_text, meta, block_id, policy, out = job
    with np.load(path_text, allow_pickle=False) as frozen:
        x, y = frozen["features"], frozen["labels"]
    buffers = deque(maxlen=3)
    previous = model = None
    scores, learners = [], []
    for t, values in enumerate(get_block(x, y, meta["periods"][block_id])):
        predicted = evaluate(values, model)
        buffers.append(tuple(np.array(v) for v in zip(*(local_histogram(xs, ys) for xs, ys in values))))
        histogram, counts = (np.sum(np.stack(items), axis=0) for items in zip(*buffers))
        local = score(histogram, counts)
        scores.append(local)
        chosen = independent_rank(local, 4)
        model, fit_up, fit_down = fit(histogram, counts, chosen, previous)
        previous = chosen
        learners.append(dict(learner_up=fit_up, learner_down=fit_down, outcomes=predicted.tolist()))
    output, ranks = run_trace(name, "native", policy, block_id, np.asarray(scores), 4, METHODS)
    for row in output:
        row.update(learners[row["round"]])
        row["frozen_policy"] = int(row["method"] == policy)
    np.savez_compressed(Path(out) / f"{name}__block{block_id}.npz", scores=scores, exact_rank=ranks,
                        start=meta["periods"][block_id]["start"], stop=meta["periods"][block_id]["stop"])
    return output


def archived_files():
    checksums = {}
    for slug in (MAIN, SCALE):
        run = RUNS / slug
        metadata = json.loads((run / "artifacts" / "sha256.json").read_text(encoding="utf-8"))
        for filename, sha in metadata.items():
            if slug == MAIN or "__width-" in filename:
                p = run / filename
                assert digest(p) == sha
                checksums[str(p)] = sha
    return checksums


def freeze(metrics, destination, evidence_hash):
    import pandas as pd
    index = ["source", "case", "variant", "unit", "d"]
    data = pd.read_json(metrics, lines=True)
    data = data[data.method.isin(CHOICES)].copy()
    data["cost"] = data["up"] + data["down"] + data.selection_bytes
    costs = data.groupby([*index, "method"]).cost.sum().unstack()
    ratios = costs.div(costs["dense-best"], axis=0).reset_index()
    ratios["bin"] = ratios.d.map(choose_bin)
    cases = ratios.groupby(["bin", "source", "case", "variant"])[list(CHOICES)].mean()
    table = cases.groupby("bin").mean()
    winners = table.idxmin(axis=1).to_dict()
    policy = {"selection": winners, "development_ratios": table.to_dict("index"),
              "development_cases": cases.reset_index().groupby("bin").size().to_dict(),
              "development_metrics_sha256": evidence_hash, "alternatives": CHOICES,
              "fit_rule": "Mean within case across seeds, then equal-weight mean across cases. First on ties."}
    file = destination / "policy.json"
    file.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return policy


def setup(run, args):
    run.mkdir()
    paths = ("artifacts", "plots", "source")
    for name in paths:
        (run / name).mkdir()
    (run / "config.yaml").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    command = [sys.executable, *sys.argv]
    (run / "commands.sh").write_text("#!/usr/bin/env bash\n" + shlex.join(command) + "\n", encoding="utf-8")
    (run / "commands.json").write_text(json.dumps(command), encoding="utf-8")
    env = dict(python=sys.version, platform=platform.platform(), cpu="Intel Core Ultra 9 275HX",
               git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               versions={module: importlib.metadata.version(module) for module in
                         ("numpy", "scipy", "pandas", "pyarrow", "matplotlib")})
    (run / "artifacts" / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    included = ("exact_tracking.py", "native_streams.py", "fetch_native.py", "screen_protocol.py",
                "run_screening.py", "stream_data.py", "fetch_streams.py", "run_exact_study.py",
                "network_screening.py", "analyze_cycle2.py")
    patches = []
    for name in included:
        source = Path(__file__).with_name(name)
        if source.is_file():
            shutil.copy2(source, run / "source" / name)
            patches.extend(difflib.unified_diff([], source.read_text(encoding="utf-8-sig").splitlines(keepends=True),
                                               fromfile="/dev/null", tofile="b/script/mve/" + name))
    (run / "diff.patch").write_text("".join(patches), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "development", "native"))
    parser.add_argument("--slug", required=True)
    parser.add_argument("--policy-run")
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run = RUNS / args.slug
    setup(run, args)
    artifacts = run / "artifacts"
    log = (run / "stdout_stderr.log").open("w", encoding="utf-8")

    def progress(message):
        print(message, flush=True)
        print(message, file=log, flush=True)

    try:
        if args.stage == "pilot":
            data, fn = (7, 13, 19, 29, 41), pilot
        elif args.stage == "development":
            manifest = archived_files()
            (artifacts / "development_input_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            data, fn = list(manifest), archive
        else:
            if not args.policy_run:
                raise ValueError("Native evaluation requires a frozen development policy")
            source = RUNS / args.policy_run / "artifacts" / "policy.json"
            policy = json.loads(source.read_text(encoding="utf-8"))
            shutil.copy2(source, artifacts / "frozen_policy.json")
            (artifacts / "frozen_policy_sha256.txt").write_text(digest(source) + "\n", encoding="utf-8")
            data, fn = [], native
            for name in SOURCES:
                path, metadata = prepare_native(name, args.data_root, ROOT / "data" / "derived" / "native_v1")
                (artifacts / f"source__{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                mode = policy["selection"][choose_bin(metadata["d"])]
                for block in range(len(metadata["periods"])):
                    data.append((name, str(path), metadata, block, mode, str(artifacts)))
        progress(f"{datetime.now(timezone.utc).isoformat()} {args.stage}: {len(data)} blocks")
        n = 0
        metrics = run / "metrics.jsonl"
        with ProcessPoolExecutor(max_workers=args.workers) as pool, metrics.open("w", encoding="utf-8") as out:
            for i, rows in enumerate(pool.map(fn, data), 1):
                for row in rows:
                    print(json.dumps(row, separators=(",", ":")), file=out)
                n += len(rows)
                if i == len(data) or i % 25 == 0:
                    progress(f"{i}/{len(data)} blocks and {n} exact comparisons")
        if args.stage == "development":
            progress(str(freeze(metrics, artifacts, digest(metrics))))
        hashes = {path.name: digest(path) for path in sorted(artifacts.glob("*.npz"))}
        (artifacts / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        progress(f"{datetime.now(timezone.utc).isoformat()} completed {n} exact comparisons")
    except BaseException:
        progress(traceback.format_exc())
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
