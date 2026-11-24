"""Run paired streaming experiments and retain reproduction artifacts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import difflib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import traceback
from collections import deque

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.special import expit

from screen_protocol import Protocol, Wire, request
from stream_data import HOLDOUT, OPENML, REAL, SYNTHETIC, build_public, build_synthetic, digest, public_data

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (7, 13, 19, 29, 41)
EXACT = {"dense", "lossless", "delta", "dense-best", "tput", "bound", "shuffled", "hybrid", "absolute"}


def local_histogram(x, y):
    n1 = int(y.sum())
    return np.stack((x[y == 0].sum(axis=0), x[y == 1].sum(axis=0))), np.array([len(y) - n1, n1])


def score(histogram, counts):
    rates = (histogram + 2) / (counts[:, :, None] + 4)
    strength = abs(rates[:, 1] - rates[:, 0])
    return np.rint(2048 * strength).astype(np.int16)


def fit(histogram, counts, features, previous):
    wire = Wire()
    n = np.zeros(2, dtype=np.int64)
    x = np.zeros((2, len(features)), dtype=np.int64)
    all_features = np.arange(histogram.shape[-1])
    reused = np.array_equal(features, previous)
    for local, size in zip(histogram, counts):
        if reused:
            wire.send("down", 12, b"")
            ids = previous
        else:
            ids = request(wire, features, len(all_features))
        packet = np.concatenate([size, local[:, ids].ravel()]).astype("<i4").tobytes()
        received = np.frombuffer(wire.send("up", 13, packet, True), dtype="<i4")
        n += received[:2]
        x += received[2:].reshape(2, -1)
    cond = (x + 1) / (n[:, None] + 2)
    odds = (n[1] + 1) / (n[0] + 1)
    log_yes = np.log(cond[1] / cond[0])
    log_no = np.log((1 - cond[1]) / (1 - cond[0]))
    weights = np.r_[np.log(odds) + log_no.sum(), log_yes - log_no]
    # Orientation and predictions of the best Bernoulli stump are sent too.
    majority_error = min(n)
    errors = np.minimum(x[0], x[1]) + np.minimum(n[0] - x[0], n[1] - x[1])
    j = int(np.argmin(errors))
    stump = np.r_[j, (n[1] - x[1, j] + 1) / (n.sum() - x[:, j].sum() + 2),
                  (x[1, j] + 1) / (x[:, j].sum() + 2), majority_error - errors[j]]
    packet = np.r_[weights, stump].astype("<f4").tobytes()
    for _ in histogram:
        model = np.frombuffer(wire.send("down", 14, packet, True), dtype="<f4")
    return (features, model[:-4], model[-4:]), wire.up, wire.down


def evaluate(data, model):
    result = np.zeros((len(data), 7), dtype=float)
    if model is None:
        return result
    ids, weights, stump = model
    for j, (x, y) in enumerate(data):
        if not len(y):
            continue
        probability = expit(x[:, ids] @ weights[1:] + weights[0])
        positive = probability >= 0.5
        stump_positive = np.where(x[:, ids[int(stump[0])]], stump[2], stump[1]) >= 0.5
        result[j] = (len(y), y.sum(), (positive & (y == 1)).sum(),
                     ((~positive) & (y == 0)).sum(), np.square(probability - y).sum(),
                     (stump_positive == y).sum(), (positive == y).sum())
    return result


def settings(d, k, q, groups, followup=False, adaptive=False):
    configs = [
        ("dense", "dense", 0, 1), ("lossless", "lossless", 0, 1), ("delta", "delta", 0, 1),
        ("tput", "tput", 0, 1), ("bound", "bound", q, groups),
        ("one-group", "bound", q, 1), ("no-sparse", "bound", 0, groups),
        ("shuffled", "shuffled", q, groups), ("random", "random", 0, 1),
        ("periodic-2", "periodic", 2, 1), ("periodic-4", "periodic", 4, 1),
        ("periodic-8", "periodic", 8, 1), ("local-k", "sparse", k, 1),
        ("local-2k", "sparse", 2 * k, 1), ("local-4k", "sparse", 4 * k, 1),
        ("local-1", "sparse", 1, 1), ("local-quarter", "sparse", max(1, d // 4), 1),
    ]
    if followup:
        configs.extend((f"q{v}", "bound", v, groups) for v in (0, 1, 4, 12))
        configs.extend((f"g{v}", "bound", q, v) for v in (1, 2, 4, 8))
        configs.extend((f"periodic-{v}", "periodic", v, 1) for v in (3, 5, 6, 10, 12, 16))
        configs.extend((f"local-{v}", "sparse", v, 1) for v in (2, 3, 5, 6, 7, 10, 12, 16, 24))
    if adaptive:
        configs.extend([("hybrid", "hybrid", q, groups), ("absolute", "absolute", k, 1),
                        ("dense-best", "dense-best", 0, 1)])
    return configs


def data_for_run(name, run):
    config = json.loads((run / "config.yaml").read_text(encoding="utf-8"))
    cleaned = config["stage"] == "revised"
    folder = ROOT / "data" / "derived" / ("streams_v2" if cleaned else "streams_v1")
    return public_data(name, Path(config["data_root"]), folder, cleaned=cleaned)


def run_job(job):
    name, scenario, seed, clients, d, rounds, n, alpha, window, k_hint, q, groups, variant, run = job
    run = Path(run)
    if name in SYNTHETIC:
        iterator = build_synthetic(name, rounds, clients, d, n, seed, alpha)
    else:
        x, y, client_groups, metadata = data_for_run(name, run)
        d = x.shape[1]
        iterator = build_public(x, y, client_groups, metadata["warm"], rounds, clients, seed, scenario, alpha)
    k = min(k_hint, d - 1)
    key = f"{name}__{scenario}__{variant}__s{seed}"
    adaptive = variant in {"revised", "natural-airport", "chronological-holdout"}
    cfs = settings(d, k, q, groups, not variant.startswith(("main", "pilot")), adaptive=adaptive)
    transports = [Protocol(proto, clients, d, k, param, g, seed) for _, proto, param, g in cfs]
    models = [None] * len(cfs)
    buffer = deque(maxlen=window)
    rows = []
    score_history, exact_history, drift_history = [], [], []
    selected_previous = [None] * len(cfs)
    for t, (data, drift) in enumerate(iterator):
        buffer.append(tuple(np.array(v) for v in zip(*(local_histogram(x, y) for x, y in data))))
        history = tuple(zip(*buffer))
        histogram, counts = (np.sum(np.stack(items), axis=0) for items in history)
        scores = score(histogram, counts)
        score_history.append(scores)
        drift_history.append(drift)
        base = np.sort(np.argsort(-scores.sum(axis=0), kind="stable")[:k])
        exact_history.append(base)
        predictions = {}
        fitted = {}
        for j, ((method, kind, setting, g), transport) in enumerate(zip(cfs, transports)):
            start = time.perf_counter_ns()
            outcome = transport.step(scores)
            runtime = time.perf_counter_ns() - start
            selected = outcome.selected
            match = np.array_equal(selected, base)
            if kind in EXACT and not match:
                raise AssertionError((key, t, method, selected, base))
            previous = selected_previous[j]
            model_key = (tuple(selected), None if previous is None else tuple(previous))
            if model_key not in fitted:
                fitted[model_key] = fit(histogram, counts, selected, previous)
            model, fit_up, fit_down = fitted[model_key]
            prediction_key = None if previous is None else tuple(previous)
            if prediction_key not in predictions:
                predictions[prediction_key] = evaluate(data, models[j])
            counts_by_client = predictions[prediction_key]
            models[j] = model
            selected_previous[j] = selected
            size, positives, tp, tn, brier_sum, stump_correct, correct = counts_by_client.sum(axis=0)
            rows.append(dict(dataset=name, scenario=scenario, seed=seed, variant=variant, method=method, round=t,
                             d=d, clients=clients, k=k, window=window, alpha=alpha, q=q, groups=groups, drift=drift,
                             up=outcome.up, down=outcome.down, fit_up=fit_up, fit_down=fit_down,
                             exact=int(match), overlap=len(np.intersect1d(selected, base)), extra=outcome.extra,
                             candidates=outcome.candidates, fallback=int(outcome.fallback), gap=outcome.gap,
                             dense_clients=outcome.dense_clients,
                             runtime_ns=runtime, n=int(size), positive=int(positives), tp=int(tp), tn=int(tn),
                             brier=float(brier_sum), stump=int(stump_correct), correct=int(correct),
                             client_n=counts_by_client[:, 0].astype(int).tolist(),
                             client_correct=counts_by_client[:, -1].astype(int).tolist()))
    np.savez_compressed(run / "artifacts" / f"{key}.npz", scores=score_history, exact_rank=exact_history, drift=drift_history)
    return key, rows


def jobs(stage, seeds, run):
    if stage == "pilot":
        for name in ("stationary", "abrupt", "gradual", "ties"):
            for d in (32, 256):
                for seed in seeds:
                    yield name, "synthetic", seed, 4, d, 24, 160, 1.0, 3, 4, 2, 4, f"pilot-d{d}", run
        return
    if stage in {"main", "revised"}:
        for name in SYNTHETIC:
            for seed in seeds:
                yield name, "synthetic", seed, 6, 256, 36, 192, 1.0, 3, 4, 2, 4, stage, run
        cases = [(n, "provider-order") for n in REAL]
        cases.extend((f"openml-{did}", "perturbed") for did in OPENML)
        for name, scenario in cases:
            for seed in seeds:
                yield name, scenario, seed, 6, 0, 30, 0, 1.0, 3, 4, 2, 4, stage, run
        if stage == "revised":
            for seed in seeds:
                yield "Airlines", "natural-groups", seed, 6, 0, 30, 0, 1.0, 3, 4, 2, 4, "natural-airport", run
                yield HOLDOUT, "chronological", seed, 6, 0, 30, 0, 1.0, 3, 4, 2, 4, "chronological-holdout", run
        return
    if stage == "network":
        for name in ("Airlines",):
            for seed in seeds:
                yield name, "natural-groups", seed, 6, 0, 30, 0, 1.0, 3, 4, 2, 4, "natural-airport", run
        return
    for seed in seeds:
        for d in (32, 128, 512, 2048):
            yield "gradual", "synthetic", seed, 6, d, 32, 192, 1.0, 3, 4, 2, 4, f"width-{d}", run
        for c in (2, 4, 8, 16):
            yield "gradual", "synthetic", seed, c, 256, 32, 192, 1.0, 3, 4, 2, 4, f"clients-{c}", run
        for k in (2, 4, 8, 16):
            yield "recurring", "synthetic", seed, 6, 256, 32, 192, 1.0, 3, k, 2, 4, f"select-{k}", run
        for w in (1, 2, 4, 8):
            yield "abrupt", "synthetic", seed, 6, 256, 32, 192, 1.0, w, 4, 2, 4, f"memory-{w}", run
        for a in (0.1, 0.3, 1.0, 3.0):
            yield "Airlines", "provider-order", seed, 6, 0, 30, 0, a, 3, 4, 2, 4, f"alpha-{a}", run


def setup(run, args, matrix):
    for name in ("artifacts", "plots", "source"):
        (run / name).mkdir(parents=True, exist_ok=False)
    config = dict(stage=args.stage, seeds=args.seeds, workers=args.workers, data_root=str(args.data_root.resolve()), key_order=
                  "dataset scenario seed clients d rounds n alpha window k q groups variant run".split(),
                  jobs=[[str(x) if isinstance(x, Path) else x for x in job] for job in matrix])
    (run / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = shlex.join([sys.executable.replace("\\", "/"), *sys.argv])
    (run / "commands.sh").write_text(f"#!/usr/bin/env bash\n{command}\n", encoding="utf-8")
    (run / "commands.json").write_text(json.dumps([sys.executable, *sys.argv], indent=2) + "\n", encoding="utf-8")
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               processor=platform.processor(), logical_cpus=os.cpu_count(), blas_threads=1,
               hardware="Intel Core Ultra 9 275HX", backend="CPU NumPy",
               versions={x: importlib.metadata.version(x) for x in ("numpy", "scipy", "pandas", "pyarrow")},
               git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (run / "artifacts" / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    patches = []
    for source in (Path(__file__), Path(__file__).with_name("screen_protocol.py"),
                   Path(__file__).with_name("stream_data.py"), Path(__file__).with_name("fetch_streams.py")):
        original = source.read_bytes()
        (run / "source" / source.name).write_bytes(original)
        relative = source.resolve().relative_to(ROOT).as_posix()
        patches.extend(difflib.unified_diff([], original.decode("utf-8-sig").splitlines(keepends=True),
                                           fromfile="/dev/null", tofile=f"b/{relative}"))
    (run / "diff.patch").write_text("".join(patches), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "main", "followup", "network", "revised"))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    args = parser.parse_args()
    run = ROOT / "result" / "runs" / args.slug
    if run.exists():
        raise FileExistsError(run)
    matrix = list(jobs(args.stage, args.seeds, run))
    setup(run, args, matrix)
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        def output(msg):
            print(msg, flush=True)
            print(msg, file=log, flush=True)

        start = time.perf_counter()
        output(f"{datetime.now(timezone.utc).isoformat()} started {len(matrix)} paired jobs")
        try:
            for name in dict.fromkeys(job[0] for job in matrix if job[0] not in SYNTHETIC):
                _, _, _, metadata = data_for_run(name, run)
                (run / "artifacts" / f"source__{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            with ProcessPoolExecutor(max_workers=args.workers) as pool, (
                run / "metrics.jsonl"
            ).open("w", encoding="utf-8") as metrics:
                for key, rows in pool.map(run_job, matrix):
                    for row in rows:
                        print(json.dumps(row, separators=(",", ":")), file=metrics)
                    metrics.flush()
                    output(f"{key}: {len(rows)} protocol-round measurements")
            manifest = {str(p.relative_to(run)): digest(p) for p in sorted((run / "artifacts").glob("*.npz"))}
            (run / "artifacts" / "sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            output(f"Elapsed {time.perf_counter() - start:.2f} seconds")
        except BaseException:
            output(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
