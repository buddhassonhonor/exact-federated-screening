"""Summarize exact tracking, held-out native streams and TCP evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_exact_study import RUNS, ROOT
from stream_data import digest

FIGURES = ROOT / "figures"
DISPLAY = {"lossless": "Dense zlib", "dense-best": "Best dense", "tput": "TPUT", "nra-8": "Batched NRA",
           "absolute": "Absolute envelope", "bound": "Grouped residuals", "hybrid-25": "Frozen policy",
           "hybrid-50": "Hybrid 50", "hybrid-75": "Hybrid 75", "safe-zone": "Safe zone", "anchor": "Anchored bound"}
LABEL = {"beijing": "Beijing stations", "load": "Electricity meters", "wide-gradual": "Wide gradual drift"}
MAIN_METHODS = ("dense-best", "tput", "nra-8", "absolute", "bound", "hybrid-25", "safe-zone", "anchor")
TCP_METHODS = ("dense-best", "tput", "nra-8", "hybrid-25", "safe-zone", "anchor")
PROFILES = ("loopback", "2ms-100Mbps", "2ms-1Mbps", "20ms-100Mbps", "20ms-1Mbps")
COLORS = {"dense-best": "#555c68", "tput": "#aa6537", "nra-8": "#8074a6", "absolute": "#bd875a",
          "bound": "#7ca0b7", "hybrid-25": "#176e83", "safe-zone": "#a34c57", "anchor": "#5b8e58"}


def table(name, head, rows, layout):
    def line(record):
        return " & ".join(record) + r" \\"

    text = [r"\begin{tabular}{" + layout + "}", r"\toprule", line(head), r"\midrule"]
    text += [line(record) for record in rows]
    text.extend([r"\bottomrule", r"\end{tabular}"])
    (FIGURES / name).write_text("\n".join(text) + "\n", encoding="utf-8")


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{filename}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def frame(run):
    return pd.read_json(RUNS / run / "metrics.jsonl", lines=True)


def summarize_native(data, native_run, frozen_hash):
    directory = RUNS / native_run / "artifacts"
    assert frozen_hash == digest(directory / "frozen_policy.json")
    columns = ["source", "unit", "method"]
    data["screen"] = data.up + data.down
    data["bytes"] = data.screen + data.selection_bytes
    data["learner_bytes"] = data.learner_up + data.learner_down
    # Learner transport already includes feature broadcasts, so count them once.
    data["total"] = data.screen + data.learner_bytes
    raw = data.groupby(columns).agg(bytes=("bytes", "sum"), total=("total", "sum"), waves=("phases_with_selection", "mean"),
                                    latency_ns=("ns", "median"), cert=("certificate", "mean"))
    baseline = raw.xs("dense-best", level="method").reindex(raw.index.droplevel("method"))
    for key in ("bytes", "total"):
        raw[key + "_ratio"] = raw[key].to_numpy() / baseline[key].to_numpy()
    raw.to_csv(directory / "native_block_summary.csv")
    sources = {}
    rows = []
    for name in ("beijing", "load"):
        subset = raw.xs(name, level="source")
        cfg = json.loads((directory / f"source__{name}.json").read_text(encoding="utf-8"))
        seeds = {}
        predictions = data[(data.source == name) & (data.method == "dense-best") & (data["round"] > 0)]
        for block, rounds in predictions.groupby("unit"):
            counts = np.sum(np.asarray(rounds.outcomes.to_list()), axis=0)
            n, p, tp, tn, bs, stump, correct = counts.sum(axis=0)
            nonempty = counts[:, 0] > 0
            seeds[str(block)] = {"ba": (tp / p + tn / (n - p)) / 2, "stump": stump / n, "brier": bs / n,
                                 "worst": np.min(counts[nonempty, -1] / counts[nonempty, 0])}
        scores = pd.DataFrame(seeds).T.mean().to_dict()
        changes = []
        for block in range(len(cfg["periods"])):
            with np.load(directory / f"{name}__block{block}.npz", allow_pickle=False) as observed:
                rank = observed["exact_rank"]
            changes.extend(np.any(rank[1:] != rank[:-1], axis=1).tolist())
        by_method = {}
        for method in MAIN_METHODS:
            values = subset.xs(method, level="method")
            stats = values.mean().to_dict()
            stats["sd_bytes_ratio"] = float(values.bytes_ratio.std())
            stats["wins_vs_dense"] = int((values.bytes_ratio < 1).sum())
            stats["range"] = [float(values.bytes_ratio.min()), float(values.bytes_ratio.max())]
            by_method[method] = stats
            rows.append([LABEL[name], DISPLAY[method], f"{stats['bytes_ratio']:.3f}",
                         f"{stats['sd_bytes_ratio']:.3f}", f"{stats['total_ratio']:.3f}",
                         f"{stats['waves']:.2f}", f"{stats['latency_ns'] / 1e6:.3f}"])
        sources[name] = {"dataset": {k: cfg[k] for k in ["clients", "d", "hours", "calibration_hours", "scored_hours",
                         "source_predictors", "rows_with_label", "positive_fraction", "missing_labels"]},
                         "methods": by_method, "prediction": scores,
                         "changed_rounds": sum(changes), "comparable_transitions": len(changes),
                         "policy": str(data[(data.source == name) & data.frozen_policy.eq(1)].method.iloc[0])}
    table("cycle2_native.tex", ["Source", "Method", "Screen", "SD", "Total", "Waves", "ms"], rows, "llrrrrr")
    return sources


def summarize_development(data, dev_run):
    data["bytes"] = data.up + data.down + data.selection_bytes
    index = ["source", "case", "variant", "unit", "d"]
    costs = data.groupby([*index, "method"]).agg(bytes=("bytes", "sum"), waves=("phases_with_selection", "mean"),
                                                cert=("certificate", "mean")).reset_index()
    base = costs[costs.method == "dense-best"].set_index(index).bytes
    costs["ratio"] = costs.bytes.to_numpy() / base.reindex(pd.MultiIndex.from_frame(costs[index])).to_numpy()
    costs.to_csv(RUNS / dev_run / "artifacts" / "case_costs.csv", index=False)
    width = costs[costs.variant.str.startswith("width-")]
    methods = ("nra-8", "tput", "absolute", "hybrid-25", "safe-zone", "anchor")
    means = width.groupby(["d", "method"])[["ratio", "waves"]].mean()
    rows = [[str(d), *[f"{means.loc[(d, method), 'ratio']:.3f}" for method in methods]] for d in sorted(width.d.unique())]
    table("cycle2_width.tex", ["$d$", "NRA", "TPUT", "Absolute", "Hybrid 25", "Safe zone", "Anchored"], rows, "rrrrrrr")
    totals = costs.groupby(["source", "case", "variant", "method"]).mean(numeric_only=True).reset_index()
    data_cases = totals[totals.case == "perturbed"]
    ratios = data_cases.groupby("method").ratio.median().to_dict()
    return {"rows": len(data), "public_medians": ratios,
            "width": {str(d): means.loc[d].ratio.to_dict() for d in sorted(width.d.unique())}}


def summarize_tcp(data, tcp_run):
    fields = ["case", "profile", "round", "repeat"]
    baseline = data[data.method == "dense-best"].set_index(fields).elapsed_ms
    data["pair_ratio"] = data.elapsed_ms.to_numpy() / baseline.reindex(pd.MultiIndex.from_frame(data[fields])).to_numpy()
    data["client_cpu_max"] = data.per_client_socket_cpu_us.map(max)
    results = {}
    rows = []
    for case in LABEL:
        results[case] = {}
        for experiment in PROFILES:
            per_profile = data[(data.case == case) & (data.profile == experiment)]
            by_method = {}
            for method, observed in per_profile.groupby("method"):
                by_method[method] = {
                    "median_ms": float(observed.elapsed_ms.median()),
                    "q90_ms": float(observed.elapsed_ms.quantile(.90)),
                    "median_pair_ratio": float(observed.pair_ratio.median()),
                    "median_goodput_kbps": float(observed.goodput_kbps.median()),
                    "median_server_us": float(observed.server_socket_cpu_us.median()),
                    "median_max_client_us": float(observed.client_cpu_max.median()),
                    "median_waves": float(observed.waves.median()),
                }
            results[case][experiment] = by_method
            kept = [by_method[m]["median_ms"] for m in ("dense-best", "hybrid-25", "nra-8")]
            rows.append([LABEL[case], experiment.replace("ms-", " / "), *[f"{x:.2f}" for x in kept],
                         f"{by_method['hybrid-25']['median_pair_ratio']:.2f}"])
    table("cycle2_tcp.tex", ["Case", "Profile", "Dense", "Hybrid", "NRA", "Hybrid ratio"], rows, "llrrrr")
    data.drop(columns="per_client_socket_cpu_us").to_csv(RUNS / tcp_run / "artifacts" / "tcp_summary.csv", index=False)

    plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    profile_names = ["Loopback", "2 ms\n100 Mb/s", "2 ms\n1 Mb/s", "20 ms\n100 Mb/s", "20 ms\n1 Mb/s"]
    for ax, (case, name) in zip(axes, LABEL.items()):
        for method in TCP_METHODS[1:]:
            comparison = data[(data.case == case) & (data.method == method)]
            stats = comparison.groupby("profile").pair_ratio.median().reindex(PROFILES)
            ax.plot(np.arange(len(stats)), stats, marker="o", label=DISPLAY[method], color=COLORS[method], alpha=.90)
        ax.axhline(1, color="#555c68", linewidth=1.2, linestyle="--")
        ax.set_ylabel("Latency ratio")
        ax.set_title(name, loc="left", fontsize=17)
        ax.set_yscale("log")
    axes[-1].set_xticks(range(len(PROFILES)), profile_names)
    axes[0].legend(frameon=False, fontsize=16, loc="lower left", bbox_to_anchor=(0, 1.13), ncol=3)
    save_plot(fig, "cycle2_network")
    return {"rows": len(data), "results": results}


def analytic_crossings(network_run):
    import gzip
    from collections import defaultdict

    archive = RUNS / network_run / "artifacts" / "frame_traces.jsonl.gz"
    records = []
    with gzip.open(archive, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            waves = defaultdict(lambda: defaultdict(int))
            for item in row["frames"]:
                waves[item["phase"], item["direction"]][item["client"]] += len(item["bytes"]) // 2
            bit_sum = 8 * sum(max(per_client.values()) for per_client in waves.values())
            records.append(dict(case=row["case"], round=row["round"], method=row["method"],
                                waves=len(waves), serial_bits=bit_sum))
    data = pd.DataFrame(records)
    baseline = data[data.method.eq("dense-best")].set_index(["case", "round"])
    data = data[data.method.eq("hybrid-25")].set_index(["case", "round"])
    marginal = data.waves - baseline.waves
    saved = baseline.serial_bits - data.serial_bits
    active = (marginal > 0) & (saved > 0)
    positive = 2e-3 * saved[active] / marginal[active]
    return {case: dict(positive_rounds=int(active.loc[case].sum()),
                       median_rtt_ms_1Mbps=float(positive.loc[case].median()) if case in positive.index else None)
            for case in LABEL}


def summarize_low_rate(data, low_rate_run):
    fields = ["case", "profile", "round", "repeat"]
    paired = data.pivot(index=fields, columns="method", values="elapsed_ms")
    for method in ("hybrid-25", "nra-8"):
        paired[method + "_ratio"] = paired[method] / paired["dense-best"]
    paired.to_csv(RUNS / low_rate_run / "artifacts" / "paired_tcp.csv")
    med = paired.groupby(["case", "profile"]).median()
    cases = {}
    rows = []
    for name in ("beijing", "load"):
        cases[name] = {}
        for rate in ("0.05", "0.1", "0.25", "0.5"):
            obs = med.loc[name, f"2ms-{rate}Mbps"]
            cases[name][rate] = obs.to_dict()
            short_name = "Beijing" if name == "beijing" else "Electricity"
            rows.append([short_name, rate, *[f"{obs[key]:.2f}" for key in
                        ("dense-best", "hybrid-25", "nra-8", "hybrid-25_ratio", "nra-8_ratio")]])
    table("cycle2_low_rate.tex", ["Source", "Mb/s", "Dense", "Hybrid", "NRA", "Hybrid ratio", "NRA ratio"],
          rows, "lrrrrrr")
    return {"rows": len(data), "results": cases}


def summarize_endpoint(data, run):
    data["max_client_call_ms"] = data.per_client_socket_call_ms.map(max)
    data["max_client_cycles"] = data.per_client_socket_cycles.map(max)
    assert data.server_socket_cycles.min() > 0
    values = ["server_socket_call_ms", "max_client_call_ms", "server_socket_cycles", "max_client_cycles"]
    medians = data.groupby(["case", "profile", "method"])[values].median()
    medians.to_csv(RUNS / run / "artifacts" / "endpoint_medians.csv")
    rows = []
    for name in ("beijing", "load"):
        for method in ("dense-best", "hybrid-25", "nra-8"):
            row = medians.loc[(name, "2ms-1Mbps", method)]
            short_name = "Beijing" if name == "beijing" else "Electricity"
            rows.append([short_name, DISPLAY[method], f"{row['server_socket_call_ms']:.2f}",
                         f"{row['max_client_call_ms']:.2f}", f"{row['server_socket_cycles'] / 1000:.0f}",
                         f"{row['max_client_cycles'] / 1000:.0f}"])
    table("cycle2_endpoints.tex", ["Source", "Method", "Server ms", "Client ms", "Server cycles", "Client cycles"],
          rows, "llrrrr")
    return {"rows": len(data), "results": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--tcp", required=True)
    parser.add_argument("--low-rate", required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    policy_file = RUNS / args.dev / "artifacts" / "policy.json"
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    result = {"development": summarize_development(frame(args.dev), args.dev),
              "native": summarize_native(frame(args.native), args.native, digest(policy_file)),
              "tcp": summarize_tcp(frame(args.tcp), args.tcp), "break_even": analytic_crossings(args.tcp),
              "low_rate": summarize_low_rate(frame(args.low_rate), args.low_rate),
              "endpoint": summarize_endpoint(frame(args.endpoint), args.endpoint),
              "policy_hash": digest(policy_file), "policy": policy}
    (ROOT / "result" / "cycle2_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for case, rows in result["native"].items():
        print(case, rows["dataset"], rows["prediction"], rows["methods"]["hybrid-25"])
    for case, cell in result["tcp"]["results"].items():
        print(case, {profile: [round(people[m]["median_ms"], 3) for m in ("dense-best", "hybrid-25", "nra-8")]
                     for profile, people in cell.items()})
    print(result["break_even"])
    print("Chronological changes:", {n: (r["changed_rounds"], r["comparable_transitions"])
                                     for n, r in result["native"].items()})


if __name__ == "__main__":
    main()
