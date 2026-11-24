"""Generate publication figures, statistics and LaTeX tables from recorded runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"
RUNS = ROOT / "result" / "runs"
METHODS = ["lossless", "delta", "dense-best", "tput", "absolute", "bound", "hybrid", "one-group", "no-sparse",
           "shuffled", "periodic-tuned", "sparse-tuned", "random"]
DISPLAY = {"lossless": "Dense zlib", "delta": "Dense delta", "tput": "TPUT", "bound": "Grouped residuals",
           "one-group": "One group", "no-sparse": "No sparse updates", "shuffled": "Shuffled updates",
           "periodic-tuned": "Tuned periodic", "sparse-tuned": "Tuned local top-$q$", "random": "Random",
           "hybrid": "Adaptive hybrid", "absolute": "Absolute envelope", "dense-best": "Best dense frame"}
COLORS = {"lossless": "#657280", "tput": "#be5c34", "bound": "#176e83", "delta": "#7d699e",
          "hybrid": "#007849", "absolute": "#a45a08"}
KEY = ["scenario", "dataset", "variant", "seed"]


def read(slug):
    data = pd.read_json(RUNS / slug / "metrics.jsonl", lines=True)
    data["screen"] = data.up + data.down
    data["total"] = data.screen + data.fit_up + data.fit_down
    return data


def with_tuning(data):
    chosen = []
    for _, group in data.groupby(KEY, sort=False):
        dev = group[group["round"] < 6]
        limit = dev.loc[dev.method.eq("bound"), "screen"].sum()
        for family, prefix in (("periodic", "periodic-"), ("sparse", "local-")):
            grid = dev[dev.method.str.startswith(prefix)].groupby("method").agg(
                cost=("screen", "sum"), overlap=("overlap", "mean"))
            feasible = grid[grid.cost <= limit]
            if len(feasible):
                winner = feasible.sort_values(["overlap", "cost"], ascending=[False, True]).index[0]
            else:
                winner = grid.sort_values(["cost", "overlap"], ascending=[True, False]).index[0]
            chosen.append(group[group.method.eq(winner)].assign(method=family + "-tuned", tuned=winner,
                                                                 dev_budget_met=bool(len(feasible))))
    return pd.concat([data, *chosen], ignore_index=True)


def summarize(data):
    data = data[data["round"] >= 6].copy()
    output = []
    for keys, group in data.groupby(KEY + ["method"], sort=True):
        n, pos, tp, tn, brier, correct, stump = group[
            ["n", "positive", "tp", "tn", "brier", "correct", "stump"]].sum()
        client_n = np.sum(group.client_n.to_list(), axis=0)
        client_correct = np.sum(group.client_correct.to_list(), axis=0)
        included = client_n > 0
        output.append(dict(zip(KEY + ["method"], keys), d=int(group.d.iloc[0]), screen=group.screen.sum(),
                           total=group.total.sum(), ba=(tp / pos + tn / (n - pos)) / 2 if 0 < pos < n else np.nan,
                           accuracy=correct / n, f1=2 * tp / max(1, tp + n - tn), brier=brier / n,
                           stump=stump / n, worst=(client_correct[included] / client_n[included]).min(),
                           extra=group.extra.mean(), fallback=group.fallback.mean(), overlap=group.overlap.mean() / group.k.iloc[0],
                           exact=group.exact.mean(), runtime=np.median(group.runtime_ns) / 1e6, cases=int(n),
                           dense_clients=float(group.dense_clients.mean()) if "dense_clients" in group else None,
                           tuned=group.tuned.iloc[0] if "tuned" in group else None,
                           dev_budget_met=group.dev_budget_met.iloc[0] if "dev_budget_met" in group else None))
    result = pd.DataFrame(output)
    base = result[result.method.eq("lossless")].set_index(KEY)
    for metric in ("screen", "total"):
        aligned = base[metric].reindex(pd.MultiIndex.from_frame(result[KEY])).to_numpy()
        result[metric + "_ratio"] = result[metric] / aligned
    return result


def ci(values, rng):
    x = np.asarray(values, dtype=float)
    distribution = np.median(rng.choice(x, (10000, len(x)), replace=True), axis=1)
    return [float(np.median(x)), *np.quantile(distribution, [0.025, .975]).tolist()]


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    running = 0.0
    output = {}
    for i, key in enumerate(ordered):
        running = min(1, max(running, pvalues[key] * (len(ordered) - i)))
        output[key] = running
    return output


def export(figure, name):
    figure.tight_layout()
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", dpi=100, bbox_inches="tight")
    plt.close(figure)


def draw(main, scale, traces):
    plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False})
    stats = main.groupby(["scenario", "dataset", "method"], as_index=False).agg(
        screen_ratio=("screen_ratio", "mean"), total_ratio=("total_ratio", "mean"), exact=("exact", "mean"),
        extra=("extra", "mean"), d=("d", "first"))
    openml = stats.query("scenario == 'perturbed' and method == 'bound'").sort_values("d")
    real = stats.query("scenario == 'provider-order' and method == 'bound'")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(openml.d, openml.screen_ratio, color="#5391a0", alpha=.78, label="Perturbed public tables")
    ax.scatter(real.d, real.screen_ratio, color="#bd6340", marker="D", label="Provider-order streams")
    ax.set_xscale("log")
    ax.axhline(1, color="0.35", linestyle="--")
    ax.set(xlabel="Number of binary candidates", ylabel="Screening bytes / dense zlib")
    ax.legend(frameon=False, fontsize=16)
    export(fig, "data_width")

    gradual = scale[scale.variant.str.startswith("width-")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in ("lossless", "tput", "bound", "delta"):
        subset = gradual[gradual.method.eq(m)].groupby("d").screen.agg(["mean", "std"])
        ax.errorbar(subset.index, subset["mean"] / 26, subset["std"] / 26, marker="o",
                    color=COLORS[m], label=DISPLAY[m])
    ax.set(xscale="log", yscale="log", xlabel="Number of binary candidates",
           ylabel="Screening bytes per round")
    ax.legend(frameon=False, ncol=2, fontsize=16)
    export(fig, "width_scaling")

    part = traces.query("dataset == 'abrupt' and seed == 7").copy()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for m in ("lossless", "tput", "bound"):
        t = part[part.method.eq(m)]
        ax.plot(t["round"] + 1, t.screen, color=COLORS[m], label=DISPLAY[m], linewidth=2)
    bounds = part[part.method.eq("bound")]
    refined = bounds[bounds.extra > 0]
    ax.scatter(refined["round"] + 1, refined.screen, facecolors="none", edgecolors="#142d35", s=70, label="Refinement")
    ax.set(ylabel="Screening bytes")
    ax.legend(frameon=False, ncol=2, fontsize=16)
    ax2.step(bounds["round"] + 1, bounds.drift.astype(float), where="post", color="#313b45")
    ax2.set(ylabel="Shifted clients\n(fraction)", xlabel="Round", ylim=(-.10, 1.10))
    export(fig, "abrupt_trace")

    fig, ax = plt.subplots(figsize=(10, 6))
    cases = ["stationary", "gradual", "abrupt", "recurring", "opposing", "ties"]
    markers = ["o", "s", "^", "D"]
    for marker, m in zip(markers, ("bound", "one-group", "no-sparse", "shuffled")):
        part = stats.query("scenario == 'synthetic' and method == @m").set_index("dataset").loc[cases]
        ax.plot(cases, part.screen_ratio, marker=marker, linewidth=1.6, label=DISPLAY[m])
    ax.axhline(1, color="0.4", linestyle="--")
    ax.set(ylabel="Screening bytes / dense zlib")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(frameon=False, ncol=2, fontsize=16)
    export(fig, "ablations")

    if "hybrid" in set(stats.method):
        public = stats.query("method in ['bound', 'hybrid', 'absolute', 'dense-best'] and scenario != 'synthetic'")
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in ("bound", "hybrid", "absolute", "dense-best"):
            ratios = np.sort(public.loc[public.method.eq(m), "screen_ratio"])
            ax.step(ratios, np.arange(1, len(ratios) + 1) / len(ratios), where="post",
                    color=COLORS.get(m, "#343c43"), linewidth=2, label=DISPLAY[m])
        ax.axvline(1, color="0.35", linestyle="--")
        ax.set(xscale="log", xlabel="Screening bytes / dense zlib", ylabel="Fraction of evaluated cases")
        ax.legend(frameon=False, fontsize=16, loc="lower right")
        export(fig, "adaptive_cost")
    return stats


def to_latex(frame, name, heading, formats):
    lines = ["\\begin{tabular}{" + "l" + "r" * (len(heading) - 1) + "}", "\\toprule",
             " & ".join(heading) + r" \\", "\\midrule"]
    for record in frame:
        lines.append(" & ".join(fmt.format(value) for fmt, value in zip(formats, record)) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (FIGURES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(main, scale, traces, additional=None, extra_traces=None):
    FIGURES.mkdir(exist_ok=True)
    stats = draw(main, scale, traces)
    tabular = stats.query("scenario == 'perturbed'").pivot(index="dataset", columns="method", values="screen_ratio")
    tab_aligned = main.query("scenario == 'perturbed'").groupby(["dataset", "method"]).mean(numeric_only=True)
    rng = np.random.default_rng(20261124)
    targets = [m for m in ("lossless", "delta", "tput", "absolute", "dense-best") if m in tabular]
    paired = {m: (1 - tabular["bound"] / tabular[m]).to_numpy() for m in targets}
    pvalues = {m: float(wilcoxon(arr).pvalue) for m, arr in paired.items()}
    corrected = holm(pvalues)
    methods = {}
    methods_present = [method for method in METHODS if method in set(main.method)]
    for method in methods_present:
        subset = main[main.method.eq(method)]
        methods[method] = {s: g.groupby("dataset").mean(numeric_only=True).median().dropna().to_dict()
                           for s, g in subset.groupby("scenario")}
    real = main.query("scenario == 'provider-order'").groupby(["dataset", "method"]).mean(numeric_only=True)
    bound = main.query("method == 'bound'")
    real_desc = []
    for dataset in real.index.get_level_values(0).unique():
        v = real.loc[dataset]
        real_desc.append([dataset, int(v.loc["bound", "d"]), *v.loc["bound", ["screen_ratio", "total_ratio", "extra", "ba"]],
                          v.loc["tput", "screen_ratio"], v.loc["delta", "screen_ratio"]])
    to_latex(real_desc, "real.tex", ["Source", "$d$", "Grouped", "Total", "Extra", "BA", "TPUT", "Delta"],
             ["{}", "{:d}"] + ["{:.3f}"] * 6)
    rows = []
    for m in methods_present:
        values = methods[m]["perturbed"]
        rows.append([DISPLAY[m], *(values[k] for k in ("screen_ratio", "total_ratio", "exact", "ba", "f1", "brier", "runtime"))])
    to_latex(rows, "tabular.tex", ["Method", "Screen", "Total", "Exact", "BA", "F1", "Brier", "ms"],
             ["{}"] + ["{:.3f}"] * 7)

    synthetic = stats.query("scenario == 'synthetic'")
    synrows = []
    for name in ("stationary", "gradual", "abrupt", "recurring", "opposing", "ties"):
        q = synthetic[synthetic.dataset.eq(name)].set_index("method")
        synrows.append([name.capitalize(), *q.loc[["bound", "tput", "delta"], "screen_ratio"],
                        q.loc["bound", "extra"], q.loc["bound", "total_ratio"]])
    to_latex(synrows, "synthetic.tex", ["Scenario", "Grouped", "TPUT", "Delta", "Extra", "Total"],
             ["{}"] + ["{:.3f}"] * 5)

    scale_table = []
    for var in ("width", "clients", "select", "memory", "alpha"):
        dataset = scale[scale.variant.str.startswith(var + "-") & scale.method.eq("bound")].copy()
        dataset["level"] = dataset.variant.str.split("-").str[-1].astype(float)
        summary = dataset.groupby("level")[["screen_ratio", "extra", "runtime"]].agg(["mean", "std"])
        for level, metrics in summary.iterrows():
            scale_table.append([var.capitalize(), level, *(metrics[(k, "mean")] for k in ("screen_ratio", "extra", "runtime")),
                                *(metrics[(k, "std")] for k in ("screen_ratio", "extra", "runtime"))])
    to_latex(scale_table, "sensitivity.tex", ["Factor", "Level", "Screen", "Extra", "ms", "SD screen", "SD extra", "SD ms"],
             ["{}", "{:g}"] + ["{:.3f}"] * 6)
    real_stream_points = stats.query("scenario == 'provider-order' and method == 'bound'")
    stats_log = {
        "records": len(traces), "runs": len(main.groupby(KEY)), "rank_rounds": int(traces.method.eq("bound").sum()),
        "mismatches": int(((traces.method.eq("bound")) & traces.exact.eq(0)).sum()),
        "paired_screen_savings": {m: dict(median_ci=ci(paired[m], rng), wins=int((paired[m] > 0).sum()),
                                         holm_p=corrected[m]) for m in targets},
        "tabular_total_savings": ci(1 - tab_aligned.xs("bound", level="method").total_ratio, rng),
        "bound_tabular_seed_sd_median": float(bound.query("scenario == 'perturbed'").groupby("dataset").screen_ratio.std().median()),
        "width_spearman": list(spearmanr(tab_aligned.xs("bound", level="method").d, 1 - tabular["bound"])),
        "real": real_desc, "synthetic": synrows, "methods": methods, "sensitivity": scale_table,
        "tuning_feasible": {str(k): float(v) for k, v in main.dropna(subset=["dev_budget_met"]).groupby("method").dev_budget_met.mean().items()},
        "extremes": tabular["bound"].sort_values().iloc[np.r_[0:3, -3:0]].to_dict(),
    }
    if "hybrid" in tabular:
        comparisons = {m: (1 - tabular.hybrid / tabular[m]).to_numpy() for m in (*targets, "bound")}
        stats_log["adaptive"] = {
            m: dict(median_ci=ci(values, rng), wins=int((values > 0).sum()), ties=int((values == 0).sum()))
            for m, values in comparisons.items()
        }
        stats_log["adaptive_exact_rounds"] = int(traces.method.eq("hybrid").sum())
        stats_log["adaptive_rank_errors"] = int((traces.method.isin(["absolute", "hybrid", "dense-best"]) & traces.exact.eq(0)).sum())
        stats_log["adaptive_synthetic"] = synthetic.query("method == 'hybrid'").set_index("dataset")[
            ["screen_ratio", "extra", "total_ratio"]].to_dict("index")
        adaptive_cases = stats.query("method == 'hybrid' and scenario != 'synthetic'").set_index("dataset")
        stats_log["adaptive_real"] = adaptive_cases[~adaptive_cases.index.str.startswith("openml-")][
            ["scenario", "screen_ratio", "extra", "total_ratio"]].reset_index().to_dict("records")
    if additional is not None:
        additional_stats = additional.groupby(["scenario", "dataset", "method"])[
            ["screen_ratio", "total_ratio", "extra", "ba", "worst", "runtime"]].mean()
        extra_rows = []
        for (scenario, name, method), row in additional_stats.iterrows():
            if method in ("lossless", "bound", "hybrid", "absolute", "tput"):
                extra_rows.append([name, DISPLAY[method], *row.to_list()])
        to_latex(extra_rows, "followup.tex", ["Source", "Transport", "Screen", "Total", "Extra", "BA", "Worst", "ms"],
                 ["{}", "{}"] + ["{:.3f}"] * 6)
        stats_log["followup"] = extra_rows
        stats_log["followup_ranks"] = int(extra_traces.method.eq("hybrid").sum())
        stats_log["followup_mismatches"] = int((extra_traces.method.isin(["hybrid", "absolute"]) & extra_traces.exact.eq(0)).sum())
    (ROOT / "result" / "study_summary.json").write_text(json.dumps(stats_log, indent=2, allow_nan=False), encoding="utf-8")
    return stats_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--additional")
    args = parser.parse_args()
    primary, scaling = read(args.main), read(args.scale)
    main, scale = summarize(with_tuning(primary)), summarize(scaling)
    main.to_csv(RUNS / args.main / "artifacts" / "seed_summary.csv", index=False)
    scale.to_csv(RUNS / args.scale / "artifacts" / "seed_summary.csv", index=False)
    more = read(args.additional) if args.additional else primary[primary.scenario.isin(["chronological", "natural-groups"])]
    if more.empty:
        more = None
    additional = summarize(more) if more is not None else None
    if additional is not None:
        additional.to_csv(RUNS / (args.additional or args.main) / "artifacts" / "additional_seed_summary.csv", index=False)
    result = analyze(main, scale, primary, additional, more)
    for key, value in result.items():
        if key not in ("methods", "sensitivity"):
            print(key, value)
