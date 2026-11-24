"""Export data provenance and post-audit summaries from verified study artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def tex(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def table(path, header, rows, layout):
    def line(values):
        return " & ".join(values) + r" \\"

    text = (
        [r"\begin{tabular}{" + layout + "}", r"\toprule", line(header), r"\midrule"]
        + [line(r) for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    )
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--thresholds", required=True)
    args = parser.parse_args()
    results = ROOT / "result" / "runs"
    run = results / args.main / "artifacts"
    figures = ROOT / "figures"
    stats = json.loads((ROOT / "result" / "study_summary.json").read_text(encoding="utf-8"))
    frozen = ROOT / "data" / "derived" / "streams_v2"
    metadata = []
    for p in sorted(run.glob("source__*.json")):
        name = p.stem.split("__", 1)[1]
        record = json.loads(p.read_text(encoding="utf-8"))
        with np.load(frozen / f"{name}.npz", allow_pickle=False) as derived:
            assert derived["features"].shape == (record["n"], record["encoded_columns"])
            assert derived["labels"].shape == derived["groups"].shape == (record["n"],)
        assert sha(frozen / f"{name}.npz") == record["derived_sha256"]
        source = Path(record["path"])
        for asset, expected in record["source_assets_sha256"].items():
            actual = sha((source if source.is_dir() else source.parent) / asset)
            assert actual == expected, (asset, actual)
        metadata.append({"id": name, **record})

    fields = ("id", "name", "source", "n", "warm", "source_columns", "numeric_multilevel", "encoded_columns",
              "positive", "evaluation_positive_fraction", "source_group_count", "target", "excluded", "access",
              "derived_sha256")
    with (ROOT / "result" / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata)

    raw = [m for m in metadata if not m["id"].startswith("openml-")]
    tables = sorted((m for m in metadata if m["id"].startswith("openml-")), key=lambda m: int(m["id"].split("-")[1]))

    def as_row(m):
        short_id = m["id"].split("-")[-1]
        aliases = {"analcatdata_authorship": "Authorship", "default-of-credit-card-clients": "Credit Card Default"}
        original = m.get("name", m["id"])
        description = aliases.get(original, original)
        return [tex(description), tex(short_id if m["id"].startswith("openml-") else "UCI" if m["id"] == "Occupancy" else "MOA"),
                str(m["n"]), str(m["warm"]), str(m["source_columns"]), str(m["numeric_multilevel"]),
                str(m["encoded_columns"]), f"{m['evaluation_positive_fraction']:.3f}"]

    for i, subset in enumerate((raw, tables[:14], tables[14:28], tables[28:]), start=1):
        table(figures / f"data_{i}.tex", ["Dataset", "Source", "$N$", "Warm", "Inputs", "Num.", "$d$", "Pos."],
              [as_row(m) for m in subset], "llrrrrrr")

    trials = pd.read_csv(run / "seed_summary.csv")
    for case in ("perturbed", "provider-order", "natural-groups", "chronological", "synthetic"):
        group = trials[trials.scenario.eq(case)]
        expected = group[group.method.eq("lossless")].set_index(["dataset", "seed"])
        for method in ("dense", "delta", "dense-best", "bound", "tput", "hybrid", "absolute"):
            observed = group[group.method.eq(method)].set_index(["dataset", "seed"])
            pd.testing.assert_frame_equal(observed[["ba", "accuracy", "f1", "brier", "stump", "worst"]],
                                          expected[["ba", "accuracy", "f1", "brier", "stump", "worst"]])
    variants = ["q0", "q1", "bound", "q4", "q12", "g1", "g2", "g4", "g8"]
    sweep = []
    for case, settings in (("Update count", variants[:5]), ("Group count", variants[5:])):
        for method in settings:
            code = ("2" if method == "bound" else method[1:])
            stats_by_source = trials[trials.method.eq(method)].groupby(["scenario", "dataset"]).screen_ratio.mean()
            sweep.append([case, code, f"{stats_by_source.loc['perturbed'].median():.3f}",
                          f"{stats_by_source.loc['synthetic', 'gradual']:.3f}",
                          f"{stats_by_source.loc['provider-order', 'Airlines']:.3f}"])
    table(figures / "envelope_sweep.tex", ["Factor", "Setting", "OpenML median", "Gradual", "Airlines"], sweep, "lrrrr")

    replay = json.loads((results / args.replay / "artifacts" / "summary.json").read_text(encoding="utf-8"))
    records = [["Dense uncompressed", "dense"], ["Dense zlib", "lossless"], ["Dense delta", "delta"],
               ["Best dense frame", "dense-best"], ["TPUT", "tput"], ["Absolute envelope", "absolute"],
               ["Grouped residuals", "bound"], ["Adaptive hybrid", "hybrid"]]
    table(figures / "isolated_time.tex", ["Exact transport", "Median ms"], [
        [name, f"{replay['median_of_source_seed_medians_ms'][key]:.3f}"] for name, key in records], "lr")
    grid = pd.read_csv(results / args.thresholds / "artifacts" / "grid.csv")
    table(figures / "adaptive_sweep.tex", ["Threshold", "OpenML", "Extra", "Airports", "Occupancy", "Ties"], [
        [f"{r.threshold / 100:.2f}", *(f"{x:.3f}" for x in (r.public_ratio, r.public_extra, r.airports, r.occupancy, r.ties))]
        for r in grid.itertuples()
    ], "lrrrrr")
    group = trials[(trials.method == "lossless")].groupby(["scenario", "dataset"])[["ba", "stump", "worst"]].mean()
    trace = {}
    with (run.parent / "metrics.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["method"] == "bound":
                trace[row["dataset"], row["scenario"], row["seed"], row["round"]] = row["extra"]
    case_rows, margins = [], []
    for path in sorted(run.glob("*__s*.npz")):
        dataset, scenario, _, seed = path.stem.split("__")
        seed = int(seed.removeprefix("s"))
        with np.load(path, allow_pickle=False) as saved:
            scores, ranks = saved["scores"], saved["exact_rank"]
        k = ranks.shape[1]
        total = np.sort(scores.sum(axis=1), axis=1)
        for t, gap in enumerate(total[:, -k] - total[:, -k - 1]):
            if t >= 6:
                margins.append((scenario, gap / scores.shape[1], trace[dataset, scenario, seed, t]))
        if seed == 7:
            features = next((m["features"] for m in metadata if m["id"] == dataset), None)
            labels = features if features is not None else [f"feature {i}" for i in range(scores.shape[2])]
            case_rows.append({"dataset": dataset, "scenario": scenario, "seed": seed,
                              "round_7_features": " | ".join(labels[i] for i in ranks[6]),
                              "final_features": " | ".join(labels[i] for i in ranks[-1]),
                              "overlap": len(np.intersect1d(ranks[6], ranks[-1]))})
    pd.DataFrame(case_rows).to_csv(ROOT / "result" / "source_case_examples.csv", index=False)
    diagnostic = pd.DataFrame(margins, columns=["case", "gap_per_client", "refine"])
    diagnostic["gap_group"] = pd.cut(diagnostic.gap_per_client, [-1, 0, 10, 50, 100, float("inf")])
    gaps = diagnostic.groupby(["case", "gap_group"], observed=True).refine.agg(["mean", "size"])
    gaps.to_csv(ROOT / "result" / "margin_diagnostics.csv")
    print(group.loc[["provider-order", "natural-groups", "chronological"]])
    print("Verified data sources:", len(metadata))
    print(gaps)
    print("Excluded source columns:", {m["id"]: m["excluded"] for m in metadata if m["excluded"]})
    print("Synth cases", stats["adaptive_synthetic"])
    print("Median public method metrics:", {m: {k: round(v, 4) for k, v in stats["methods"][m]["perturbed"].items()
          if k in ["screen_ratio", "total_ratio", "exact", "ba", "extra", "worst", "stump"]} for m in stats["methods"]})


if __name__ == "__main__":
    main()
