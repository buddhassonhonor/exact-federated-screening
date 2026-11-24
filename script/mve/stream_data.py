"""Build auditable binary feature streams from public data and controlled generators."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from fetch_streams import DATASETS

OPENML = (
    3, 6, 12, 21, 23, 24, 28, 31, 32, 36, 38, 44, 46, 50, 54, 179, 181, 182,
    184, 185, 307, 312, 458, 1049, 1050, 1053, 1067, 1068, 1120, 1461, 1462,
    1485, 1486, 1487, 1489, 1494, 1497, 1504, 40701, 40983, 42477, 44130,
)
REAL = DATASETS
HOLDOUT = "Occupancy"
SYNTHETIC = ("stationary", "abrupt", "gradual", "recurring", "opposing", "ties")
EXCLUSIONS = {
    312: ["Beach", "Sunset", "FallFoliage", "Field", "Mountain"],  # Sibling labels.
    458: ["BookID"],  # Source document identifier.
    1461: ["V12"],  # UCI duration is known only after the call.
    1504: [f"V{i}" for i in range(28, 34)],  # UCI has 27 predictors and seven label flags.
}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def encode(frame, warm):
    """Fit quantiles and categorical vocabulary on strictly earlier rows."""
    binary = []
    labels = []
    numeric = 0
    for name, series in frame.items():
        calibration = series.iloc[:warm].dropna()
        if pd.api.types.is_numeric_dtype(series) and calibration.nunique() > 2:
            numeric += 1
            values = series.to_numpy(dtype=float, na_value=np.nan)
            for cutoff in np.unique(np.quantile(calibration, [0.25, 0.5, 0.75])):
                binary.append(values > cutoff)
                labels.append(f"{name}>{cutoff:.8g}")
        else:
            values = series.astype("string")
            levels = values.iloc[:warm].value_counts().head(8).index
            for level in levels:
                binary.append(values.eq(level).fillna(False).to_numpy(dtype=bool))
                labels.append(f"{name}={level}")
        if series.isna().iloc[:warm].any():
            binary.append(series.isna().to_numpy())
            labels.append(f"{name}=missing")
    if len(binary) < 2:
        raise ValueError("Fewer than two binary candidates")
    return np.column_stack(binary), labels, numeric


def public_data(name, data_root, output, cleaned=True):
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{name}.npz"
    meta_path = output / f"{name}.json"
    if data_path.exists() and meta_path.exists():
        with np.load(data_path, allow_pickle=False) as saved:
            return saved["features"], saved["labels"], saved["groups"], json.loads(meta_path.read_text(encoding="utf-8"))
    excluded = []
    if name.startswith("openml"):
        did = int(name.split("-")[1])
        base = data_root / "openml" / "org" / "openml" / "www" / "datasets" / str(did)
        path = base / f"dataset_{did}.pq"
        source = ET.parse(base / "description.xml").getroot()
        target = source.findtext("{*}default_target_attribute")
        frame = pd.read_parquet(path)
        if target not in frame:
            raise ValueError(f"Missing OpenML target {target}: {did}")
        frame = frame.loc[frame[target].notna()].sample(frac=1, random_state=871).iloc[:32000].reset_index(drop=True)
        warm = max(128, len(frame) // 5)
        src_meta = dict(source=f"https://www.openml.org/d/{did}", name=source.findtext("{*}name"),
                        version=source.findtext("{*}version"), access=source.findtext("{*}licence"))
        positive = frame[target].iloc[:warm].value_counts().index[0]
        groups = np.zeros(len(frame), dtype=np.int32)
        if cleaned:
            ignored = source.findtext("{*}ignore_attribute") or ""
            identifiers = source.findtext("{*}row_id_attribute") or ""
            exclusions = ignored.split(",") + identifiers.split(",") + EXCLUSIONS.get(did, [])
            excluded = sorted({column.strip() for column in exclusions if column.strip() in frame and column.strip() != target})
            frame = frame.drop(columns=excluded)
        assets = {path.name: digest(path), "description.xml": digest(base / "description.xml")}
    elif name == HOLDOUT:
        path = data_root / "uci_occupancy_detection"
        src_meta = json.loads((path / "source.json").read_text(encoding="utf-8"))
        assets = {name: digest(path / name) for name in src_meta["files"]}
        assert assets == {name: item["sha256"] for name, item in src_meta["files"].items()}
        parts = [pd.read_csv(path / name, index_col=0, parse_dates=["date"]) for name in assets]
        frame = pd.concat(parts, ignore_index=True).sort_values("date", kind="stable")
        frame = frame.drop_duplicates()
        if frame["date"].duplicated().any():
            raise ValueError("Conflicting rows at the same timestamp in the Occupancy source")
        src_meta.update(first_time=str(frame["date"].iloc[0]), last_time=str(frame["date"].iloc[-1]))
        frame = frame.drop(columns="date").reset_index(drop=True)
        excluded = ["row ID", "date"]
        target, positive = "Occupancy", 1
        warm = max(128, len(frame) // 5)
        groups = np.zeros(len(frame), dtype=np.int32)
    else:
        path = data_root / f"capymoa_{name.lower()}" / f"{name}.csv.gz"
        src_meta = json.loads((path.parent / "source.json").read_text(encoding="utf-8"))
        limits = dict(Airlines=180000, Sensor=260000, CovtypeNorm=180000, PokerHand=80000)
        frame = pd.read_csv(path, nrows=limits.get(name))
        target = frame.columns[-1]
        if name == "Electricity":
            # Delay observed market measurements by one row.
            lagged = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
            frame[lagged] = frame[lagged].shift(1)
            frame = frame.iloc[1:].reset_index(drop=True)
            frame["day"] = frame["day"].astype("string")
            positive = 1
        elif name == "Airlines":
            frame["Flight"] = frame["Flight"].astype("string")
            frame["DayOfWeek"] = frame["DayOfWeek"].astype("string")
            positive = 1
        elif name == "Sensor":
            # The original target is sensor ID. Parity defines a balanced binary task.
            frame[target] = frame[target].astype(int) % 2
            positive = 1
        elif name == "PokerHand":
            for column in frame.columns[:-1]:
                frame[column] = frame[column].astype("string")
            positive = 2
        else:
            positive = 2
        warm = max(128, len(frame) // 5)
        client_columns = {"Airlines": "AirportFrom", "Electricity": "day"}
        groups = (pd.factorize(frame[client_columns[name]].astype("string"), sort=True)[0].astype(np.int32)
                  if name in client_columns else np.zeros(len(frame), dtype=np.int32))
        assets = {path.name: digest(path)}

    x, features, numeric = encode(frame.drop(columns=target), warm)
    y = frame[target].eq(positive).to_numpy(dtype=np.uint8)
    metadata = {**src_meta, "path": str(path), "source_assets_sha256": assets, "warm": warm,
                "source_columns": len(frame.columns) - 1, "excluded": excluded,
                "numeric_multilevel": numeric, "features": features, "target": target, "positive": str(positive),
                "n": len(frame), "positive_fraction": float(y.mean()), "encoded_columns": len(features),
                "evaluation_positive_fraction": float(y[warm:].mean()),
                "source_group_count": int(len(np.unique(groups)))}
    np.savez_compressed(data_path, features=x, labels=y, groups=groups)
    metadata["derived_sha256"] = digest(data_path)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return x, y, groups, metadata


def build_public(x, y, groups, warm, rounds, clients, seed, scenario, alpha=1.0):
    rng = np.random.default_rng(seed)
    x, y, groups = x[warm:], y[warm:], groups[warm:]
    # All seeds see identical held-forward rows. Only client assignment and
    # the artificial relevance mask are seeded.
    edges = np.linspace(0, len(y), rounds + 1, dtype=int)
    class_mix = rng.dirichlet(np.full(clients, alpha), size=2)
    if scenario == "natural-groups":
        group_assignment = rng.permutation(int(groups.max()) + 1) % clients
    permuted = False
    drift_idx = rng.permutation(x.shape[1])
    for t, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        rows = x[left:right].copy()
        labels = y[left:right]
        if scenario == "natural-groups":
            assignment = group_assignment[groups[left:right]]
        else:
            assignment = np.empty(len(labels), dtype=int)
            for label in (0, 1):
                mask = labels == label
                assignment[mask] = rng.choice(clients, size=mask.sum(), p=class_mix[label])
        if scenario == "perturbed":
            permuted = (t >= rounds // 3) and (t < 2 * rounds // 3)
            chosen = drift_idx[: max(1, x.shape[1] // 3)] if permuted else drift_idx[-max(1, x.shape[1] // 3):]
            for client in range(clients):
                ix = np.flatnonzero(assignment == client)
                if len(ix):
                    rows[np.ix_(ix, chosen)] = rows[np.ix_(rng.permutation(ix), chosen)]
        yield [(rows[assignment == j], labels[assignment == j]) for j in range(clients)], bool(permuted)


def build_synthetic(scenario, rounds, clients, d, n, seed, intensity=1.0):
    rng = np.random.default_rng(seed)
    client_flip = rng.uniform(0.04, 0.17, clients)
    for t in range(rounds):
        row = []
        weights = []
        for i, flip in enumerate(client_flip):
            x = rng.integers(0, 2, (n, d), dtype=np.uint8).view(bool)
            w = 0.0
            center = rounds * (0.36 + 0.24 * i / max(1, clients - 1))
            if scenario == "abrupt":
                w = float(t >= center) * intensity
            elif scenario == "gradual":
                w = float(np.clip((t - center + rounds / 8) / (rounds / 4), 0, 1)) * intensity
            elif scenario == "recurring":
                w = float(center <= t < center + rounds / 4) * intensity
            elif scenario == "opposing":
                w = float((t >= center) ^ (i % 2 == 1)) * intensity
            if scenario == "ties":
                y = (x[:, 0] ^ x[:, 1]).astype(np.uint8)
                x[:, 2:] = x[:, 2, None]
            else:
                active = np.where(rng.random(n) < w, 4, 0)
                # Four redundant sensors with ranked reliability, plus null features.
                y = rng.integers(0, 2, size=n, dtype=np.uint8)
                for j, err in enumerate((0.02, 0.09, 0.16, 0.24)):
                    x[np.arange(n), active + j] = y ^ (rng.random(n) < err + flip)
            row.append((x, y))
            weights.append(w)
        yield row, float(np.mean(weights))
