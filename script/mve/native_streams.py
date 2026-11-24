"""Chronological feature dictionaries for two native-client domains."""

from pathlib import Path
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from fetch_native import SOURCES
from stream_data import digest, encode

HOURS_PER_ROUND = 7 * 24
ROUNDS = 22
BLOCKS = 5


def calendar(index):
    return pd.DataFrame({"hour": index.hour, "weekday": index.dayofweek, "month": index.month,
                         "weekend": index.dayofweek >= 5}, index=index)


def beijing(path):
    panels, labels, sites = [], [], []
    with ZipFile(path) as source:
        names = sorted(n for n in source.namelist() if n.endswith("20130301-20170228.csv") and "PRSA_Data_" in n)
        for member in names:
            with source.open(member) as reader:
                frame = pd.read_csv(reader)
            times = pd.to_datetime(frame[["year", "month", "day", "hour"]])
            index = pd.DatetimeIndex(times, name="timestamp")
            assert index.is_unique and index.is_monotonic_increasing
            assert len(frame) == 35064 and frame.station.nunique() == 1
            frame.index = index
            sites.append(frame.station.iloc[0])
            sensors = frame.drop(columns=["No", "year", "month", "day", "hour", "station"])
            variables = [calendar(index), sensors.shift(1).add_suffix("_lag1")]
            for lag in (6, 24, 168):
                variables.append(sensors[["PM2.5", "PM10", "TEMP", "WSPM"]].shift(lag).add_suffix(f"_lag{lag}"))
            lagged = sensors["PM2.5"].shift(1)
            variables += [lagged.rolling(w, min_periods=w // 2).mean().rename(f"pm25_roll{w}")
                          for w in (6, 24, 168)]
            panels.append(pd.concat(variables, axis=1))
            labels.append(np.where(sensors["PM2.5"].notna(), sensors["PM2.5"] >= 75, -1).astype(np.int8))
    assert len(panels) == 12 and all(panel.index.equals(panels[0].index) for panel in panels)
    return panels, labels, sites, dict(task="hourly PM2.5 >= 75", unit="micrograms per cubic metre",
                                      origin="12 native monitoring stations", n_available_clients=12)


def load(path):
    if path.suffix == ".gz":
        original = pd.read_csv(path, header=None)
        assert original.shape == (26304, 321)
        original.columns = [f"series_{i:03}" for i in original.columns]
        original.index = pd.date_range("2012-01-01", periods=len(original), freq="h")
    else:
        with ZipFile(path) as source:
            member = next(n for n in source.namelist() if n.endswith("LD2011_2014.txt") and "__MACOSX" not in n)
            with source.open(member) as reader:
                original = pd.read_csv(reader, sep=";", decimal=",", index_col=0)
        original.index = pd.to_datetime(original.index)
        original = original.resample("h").mean().loc["2012-01-01":"2014-12-31 23:00:00"]
    assert len(original) == 26304 and original.index.is_unique and original.index.is_monotonic_increasing
    warm = len(original) // 4
    calibration = original.iloc[:warm]
    eligible = calibration.columns[calibration.gt(0).mean() >= .98]
    sites = sorted(np.random.default_rng(871).choice(eligible.to_list(), 24, replace=False).tolist())
    scales = calibration[sites].median(axis=0)
    panels, labels = [], []
    for member in sites:
        normalized = original[member] / scales[member]
        panel = calendar(original.index)
        past = normalized.shift(1)
        for lag in (1, 2, 3, 6, 12, 24, 48, 72, 120, 168, 336):
            panel[f"load_lag{lag}"] = normalized.shift(lag)
        for w in (3, 6, 12, 24, 48, 168):
            rolling = past.rolling(w, min_periods=w // 2 + 1)
            panel[f"load_avg{w}"] = rolling.mean()
            panel[f"load_std{w}"] = rolling.std()
        labels.append(np.where(normalized.notna(), normalized >= 1, -1).astype(np.int8))
        panels.append(panel)
    return panels, labels, sites, dict(task="hourly load >= series calibration median",
                                     origin="24 original meter series, anonymous in LSTNet",
                                     n_available_clients=original.shape[1],
                                     n_eligible_clients=len(eligible),
                                     calibration_medians={k: float(v) for k, v in scales.to_dict().items()})


def prepare_native(name, data_root, output):
    output.mkdir(parents=True, exist_ok=True)
    out = output / (name + ".npz")
    manifest = output / (name + ".json")
    if out.is_file() and manifest.is_file():
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        assert stored["derived_sha256"] == digest(out)
        return out, stored

    directory = data_root / SOURCES[name][0]
    metadata = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    path = directory / metadata["filename"]
    assert metadata["sha256"] == digest(path)
    panels, ys, sites, task = {"beijing": beijing, "load": load}[name](path)
    n, c = len(panels[0]), len(sites)
    warm = n // 4
    index = panels[0].index
    pooled = pd.concat(panels, keys=range(c)).swaplevel().sort_index()
    x, columns, numeric = encode(pooled, warm * c)
    x = x.reshape(n, c, -1)
    y = np.stack(ys, axis=1)

    intervals = []
    for interval in np.array_split(np.arange(warm, n), BLOCKS):
        start = int(interval[0])
        stop = start + ROUNDS * HOURS_PER_ROUND
        assert stop <= int(interval[-1]) + 1
        intervals.append(dict(start=start, stop=stop, start_time=str(index[start]), end_time=str(index[stop - 1])))
    used = np.concatenate([y[block["start"]:block["stop"]] for block in intervals])
    meta = dict(source=metadata, **task, sites=sites, clients=c, hours=n, calibration_hours=warm,
                calibration_last_hour=str(index[warm - 1]), scored_hours=sum(b["stop"] - b["start"] for b in intervals),
                periods=intervals, interval_hours=HOURS_PER_ROUND, rounds=ROUNDS, columns=columns,
                source_predictors=len(panels[0].columns), numeric_multilevel=numeric, d=len(columns),
                rows_with_label=int(np.sum(used >= 0)), missing_labels=int(np.sum(used < 0)),
                positive_fraction=float(np.sum(used == 1) / np.sum(used >= 0)))
    np.savez_compressed(out, features=x, labels=y, timestamp=index.to_numpy(dtype="datetime64[ns]"))
    meta["derived_sha256"] = digest(out)
    manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out, meta


def get_block(x, y, block):
    for left in range(block["start"], block["stop"], HOURS_PER_ROUND):
        right = left + HOURS_PER_ROUND
        part_x, part_y = x[left:right], y[left:right]
        yield [(part_x[:, i][labels >= 0], labels[labels >= 0].astype(np.uint8))
               for i, labels in enumerate(part_y.T)]
