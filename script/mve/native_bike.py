"""Build an hourly native-station benchmark from provider trip archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd

from native_streams import calendar
from stream_data import digest, encode

BASE = "https://s3.amazonaws.com/capitalbikeshare-data/"
PROVIDER = "https://capitalbikeshare.com/system-data"
MONTHS = tuple(f"2022{m:02}" for m in range(1,13))
HOURS, ROUNDS, CLIENTS = 168, 7, 24


def fetch(root, month):
    name = f"{month}-capitalbikeshare-tripdata.zip"
    path = root / name
    if not path.is_file():
        staging = path.with_suffix(".part")
        with urllib.request.urlopen(BASE + name, timeout=40) as response, staging.open("wb") as target:
            shutil.copyfileobj(response, target)
        staging.replace(path)
    with ZipFile(path) as archived:
        files = [f for f in archived.namelist() if f.endswith(".csv") and not f.startswith("__MACOSX")]
        assert len(files) == 1, files
    return {"month":month, "name":name, "member":files[0], "sha256":digest(path), "url":BASE+name}


def hourly(root, item):
    columns = ["started_at", "ended_at", "start_station_id", "end_station_id"]
    with ZipFile(root/item["name"]) as archived, archived.open(item["member"]) as src:
        df = pd.read_csv(src, usecols=columns, dtype={"start_station_id":"string", "end_station_id":"string"})
    counts = []
    for time, station in zip(columns[:2], columns[2:]):
        t = pd.to_datetime(df[time], format="mixed", errors="coerce").dt.floor("h")
        count = pd.DataFrame({"time":t, "station":df[station]}).groupby(["time","station"]).size()
        counts.append(count)
    return counts


def prepare(data_root, output, wide=False):
    root = data_root / "capital_bikeshare_2022"
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    name = "bike_wide" if wide else "bike"
    npz, manifest = output / f"{name}.npz", output / f"{name}.json"
    if npz.exists() and manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        assert digest(npz) == meta["derived_sha256"]
        return npz, meta
    with ThreadPoolExecutor(max_workers=5) as pool:
        files = list(pool.map(lambda m: fetch(root,m), MONTHS))
        chunks = list(pool.map(lambda m: hourly(root,m), files))
    provenance = {"provider":PROVIDER,"terms":"Provider's data license, linked from the system data page",
                  "assets":files}
    index = pd.date_range("2022-01-01","2022-12-31 23:00:00",freq="h")
    departures, arrivals = (
        pd.concat(records).groupby(level=(0,1)).sum().unstack(fill_value=0).reindex(index,fill_value=0)
        for records in zip(*chunks)
    )
    warm = len(index)//4
    calibration = departures.iloc[:warm]
    eligible = calibration.columns[calibration.median().ge(1)]
    high_use = calibration[eligible].sum().nlargest(100).index
    selected = sorted(np.random.default_rng(871).choice(high_use.to_list(),CLIENTS,replace=False))
    lags=(1,2,3,6,12,24,48,168)
    windows=(3,6,24,168)
    if wide:
        lags=tuple(sorted(set(range(1,49))|{60,72,84,96,120,144,168}))
        windows=(3,6,12,18,24,36,48,72,96,120,144,168)
    panel, labels = [], []
    for site in selected:
        dep, arr = departures[site], arrivals.reindex(columns=selected,fill_value=0)[site]
        columns = {}
        for series,label in ((dep,"departures"),(arr,"arrivals")):
            for lag in lags:
                columns[f"{label}_lag{lag}"] = series.shift(lag)
            for window in windows:
                past = series.shift(1).rolling(window,min_periods=window//2+1)
                columns[f"{label}_mean{window}"] = past.mean()
        panel.append(pd.concat([calendar(index),pd.DataFrame(columns,index=index)],axis=1))
        labels.append((dep>=calibration[site].median()).to_numpy(np.int8))
    pooled = pd.concat(panel, keys=range(CLIENTS)).swaplevel().sort_index()
    x,names,numeric = encode(pooled,warm*CLIENTS)
    y=np.stack(labels,axis=1)
    spans=[]
    for b in np.array_split(np.arange(warm,len(index)),5):
        start = int(b[0])
        stop = start+HOURS*ROUNDS
        assert stop<=int(b[-1])+1
        spans.append(dict(start=start,stop=stop,start_time=str(index[start]),end_time=str(index[stop-1])))
    scored=np.concatenate([y[b["start"]:b["stop"]] for b in spans])
    np.savez_compressed(npz,features=x.reshape(len(index),CLIENTS,-1),labels=y,timestamp=index.to_numpy(dtype="datetime64[ns]"))
    meta={"source":provenance,"sites":selected,"clients":CLIENTS,"eligible":len(eligible),"hours":len(index),
          "rounds":ROUNDS,"interval_hours":HOURS,"periods":spans,"calibration_hours":warm,"d":len(names),
          "columns":names,"numeric":numeric,"labeled_rows":int(scored.size),"positive_fraction":float(scored.mean()),
          "dictionary":name,"lags":lags,"windows":windows,
          "task":"Current hourly departures at least the station's calibration median",
          "time":"Provider local clock; repeated civil hours aggregated together",
          "derived_sha256":digest(npz)}
    (root/"source.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    manifest.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print({k:meta[k] for k in ("clients","eligible","hours","d","labeled_rows","positive_fraction")},flush=True)
    return npz,meta


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,default=Path("D:/data"))
    p.add_argument("--out",type=Path,default=Path("data/derived/native_v2"))
    p.add_argument("--wide",action="store_true")
    args=p.parse_args()
    prepare(args.data_root,args.out,args.wide)
