"""Fetch public native-client streams with source and integrity metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil
import urllib.request
from zipfile import ZipFile
import gzip

from stream_data import digest

AIR = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
AIR_MIRROR = "https://codeload.github.com/KM-drago/Beijing-Multi-Site-Air-Quality-Data/zip/refs/heads/main"
LOAD = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
LOAD_MIRROR = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz"
SOURCES = {
    "beijing": ("uci_beijing_multisite", "10.24432/C5RK5G", "https://archive.ics.uci.edu/dataset/501/beijing",
                ((AIR, "PRSA2017_Data_20130301-20170228.zip"), (AIR_MIRROR, "beijing-provider-files-mirror.zip"))),
    "load": ("uci_electricity_load_diagrams", "10.24432/C58C86", "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
             ((LOAD, "LD2011_2014.txt.zip"), (LOAD_MIRROR, "lstnet_electricity_hourly.txt.gz"))),
}


def validate(path, name):
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as reader:
            assert len(reader.readline().split(b",")) == 321
        return
    with ZipFile(path) as reader:
        members = reader.namelist()
        if name == "beijing":
            assert sum(m.endswith("20130301-20170228.csv") and "PRSA_Data_" in m for m in members) == 12
        else:
            assert any(m.endswith("LD2011_2014.txt") and "__MACOSX" not in m for m in members)


def fetch(name, root):
    directory, doi, provider, candidates = SOURCES[name]
    folder = root / directory
    folder.mkdir(parents=True, exist_ok=True)
    for url, filename in candidates:
        path = folder / filename
        try:
            if not path.is_file():
                staged = folder / (filename + ".part")
                with urllib.request.urlopen(url, timeout=16) as response, staged.open("wb") as out:
                    shutil.copyfileobj(response, out, length=2 ** 20)
                staged.replace(path)
            validate(path, name)
        except (OSError, ValueError, AssertionError) as exc:
            print(f"{name} {url}: {exc}", flush=True)
            continue
        metadata = dict(name=name, doi=doi, provider=provider, reported_provider_access="CC BY 4.0",
                        url=url, filename=filename, sha256=digest(path), size=path.stat().st_size)
        (folder / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata), flush=True)
        return
    raise RuntimeError(f"Unable to fetch and validate {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for task in [pool.submit(fetch, s, args.data_root) for s in SOURCES]:
            task.result()
