"""Fetch canonical real-stream assets to the shared source catalog."""

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SOURCE = "https://raw.githubusercontent.com/adaptive-machine-learning/CapyMOA/main/src/capymoa/datasets/_source_list.py"
DATASETS = ("Electricity", "Airlines", "Sensor", "CovtypeNorm", "PokerHand")
OCCUPANCY_COMMIT = "ba2b1d7b53a71292ff61b867d3e16f0e539c7eb6"
OCCUPANCY_FILES = ("datatraining.txt", "datatest.txt", "datatest2.txt")


def fetch(root):
    with urllib.request.urlopen(SOURCE, timeout=30) as response:
        tree = ast.parse(response.read().decode())
    mapping = next(node.value for node in tree.body if isinstance(node, ast.AnnAssign)
                   and node.target.id == "SOURCE_LIST")
    urls = {key.value: ast.literal_eval(value.args[1]) for key, value in zip(mapping.keys, mapping.values)}

    def save(name):
        folder = root / f"capymoa_{name.lower()}"
        folder.mkdir(exist_ok=True)
        path = folder / f"{name}.csv.gz"
        if not path.exists():
            staged = path.with_suffix(".part")
            with urllib.request.urlopen(urls[name], timeout=45) as response, staged.open("wb") as out:
                shutil.copyfileobj(response, out)
            with gzip.open(staged, "rt", encoding="utf-8") as stream:
                header = next(stream).strip()
                next(stream)
            staged.replace(path)
        else:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                header = next(stream).strip()
        metadata = dict(name=name, provider="CapyMOA dataset distribution", source=urls[name],
                        access="Public download. Check upstream dataset terms before redistribution.",
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), header=header, path=str(path))
        (folder / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata

    with ThreadPoolExecutor(max_workers=5) as pool:
        for result in pool.map(save, DATASETS):
            print(json.dumps(result), flush=True)


def fetch_occupancy(root):
    folder = root / "uci_occupancy_detection"
    folder.mkdir(exist_ok=True)
    base = f"https://raw.githubusercontent.com/LuisM78/Occupancy-detection-data/{OCCUPANCY_COMMIT}/"
    files = {}
    for name in OCCUPANCY_FILES:
        path = folder / name
        url = base + name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "url": url}
    source = dict(name="Occupancy", source="https://archive.ics.uci.edu/dataset/357/occupancy+detection",
                  mirror=base, doi="10.24432/C5X01N", access="CC BY 4.0", files=files)
    (folder / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(source), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--with-occupancy", action="store_true")
    args = parser.parse_args()
    fetch(args.data_root)
    if args.with_occupancy:
        fetch_occupancy(args.data_root)
