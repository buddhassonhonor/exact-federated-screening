# Data sources

| Source | Reproduction |
|---|---|
| [Capital Bikeshare system data](https://capitalbikeshare.com/system-data) | Use all 12 calendar-month archives from 2022. The held-out runner puts downloads in `data/cache/capital_bikeshare_2022` when given `--data-root data/cache`. The archive names, URLs and SHA-256 values are in `bikeshare_archives.json`. Follow the [provider data terms](https://capitalbikeshare.com/data-license-agreement). |
| [CapyMOA dataset distribution](https://capymoa.org/v0.15.1/api/modules/capymoa.datasets.html) | Use the provider order for Airlines, CovtypeNorm, Electricity, PokerHand and Sensor. Run `fetch_streams.py`. |
| [OpenML](https://www.openml.org/) | Use the IDs in `dataset_manifest.csv`. Place each provider Parquet table and OpenML dataset-description XML in `data/cache/openml/org/openml/www/datasets/<id>/`, naming them `dataset_<id>.pq` and `description.xml`. The runner reads the target name from the description. |
| [UCI Occupancy Detection](https://doi.org/10.24432/C5X01N) | Use the three original measurement files, sorted together by timestamp. The pinned source is in `fetch_streams.py`. |
| [Beijing Multi-Site Air Quality](https://doi.org/10.24432/C5RK5G) | Use the 12 original station CSV records. `fetch_native.py` stages the provider files and can use a transport mirror. |
| [Hourly Electricity release used by LSTNet](https://github.com/laiguokun/multivariate-time-series-data) | The study uses the hourly LSTNet file across 321 anonymous original series. The original [UCI source](https://doi.org/10.24432/C58C86) is a separate, finer-resolution release. Place the hourly gzip under `data/cache/uci_electricity_load_diagrams/lstnet_electricity_hourly.txt.gz`, and save the `load` object from `native_sources.json` as `source.json` in that directory. This fixes the hourly source variant and expected checksum. |

The release contains source identifiers, provider asset checksums and aggregate research outputs. Source datasets and derived observation-level inputs are assembled locally from their respective providers.
