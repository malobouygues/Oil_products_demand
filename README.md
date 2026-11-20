# Forecasting oil products demand with XGBoost

Can a gradient-boosted tree, fed lagged demand and one physical leading indicator, forecast
monthly oil products demand twelve months ahead better than a simple time series?

Two products, each with a physical indicator that a desk could read before the official figure:

- **US jet fuel** (EIA product supplied). The indicator is the kerosene burned by the commercial
  jets leaving US airports, computed flight by flight from OpenSky's ADS-B record with OpenAP's
  fuel model and readable the next day.
- **Naphtha in Korea and Japan** (JODI total demand). The indicator is the price Japan pays for
  propane against naphtha at its border, from the customs unit values. Korean crackers can
  switch to propane and Japanese ones almost cannot, so Japan is the control.

## Notebooks

| Notebook | Content |
|---|---|
| [`notebooks/01_data_jet.ipynb`](notebooks/01_data_jet.ipynb) | Data pipeline, US jet fuel: raw files into DuckDB, data dictionary, validation, publication lags tested against a real download, revisions, `merge_asof` alignment, no-look-ahead assertions, final dataset, and how the burn index is built: query, aircraft types, OpenAP fuel table, unit check |
| [`notebooks/01_data_naphtha.ipynb`](notebooks/01_data_naphtha.ipynb) | Data pipeline, naphtha: the same sections, plus the customs codes, the unit values and the Gulf's share of Japan's imports |
| [`notebooks/02_model_jet.ipynb`](notebooks/02_model_jet.ipynb) | Research and results, US jet fuel: key findings, the forecasting set-up, seasonality, features, training and tuning without look-ahead, walk-forward backtest against two naive benchmarks, the gain of the burn index by horizon and the index against the EIA weekly figure, current forecast, limitations |
| [`notebooks/02_model_naphtha.ipynb`](notebooks/02_model_naphtha.ipynb) | Research and results, naphtha: the same sections, with the propane/naphtha ablation by horizon, Korea against Japan, read against the seed band, and the months after the closure of Hormuz shown apart |

All four are committed with their outputs and read on GitHub without being run.

## Run

```bash
pip install -r requirements.txt
python -m src.download eia jodi customs aircraft   # optional: refreshes the raw files, no key needed
python -m src.download opensky                     # needs an OpenSky Trino account in pyopensky's settings file
jupyter lab                 # run notebooks/01_data_jet.ipynb then 01_data_naphtha.ipynb: they load the database and build the panel
python -m src.model         # the walk-forward backtests, about an hour, written to the database
jupyter lab                 # run notebooks/02_model_jet.ipynb and 02_model_naphtha.ipynb: they read the backtests
```

Every command runs from the repository root. Notebooks 01 are the same as `python -m src.sql` then `python -m src.data`. The database is committed with the backtests in it, so notebooks 02 run as cloned. `python -m src.flights` rebuilds the OpenAP fuel table, which is committed.

OpenSky's database is queried one UTC day at a time, filtered on its `day` partition, one query at a time. The login stays in pyopensky's own settings file, never in this repository.

## Data architecture

```
data/US_jet_fuel/         EIA monthly and weekly files, OpenSky's aircraft types, the OpenAP fuel
                          table, and opensky/YYYY-MM.csv: daily flights and fuel by aircraft type
data/KR_JP_naphtha/       JODI's demand and imports, Japan's customs by code, country and month
data/catalog.csv          data dictionary: source, series id, definition, unit, frequency, release
data/oil.duckdb
  raw                     every observation as published, with ingested_at; append-only, a revised
                          value is kept next to the one it replaces
  demand, predictors,     clean layer, rebuilt from the latest value of each observation,
  imports                 dated by period_end
  panel                   each driver as readable at each month end, with period_end and available_at
  backtest, params,       written by python -m src.model
  ablation, seeds
src/config.py             paths, publication lags, shocks, aircraft types, customs codes
src/download.py           the five sources
src/flights.py            the burn index: fuel table, distances, the daily Trino query
src/sql.py                raw files -> raw -> clean tables; idempotent
src/data.py               available_at = period_end + lag, merge_asof backward, panel
src/model.py              features, direct multi-horizon XGBoost, benchmarks, walk-forward backtest
```

Publication delays are handled; revisions are not. The sources publish only their latest figures, so the dataset is not a vintage dataset.

## Principal results

RESULTS
