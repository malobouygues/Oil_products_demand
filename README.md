# Forecasting oil demand with XGBoost

Can a gradient-boosted tree, fed lagged demand and macro predictors, forecast monthly oil
demand twelve months ahead better than a simple timeseries ?

Three products, each where demand is measured directly: US gasoline and US distillate fuel oil (EIA product supplied) and Singapore bunker fuel oil (MPA sales).

## Notebooks

| Notebook | Content |
|---|---|
| [`notebooks/01_data.ipynb`](notebooks/01_data.ipynb) | Data pipeline: raw files into DuckDB, data dictionary, validation, publication lags tested against a real download, revisions measured between two downloads, `merge_asof` alignment, no-look-ahead assertions, final dataset |
| [`notebooks/02_model.ipynb`](notebooks/02_model.ipynb) | Research and results: key findings, the forecasting set-up, seasonality, features, training and tuning without look-ahead, walk-forward backtest against two naive benchmarks, driver ablation, current forecast, limitations |

Both are committed with their outputs and read on GitHub without being run.

## Run

```bash
pip install -r requirements.txt
python -m src.download      # optional: refreshes the raw files, no API key needed
jupyter lab                 # run notebooks/01_data.ipynb: it loads the database and builds the panel
python -m src.model         # the walk-forward backtests, about 50 minutes, written to the database
jupyter lab                 # run notebooks/02_model.ipynb: it reads the backtests, about one minute
```

Every command runs from the repository root. Notebook 01 is the same as `python -m src.sql` then `python -m src.data`. The database is committed with the backtests in it, so notebook 02 runs as cloned.

## Data architecture

```
data/<product>/*.csv      raw files as published by EIA, FRED, NOAA, MPA and CPB
data/catalog.csv          data dictionary: source, series id, definition, unit, frequency, release
data/oil.duckdb
  raw                     every observation as published, with ingested_at; append-only, a revised
                          value is kept next to the one it replaces
  demand, predictors      clean layer, rebuilt from the latest value of each observation,
                          dated by period_end
  panel                   each driver as readable at each month end, with period_end and available_at
  backtest, params,       written by python -m src.model
  ablation, seeds
src/config.py             paths, publication lags, sample and backtest dates
src/download.py           the five sources
src/sql.py                raw files -> raw -> clean tables; idempotent
src/data.py               available_at = period_end + lag, merge_asof backward, panel
src/model.py              features, direct multi-horizon XGBoost, benchmarks, walk-forward backtest
```

Publication delays are handled; revisions are not. The sources publish only their latest figures, so the dataset is not a vintage dataset. Notebook 01 measures one month of revisions from the two downloads the database holds.

## Principal results

Walk-forward backtest, 125 monthly origins from January 2015 to August 2026, twelve horizons each. WAPE in %, averaged over the horizons; 2020 is left out of the headline and shown separately in notebook 02.

| | XGBoost | Seasonal naive | Last value |
|---|---|---|---|
| US gasoline | 1.77 | 1.82 | 3.53 |
| US distillate fuel oil | 3.77 | 3.84 | 4.57 |
| Singapore bunker fuel oil | 6.91 | 6.81 | 6.10 |

The model matches the seasonal naive within its own seed noise on the two US products, with a small gain at one to two months ahead, and is worse than both benchmarks on bunker. The answer to the question is no.
