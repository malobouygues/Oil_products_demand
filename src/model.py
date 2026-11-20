"""Direct multi-horizon XGBoost against the seasonal naive: from origin t, one model per step
forecasts demand(t + step), step = 1 ... 12.

The seasonal naive repeats the same calendar month of the latest year published at t. The
model predicts what it misses, y = log demand(t + step) - log naive, so a tree never has to
extrapolate a trend it has not seen, and beating the naive is the objective itself. The
forecast is the naive times exp(prediction).

Demand is published `lag` months late (2 for EIA and JODI), so the newest print at t is
d(t - lag); the drivers were aligned point-in-time in data.py. Nothing is scaled: a tree splits
on thresholds, and a monotone rescaling leaves every split where it is.

    python -m src.model    # the backtests, the driver ablation and the seed band, by horizon
"""

import math

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import config as c, data, sql

HORIZON = 12
FORECASTS = ["xgboost", "naive", "last"]  # the model and its two free benchmarks

PARAMS = dict(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
              random_state=42, n_jobs=4, verbosity=0)
# eight fits on purpose: some 200 monthly origins, and a large search on 200 rows selects noise
GRID = [{"learning_rate": lr, "max_depth": d, "n_estimators": n}
        for lr in (0.03, 0.1) for d in (3, 5) for n in (200, 600)]


def back(step, lag):
    """Months from a target to its naive reference: the first multiple of 12 that reaches a
    month already published at the origin."""
    return 12 * math.ceil((step + lag) / 12)


def features(df, lag, step):
    """X at origin t, in year-on-year terms: the last twelve published demand prints and every
    driver as of t, each against its value a year earlier. y: log demand at t + step over the
    seasonal naive."""
    d = np.log(df["demand"])
    X = pd.DataFrame({f"yoy_lag{k}": d.shift(k) - d.shift(k + 12) for k in range(lag, lag + 12)})
    for name in df.columns.drop("demand"):
        X[name] = np.log(df[name] / df[name].shift(12))
    if "propane_naphtha" in df:
        # crackers switch feed past a threshold, propane some 50 USD/t under naphtha: the level
        X["propane_naphtha_level"] = df["propane_naphtha"]
    return X, d.shift(-step) - d.shift(back(step, lag) - step)


def excluded(months, product):
    """Whether each month falls in one of the product's shocks: 2020, and Hormuz for naphtha."""
    months = pd.DatetimeIndex(months)
    return np.any([(months >= start) & (months <= end) for start, end in c.EXCLUDE[product]],
                  axis=0)


def trainable(y, step, product, origin):
    """Origins from TRAIN_START whose target had been published by the forecast origin, with
    neither the origin, the target nor its naive reference month in one of the product's
    shocks."""
    lag = c.DEMAND_LAG[product]
    target = y.index + pd.DateOffset(months=step)
    published = target + pd.DateOffset(months=lag) <= pd.Timestamp(origin)
    reference = target - pd.DateOffset(months=back(step, lag))
    shock = (excluded(y.index, product) | excluded(target, product)
             | excluded(reference, product))
    return (y.index >= c.TRAIN_START) & published & y.notna() & ~shock


def cv_score(X, y, step, lag, params, folds=3):
    """Expanding window on the trainable rows: each block is scored by a model fitted only on
    the rows whose target was published by the block's first origin, the real-time rule."""
    edges = [len(X) * k // (folds + 1) for k in range(folds + 2)]
    errors = []
    for a, b in zip(edges[1:-1], edges[2:]):
        fit = X.index + pd.DateOffset(months=step + lag) <= X.index[a]
        model = XGBRegressor(**{**PARAMS, **params}).fit(X[fit], y[fit])
        errors.append(np.mean((y.iloc[a:b] - model.predict(X.iloc[a:b])) ** 2))
    return np.mean(errors)


def tune(df, product, origin):
    """Parameters chosen at the longest step and reused for the shorter ones."""
    lag = c.DEMAND_LAG[product]
    X, y = features(df, lag, HORIZON)
    rows = trainable(y, HORIZON, product, origin)
    return min(GRID, key=lambda p: cv_score(X[rows], y[rows], HORIZON, lag, p))


def tuned(product):
    """The parameters the backtest chose for one product, read back from DuckDB."""
    row = sql.query(f"SELECT * FROM params WHERE product = '{product}'").iloc[0]
    return {"learning_rate": float(row["learning_rate"]), "max_depth": int(row["max_depth"]),
            "n_estimators": int(row["n_estimators"])}


def seasonal_naive(df, lag, origin):
    """Same calendar month in the latest year published at the origin: the free benchmark."""
    months = pd.date_range(pd.Timestamp(origin) + pd.DateOffset(months=1), periods=HORIZON,
                           freq="MS")
    reference = [m - pd.DateOffset(months=back(step, lag)) for step, m in enumerate(months, 1)]
    return pd.Series(df.loc[reference, "demand"].to_numpy(), index=months)


def forecast(df, product, origin, params):
    """Demand at t+1 ... t+12 from origin t, one model per step."""
    lag = c.DEMAND_LAG[product]
    origin, deviation = pd.Timestamp(origin), []
    for step in range(1, HORIZON + 1):
        X, y = features(df, lag, step)
        rows = trainable(y, step, product, origin)
        fit = XGBRegressor(**{**PARAMS, **params}).fit(X[rows], y[rows])
        deviation.append(fit.predict(X.loc[[origin]])[0])
    return seasonal_naive(df, lag, origin) * np.exp(deviation)


def backtest(df, product, params=None):
    """Walk forward: forecast from every month end since BACKTEST_START, each time refitting on
    what was published by then. Parameters are tuned once, on what was published by the first
    origin, so the backtest years play no part in choosing them. Passing params reruns the same
    test without tuning, for the driver ablation and the seeds."""
    lag = c.DEMAND_LAG[product]
    origins = pd.date_range(c.BACKTEST_START, df["demand"].last_valid_index(), freq="MS")
    params, rows = params or tune(df, product, origins[0]), []
    for origin in origins:
        path = forecast(df, product, origin, params)
        reference = [m - pd.DateOffset(months=back(s, lag)) for s, m in enumerate(path.index, 1)]
        rows.append(pd.DataFrame({
            "origin": origin, "step": range(1, HORIZON + 1), "reference": reference,
            "actual": df["demand"].reindex(path.index).to_numpy(),
            "xgboost": path.to_numpy(), "naive": seasonal_naive(df, lag, origin).to_numpy(),
            # the other free benchmark: the latest print published at the origin, held flat
            "last": df.loc[origin - pd.DateOffset(months=lag), "demand"]},
            index=path.index.rename("month")))
    bt = pd.concat(rows).reset_index()
    bt.insert(0, "product", product)
    return bt, params


def scored(bt, shocks=False):
    """The forecasts that can be marked: target already published and, unless shocks is True, no
    origin, target or naive reference month in one of the product's shocks."""
    bt = bt[bt["actual"].notna()]
    if shocks:
        return bt
    keep = []
    for product, rows in bt.groupby("product"):
        shock = (excluded(rows["origin"], product) | excluded(rows["month"], product)
                 | excluded(rows["reference"], product))
        keep.append(rows[~shock])
    return pd.concat(keep)


def score(bt, by="step"):
    """Each forecast against the actual, by horizon unless told otherwise. MAE and RMSE in the
    unit of demand, WAPE (sum of absolute errors over sum of actuals) and bias (mean error over
    mean actual) in %."""
    error = bt[FORECASTS].sub(bt["actual"], axis=0)
    actual = bt["actual"].groupby(bt[by])
    return pd.concat({
        "mae": error.abs().groupby(bt[by]).mean(),
        "rmse": (error ** 2).groupby(bt[by]).mean() ** 0.5,
        "wape": 100 * error.abs().groupby(bt[by]).sum().div(actual.sum(), axis=0),
        "bias": 100 * error.groupby(bt[by]).sum().div(actual.sum(), axis=0)}, axis=1)


def wape(bt):
    """One number per forecast: WAPE in %, averaged over the twelve horizons."""
    return score(scored(bt))["wape"].mean()


if __name__ == "__main__":
    # About an hour, so notebooks 02 read the tables instead. Rerun when the data or the model
    # change.
    backtests, chosen, ablation, seeds = [], [], [], []
    for product in c.DEMAND_LAG:
        print(product)
        df = data.dataset(product)
        bt, params = backtest(df, product)
        backtests.append(bt)
        chosen.append({"product": product, **params})
        # two windows: every origin, and the origins where every driver has a year-on-year
        # value, since a driver can start late (OpenSky's flights are online from 2023 only)
        X = features(df, c.DEMAND_LAG[product], 1)[0]
        drivers = X.drop(columns=X.filter(like="yoy_lag").columns).dropna()
        windows = sorted({pd.Timestamp(c.BACKTEST_START),
                          max(drivers.index.min(), pd.Timestamp(c.BACKTEST_START))})
        # the drivers out, same hyperparameters, every origin: the error by horizon
        variants = {"all drivers": bt}
        if df.shape[1] > 1:
            variants["lags only"] = backtest(df[["demand"]], product, params)[0]
        # how much the score moves for no reason: same features, other seeds
        runs = {seed: backtest(df, product, {**params, "random_state": seed})[0]
                for seed in (1, 2, 3)}
        for since in windows:
            for name, result in variants.items():
                by_step = score(scored(result[result["origin"] >= since]))["wape"]
                ablation.append(by_step.assign(product=product, variant=name, since=since)
                                .reset_index())
            for seed, result in runs.items():
                by_step = score(scored(result[result["origin"] >= since]))["wape"]
                seeds.append(by_step.assign(product=product, seed=seed, since=since)
                             .reset_index())
    sql.write("backtest", pd.concat(backtests, ignore_index=True))
    sql.write("params", pd.DataFrame(chosen))
    sql.write("ablation", pd.concat(ablation, ignore_index=True))
    sql.write("seeds", pd.concat(seeds, ignore_index=True))
