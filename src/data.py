"""From the clean DuckDB tables to one point-in-time modelling frame per product.

    python -m src.data   ->  table panel: one row per product, month and driver, with the value
                             readable at that month end and the two dates that prove it
    dataset("jet")       ->  the same panel as a frame, one row per month: demand, then drivers

Every driver observation carries two dates. period_end is the last day of the period the number
describes; available_at is the day it was published. Japan's customs figures for March are out
at the end of April and must not appear in a March row, so the join is merge_asof backward on
available_at: as of each month end, the most recent value already published. The burn index is
daily and readable the day after: a month end carries its mean over the 28 days to the day
before, leaving out the days OpenSky lost (lost_days).

The lags are release-calendar conventions applied to today's figures. They stop a number from
being used before it existed; they do not undo later revisions, because the sources publish no
history of vintages.
"""

import pandas as pd

from src import config as c, sql


def demand(product):
    """The target, indexed by the first day of its month."""
    df = sql.query(f"SELECT date_trunc('month', period_end) AS month, value AS demand "
                   f"FROM demand WHERE product = '{product}' ORDER BY month")
    df["month"] = pd.to_datetime(df["month"])
    return df.set_index("month")


def predictors(product):
    """Each driver as a series at its own frequency, indexed by period_end. The context series
    stay in the database and are not fed to the model."""
    df = sql.query(f"SELECT variable, period_end, value FROM predictors "
                   f"WHERE product = '{product}' AND variable NOT IN {tuple(c.CONTEXT)}")
    df["period_end"] = pd.to_datetime(df["period_end"])
    series = {name: s.set_index("period_end")["value"].sort_index()
              for name, s in df.groupby("variable")}
    if "burn" in series:
        # a day OpenSky lost data for is missing, not low; one day is noisy, 28 make a month
        burn = series["burn"].asfreq("D")
        burn = burn.mask(lost_days().reindex(burn.index, fill_value=False))
        series["burn"] = burn.rolling("28D", min_periods=c.BURN_MIN_DAYS).mean()
    return series


def lost_days():
    """The days of the burn index treated as missing: OpenSky's documented outages, and the days
    whose own coverage says data were lost. Point-in-time: a day is judged on itself and on the
    four weeks before it, all readable the next day."""
    cover = sql.query("SELECT period_end, variable, value FROM predictors WHERE product = 'jet' "
                      "AND variable IN ('flights', 'unknown_airport')")
    cover = cover.pivot(index="period_end", columns="variable", values="value")
    cover.index = pd.to_datetime(cover.index)
    cover = cover.sort_index().asfreq("D")
    usual = cover["flights"].rolling("28D", closed="left", min_periods=7).median()
    lost = ((cover["flights"] < c.LOST_FLIGHTS * usual)
            | (cover["unknown_airport"] > c.LOST_UNKNOWN))
    for start, end in c.OPENSKY_OUTAGES:
        lost[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()] = True
    return lost


def origins(product):
    """The forecast origins: every month from SAMPLE_START to the month end by which the last
    demand figure was published."""
    last = demand(product).index.max() + pd.DateOffset(months=c.DEMAND_LAG[product])
    months = pd.DataFrame({"month": pd.date_range(c.SAMPLE_START, last, freq="MS")})
    # the forecast origin: whatever was published by the last day of the month
    months["month_end"] = months["month"] + pd.offsets.MonthEnd(0)
    return months


def align(product):
    """Every driver as it was readable at each month end, in long form, on a complete monthly
    index so a gap in a source shows as a missing value and not as a missing row."""
    months = origins(product)
    frames = []
    for name, s in predictors(product).items():
        s = s.dropna().rename("value").reset_index()
        if name == "burn":
            s["available_at"] = s["period_end"] + pd.Timedelta(days=c.BURN_LAG_DAYS)
        else:
            s["available_at"] = s["period_end"] + pd.offsets.MonthEnd(c.LAG[name])
        s = s.sort_values("available_at")
        df = pd.merge_asof(months, s, left_on="month_end", right_on="available_at",
                           direction="backward", tolerance=pd.Timedelta(days=c.STALE_DAYS))
        df.insert(0, "variable", name)
        frames.append(df)
    if not frames:  # no driver yet: an empty panel with the same columns and types
        empty = months.iloc[:0].assign(period_end=pd.NaT, value=float("nan"), available_at=pd.NaT)
        empty.insert(0, "variable", "")
        frames = [empty]
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "product", product)
    return df


def dataset(product):
    """The stored panel of one product as the modelling frame: demand, then every driver as
    readable at that month end. Trailing rows without demand are the latest forecast origins."""
    panel = sql.query(f"SELECT month, variable, value FROM panel WHERE product = '{product}'")
    drivers = panel.pivot(index="month", columns="variable", values="value")
    df = pd.DataFrame(index=pd.DatetimeIndex(origins(product)["month"], name="month"))
    df["demand"] = demand(product)["demand"]
    df = df.join(drivers)
    df.columns.name = None
    return df


def build():
    """Align every product and store the result as the table panel."""
    frames = [align(product) for product in c.DEMAND_LAG]
    panel = pd.concat([df for df in frames if not df.empty], ignore_index=True)
    sql.write("panel", panel)
    return sql.query("SELECT product, COUNT(DISTINCT variable) AS drivers, MIN(month) AS first, "
                     "MAX(month) AS last, COUNT(*) AS rows FROM panel GROUP BY 1 ORDER BY 1")


if __name__ == "__main__":
    print(build())
