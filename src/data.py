"""From the clean DuckDB tables to one point-in-time modelling frame per product.

    python -m src.data   ->  table panel: one row per product, month and driver, with the value
                             readable at that month end and the two dates that prove it
    dataset("gasoline")  ->  the same panel as a frame, one row per month: demand, then drivers

Every driver observation carries two dates. period_end is the last day of the period the number
describes; available_at is the day it was published, period_end plus the source's lag. Truck
tonnage for July is not out until October and must not appear in an August or September row,
so the join is merge_asof backward on available_at: as of each month end, the most recent value
already published. Drivers become step functions rather than smooth lines, which costs a tree
nothing.

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
    """Each driver as a series at its own frequency, indexed by period_end. Exports and stocks
    stay in the database as supply-side context and are not fed to the model."""
    df = sql.query(f"SELECT variable, period_end, value FROM predictors "
                   f"WHERE product = '{product}' AND variable NOT IN ('exports', 'stocks')")
    df["period_end"] = pd.to_datetime(df["period_end"])
    series = {name: s.set_index("period_end")["value"].sort_index()
              for name, s in df.groupby("variable")}
    if product in ("gasoline", "distillate"):
        # USD/gal in 1982-84 dollars, the CPI base
        series["real_price"] = series.pop("retail_price") / series.pop("cpi") * 100
        # one week is noisy: EIA itself reports the four-week average
        series["product_supplied_4wk"] = series.pop("product_supplied_weekly").rolling("28D").mean()
    return series


def align(product):
    """Every driver as it was readable at each month end, in long form, on a complete monthly
    index so a gap in a source shows as a missing value and not as a missing row. The index
    runs to the latest origin: the month end by which the last demand figure was published."""
    last_origin = demand(product).index.max() + pd.DateOffset(months=c.DEMAND_LAG[product])
    months = pd.DataFrame({"month": pd.date_range(c.SAMPLE_START, last_origin, freq="MS")})
    # the forecast origin: whatever was published by the last day of the month
    months["month_end"] = months["month"] + pd.offsets.MonthEnd(0)

    frames = []
    for name, s in predictors(product).items():
        s = s.dropna().rename("value").reset_index()
        if name == "product_supplied_4wk":
            s["available_at"] = s["period_end"] + pd.Timedelta(days=c.WEEKLY_LAG_DAYS)
        else:
            s["available_at"] = s["period_end"] + pd.offsets.MonthEnd(c.LAG[name])
        s = s.sort_values("available_at")
        df = pd.merge_asof(months, s, left_on="month_end", right_on="available_at",
                           direction="backward", tolerance=pd.Timedelta(days=c.STALE_DAYS))
        df.insert(0, "variable", name)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "product", product)
    return df


def dataset(product):
    """The stored panel of one product as the modelling frame: demand, then every driver as
    readable at that month end. Trailing rows without demand are the latest forecast origins."""
    panel = sql.query(f"SELECT month, variable, value FROM panel WHERE product = '{product}'")
    df = panel.pivot(index="month", columns="variable", values="value")
    df.columns.name = None
    df.insert(0, "demand", demand(product)["demand"])
    return df


def build():
    """Align every product and store the result as the table panel."""
    panel = pd.concat([align(product) for product in c.DEMAND_LAG], ignore_index=True)
    sql.write("panel", panel)
    return sql.query("SELECT product, COUNT(DISTINCT variable) AS drivers, MIN(month) AS first, "
                     "MAX(month) AS last, COUNT(*) AS rows FROM panel GROUP BY 1 ORDER BY 1")


if __name__ == "__main__":
    print(build())
