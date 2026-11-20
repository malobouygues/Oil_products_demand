"""Load the raw files into data/oil.duckdb, build the clean tables, and read them back.

    python -m src.sql

Two layers. raw holds every observation as the source published it, one row per file, series
and date, stamped with ingested_at, the time the file was downloaded. Loading is append-only
and idempotent: a row is added only when the observation is new or its value has changed, so
running it twice adds nothing, and a revised number is kept next to the one it replaces.

The clean tables are rebuilt from the latest value of each observation. demand holds the three
targets, predictors every driver in long form; both are dated by period_end, the last day of
the period the number describes, not the day it was published. imports is Japan's customs
record by product, country and month, in tonnes and thousand yen. catalog is the data dictionary.

The naphtha feature is the CIF unit value of Japan's propane imports over that of its naphtha
imports: what was actually paid at the border, a ratio without a unit, so no exchange rate. It
has three limits. A unit value averages term and spot cargoes priced on different dates. It
follows the market with a lag, since a cargo landed this month was priced weeks before. And it is
a Japanese price used for Korea too, which is defensible only because both buy on the same
North Asian market, from the same sellers.
"""

from datetime import datetime, timezone

import duckdb
import pandas as pd

from src import config as c

# raw files -> the column holding the date, and the columns naming a category when the file is
# long (customs: one row per month, code, country and unit; OpenSky: per day and aircraft type)
FILES = {
    "US_jet_fuel/eia_monthly.csv": ("date", []),
    "US_jet_fuel/eia_weekly.csv": ("date", []),
    "US_jet_fuel/opensky/*.csv": ("day", ["typecode"]),
    "KR_JP_naphtha/jodi.csv": ("month", []),
    "KR_JP_naphtha/customs.csv": ("month", ["code", "country", "unit"]),
}


def query(sql):
    with duckdb.connect(c.DB, read_only=True) as con:
        return con.sql(sql).df()


def write(table, df):
    """Replace one derived table (the panel, the backtests) with a frame."""
    with duckdb.connect(c.DB) as con:
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")


def load(con):
    """Append to raw what each file holds that the table does not: new observations, and
    revised values. Returns the number of rows added per file."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw (
            file VARCHAR,           -- path under data/
            series VARCHAR,         -- the column on the file, after its categories if it has any
            obs_date DATE,          -- the date as the source stamps it
            value DOUBLE,
            ingested_at TIMESTAMP   -- UTC, when the file was downloaded
        )""")
    added = {}
    for pattern, (date, categories) in FILES.items():
        for path in sorted(c.DATA.glob(pattern)):
            file = str(path.relative_to(c.DATA))
            ingested_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            ingested_at = ingested_at.replace(tzinfo=None, microsecond=0)
            # a month is written 2026-08, a day 2026-08-15: both become a DATE
            obs_date = (f"CAST({date} || '-01' AS DATE)" if date == "month"
                        else f"CAST({date} AS DATE)")
            series = " || ' / ' || ".join([*categories, "series"])
            keys = ", ".join([date, *categories])
            before = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
            con.execute(f"""
                INSERT INTO raw
                SELECT ? AS file, new.series, new.obs_date, new.value, ? AS ingested_at
                FROM (SELECT {series} AS series, {obs_date} AS obs_date, value
                      FROM (UNPIVOT read_csv('{path}') ON COLUMNS(* EXCLUDE ({keys}))
                            INTO NAME series VALUE value)) AS new
                LEFT JOIN (SELECT series, obs_date, arg_max(value, ingested_at) AS value
                           FROM raw WHERE file = ? GROUP BY series, obs_date) AS old
                  ON old.series = new.series AND old.obs_date = new.obs_date
                WHERE old.value IS DISTINCT FROM new.value""", [file, ingested_at, file])
            added[file] = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0] - before
    return added


def clean(con):
    """Rebuild demand, predictors, imports and catalog from the latest value of every raw
    observation."""
    con.execute("""
        CREATE OR REPLACE VIEW latest AS
        SELECT file, series, obs_date, arg_max(value, ingested_at) AS value,
               MAX(ingested_at) AS ingested_at, COUNT(*) AS vintages
        FROM raw GROUP BY file, series, obs_date""")
    con.execute("CREATE OR REPLACE TABLE demand (product VARCHAR, period_end DATE, value DOUBLE, "
                "unit VARCHAR, PRIMARY KEY (product, period_end))")
    con.execute("CREATE OR REPLACE TABLE predictors (product VARCHAR, variable VARCHAR, "
                "period_end DATE, value DOUBLE, PRIMARY KEY (product, variable, period_end))")

    # US jet: EIA monthly files are dated mid-month, so they become the last day of the month;
    # the weekly file keeps the Friday that ends the week
    con.execute("""
        INSERT INTO demand
        SELECT 'jet', last_day(obs_date), value, 'kb/d' FROM latest
        WHERE file = 'US_jet_fuel/eia_monthly.csv' AND series = 'product_supplied'""")
    con.execute("""
        INSERT INTO predictors
        SELECT 'jet', 'product_supplied_weekly', obs_date, value FROM latest
        WHERE file = 'US_jet_fuel/eia_weekly.csv'""")
    # the burn index, kb/d: the fuel of the day's departures from US airports, every type
    # summed; the day's departures; and the share of flights OpenSky left without an airport
    con.execute(f"""
        INSERT INTO predictors
        SELECT 'jet', variable, obs_date, value FROM (
            SELECT obs_date,
                   SUM(value) FILTER (WHERE series LIKE '%/ fuel_kg') / {c.KG_PER_BARREL} / 1000
                       AS burn,
                   SUM(value) FILTER (WHERE series LIKE '%/ departures') AS flights,
                   SUM(value) FILTER (WHERE series LIKE '%/ unknown_airport')
                       / SUM(value) FILTER (WHERE series LIKE '%/ departures'
                                            OR series LIKE '%/ unknown_airport') AS unknown_airport
            FROM latest WHERE file LIKE 'US_jet_fuel/opensky/%' GROUP BY obs_date)
        UNPIVOT (value FOR variable IN (burn, flights, unknown_airport))""")

    # Korea and Japan: JODI's total demand in kb/d, naphtha the target, LPG context
    for country in ("kr", "jp"):
        con.execute(f"""
            INSERT INTO demand
            SELECT 'naphtha_{country}', last_day(obs_date), value, 'kb/d' FROM latest
            WHERE file = 'KR_JP_naphtha/jodi.csv' AND series = 'naphtha_{country}'
              AND value IS NOT NULL""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'naphtha_{country}', 'lpg', last_day(obs_date), value FROM latest
            WHERE file = 'KR_JP_naphtha/jodi.csv' AND series = 'lpg_{country}'
              AND value IS NOT NULL""")

    # Japan's imports by product, country and month: naphtha in kg, propane and butane in
    # tonnes, values in thousand yen
    codes = pd.DataFrame(list(c.CUSTOMS.items()), columns=["code", "product"])
    con.execute("""
        CREATE OR REPLACE TABLE imports AS
        SELECT last_day(obs_date) AS period_end, product, country,
               SUM(value * CASE unit WHEN 'KG' THEN 0.001 ELSE 1 END)
                   FILTER (WHERE measure = 'quantity') AS tonnes,
               SUM(value) FILTER (WHERE measure = 'value') AS kyen
        FROM (SELECT obs_date, value, split_part(series, ' / ', 1) AS code,
                     CAST(split_part(series, ' / ', 2) AS INTEGER) AS country,
                     split_part(series, ' / ', 3) AS unit, split_part(series, ' / ', 4) AS measure
              FROM latest WHERE file = 'KR_JP_naphtha/customs.csv')
        JOIN codes USING (code)
        GROUP BY ALL""")
    # the feedstock ratio: CIF unit value of propane over that of naphtha, the price Japan
    # actually paid for each, a ratio without a unit; and the Gulf's share of each, in %. The
    # customs file has no row for a country that shipped nothing: no Gulf row is a share of zero
    gulf = f"country IN {tuple(c.GULF)}"
    for product in ("naphtha_kr", "naphtha_jp"):
        con.execute(f"""
            INSERT INTO predictors
            SELECT '{product}', variable, period_end, value FROM (
                SELECT period_end,
                       (SUM(kyen) FILTER (WHERE product = 'propane')
                        / SUM(tonnes) FILTER (WHERE product = 'propane'))
                       / (SUM(kyen) FILTER (WHERE product = 'naphtha')
                          / SUM(tonnes) FILTER (WHERE product = 'naphtha')) AS propane_naphtha,
                       100 * COALESCE(SUM(tonnes) FILTER (WHERE product = 'naphtha' AND {gulf}), 0)
                       / SUM(tonnes) FILTER (WHERE product = 'naphtha') AS gulf_naphtha,
                       100 * COALESCE(SUM(tonnes) FILTER (WHERE product = 'propane' AND {gulf}), 0)
                       / SUM(tonnes) FILTER (WHERE product = 'propane') AS gulf_propane
                FROM imports GROUP BY period_end)
            UNPIVOT (value FOR variable IN (propane_naphtha, gulf_naphtha, gulf_propane))""")

    con.execute(f"CREATE OR REPLACE TABLE catalog AS "
                f"SELECT * FROM read_csv('{c.DATA}/catalog.csv')")


def build():
    """Load the raw files and rebuild the clean tables."""
    with duckdb.connect(c.DB) as con:
        added = load(con)
        clean(con)
    for file, rows in added.items():
        print(f"{rows:>7,} rows added to raw from {file}")


if __name__ == "__main__":
    build()
    print(query("SELECT product, unit, MIN(period_end) AS first, MAX(period_end) AS last, "
                "COUNT(*) AS months FROM demand GROUP BY 1, 2 ORDER BY 1"))
