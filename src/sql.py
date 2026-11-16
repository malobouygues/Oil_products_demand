"""Load the raw files into data/oil.duckdb, build the clean tables, and read them back.

    python -m src.sql

Two layers. raw holds every observation as the source published it, one row per file, series
and date, stamped with ingested_at, the time the file was downloaded. Loading is append-only
and idempotent: a row is added only when the observation is new or its value has changed, so
running it twice adds nothing, and a revised number is kept next to the one it replaces.

The clean tables are rebuilt from the latest value of each observation. demand holds the three
targets, predictors every driver in long form; both are dated by period_end, the last day of
the period the number describes, not the day it was published. catalog is the data dictionary.
"""

from datetime import datetime, timezone

import duckdb

from src import config as c

# raw file -> the column holding the date, and the column naming a category when the file is
# long (MPA publishes one row per month and fuel grade, or per month and purpose of call)
FILES = {
    "US_gasoline/eia_monthly.csv": ("date", None),
    "US_gasoline/eia_weekly.csv": ("date", None),
    "US_gasoline/fred.csv": ("date", None),
    "US_distillate_fuel_oil/eia_monthly.csv": ("date", None),
    "US_distillate_fuel_oil/eia_weekly.csv": ("date", None),
    "US_distillate_fuel_oil/fred.csv": ("date", None),
    "US_distillate_fuel_oil/noaa_hdd.csv": ("month", None),
    "SGP_bunker_fuel_oil/mpa_bunker_sales.csv": ("month", "bunker_type"),
    "SGP_bunker_fuel_oil/mpa_vessel_calls.csv": ("month", "purpose_type"),
    "SGP_bunker_fuel_oil/mpa_cargo.csv": ("month", None),
    "SGP_bunker_fuel_oil/mpa_tankers.csv": ("month", None),
    "SGP_bunker_fuel_oil/cpb_world_trade.csv": ("month", None),
}
US = {"gasoline": "US_gasoline", "distillate": "US_distillate_fuel_oil"}


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
            series VARCHAR,         -- the column on the file, after its category if it has one
            obs_date DATE,          -- the date as the source stamps it
            value DOUBLE,
            ingested_at TIMESTAMP   -- UTC, when the file was downloaded
        )""")
    added = {}
    for file, (date, category) in FILES.items():
        path = c.DATA / file
        ingested_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        ingested_at = ingested_at.replace(tzinfo=None, microsecond=0)
        # a month is written 2026-08, a day 2026-08-15: both become a DATE
        obs_date = f"CAST({date} || '-01' AS DATE)" if date == "month" else f"CAST({date} AS DATE)"
        series = f"{category} || ' / ' || series" if category else "series"
        keys = f"{date}, {category}" if category else date
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
    """Rebuild demand, predictors and catalog from the latest value of every raw observation."""
    con.execute("""
        CREATE OR REPLACE VIEW latest AS
        SELECT file, series, obs_date, arg_max(value, ingested_at) AS value,
               MAX(ingested_at) AS ingested_at, COUNT(*) AS vintages
        FROM raw GROUP BY file, series, obs_date""")
    con.execute("CREATE OR REPLACE TABLE demand (product VARCHAR, period_end DATE, value DOUBLE, "
                "unit VARCHAR, PRIMARY KEY (product, period_end))")
    con.execute("CREATE OR REPLACE TABLE predictors (product VARCHAR, variable VARCHAR, "
                "period_end DATE, value DOUBLE, PRIMARY KEY (product, variable, period_end))")

    # US: EIA monthly files are dated mid-month and FRED files on the first day of the month, so
    # both become the last day of that month; weekly files keep the Friday that ends the week
    for product, folder in US.items():
        con.execute(f"""
            INSERT INTO demand
            SELECT '{product}', last_day(obs_date), value, 'kb/d' FROM latest
            WHERE file = '{folder}/eia_monthly.csv' AND series = 'product_supplied'""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT '{product}', series, last_day(obs_date), value FROM latest
            WHERE file = '{folder}/eia_monthly.csv' AND series <> 'product_supplied'
               OR file = '{folder}/fred.csv'""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT '{product}', replace(series, 'product_supplied', 'product_supplied_weekly'),
                   obs_date, value
            FROM latest WHERE file = '{folder}/eia_weekly.csv'""")
    con.execute("""
        INSERT INTO predictors
        SELECT 'distillate', series, last_day(obs_date), value FROM latest
        WHERE file = 'US_distillate_fuel_oil/noaa_hdd.csv'""")

    # Singapore: monthly sales in kt summed over the fuel-oil grades (HSFO, LSFO, ULSFO and
    # their bio-blends; gasoil, LNG and the other alternatives are left out), then divided by
    # the days in the month, so February is not a demand drop and the unit is a daily rate
    # like EIA's
    con.execute("""
        INSERT INTO demand
        SELECT 'bunker', last_day(obs_date) AS period_end, SUM(value) / day(period_end), 'kt/d'
        FROM latest
        WHERE file = 'SGP_bunker_fuel_oil/mpa_bunker_sales.csv' AND series LIKE '%Fuel Oil / %'
        GROUP BY obs_date""")
    con.execute("""
        INSERT INTO predictors
        SELECT 'bunker', CASE series WHEN 'Bunkers / number_of_vessel_calls' THEN 'bunker_calls'
                                     WHEN 'Bunkers / gross_tonnage' THEN 'bunker_calls_gt'
                                     WHEN 'cargo_throughput' THEN 'cargo'
                                     WHEN 'number_of_tankers' THEN 'tankers'
                                     WHEN 'wtr' THEN 'wtr' END AS variable,
               last_day(obs_date), value
        FROM latest
        WHERE file LIKE 'SGP_bunker_fuel_oil/%' AND variable IS NOT NULL""")

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
