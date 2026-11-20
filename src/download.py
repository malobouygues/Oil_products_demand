"""Fetch the raw inputs into data/, one folder per product. Only OpenSky needs an account.

    python -m src.download                  # everything
    python -m src.download eia customs      # only these

Requests go through curl rather than urllib: some of these servers hang up on Python's TLS
handshake, while curl on the same URL answers in half a second.

OpenSky's historical flights come from its Trino database, one UTC day per query and one query
at a time (the limit is two), filtered on the day partition. The login stays in pyopensky's own
configuration file, never here. A day already on file is not asked for again.
"""

import html
import io
import re
import subprocess
import sys
import zipfile

import pandas as pd

from src import config as c, flights

CURL = ["curl", "-sS", "-L", "--retry", "3", "--max-time", "900", "-A", "Mozilla/5.0"]

# EIA id -> column. Kerosene-type jet fuel product supplied, kb/d: the monthly is the target, the
# weekly the same concept a week after the fact, kept as the yardstick of the burn index
EIA = {"monthly": {"MKJUPUS2": "product_supplied"}, "weekly": {"WKJUPUS2": "product_supplied"}}

JODI = "https://www.jodidata.org/_resources/files/downloads/oil-data/world_secondary_csv.zip"

# e-Stat files of Japan's trade statistics: commodity by country, imports, one file a year
# holding its months; chapters 25-27 are table 5 of 22
ESTAT = ("https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&data=1&metadata=1"
         "&cycle=1&toukei=00350300&tstat=000001013141&tclass1=000001013180&tclass2=000001013182"
         "&cycle_facet=cycle&tclass3val=0&year=")
ESTAT_FILE = "https://www.e-stat.go.jp/stat-search/file-download?fileKind=1&statInfId="

AIRCRAFT = ("https://s3.opensky-network.org/data-samples/metadata/"
            "aircraft-database-complete-2025-08.csv")


def get(url):
    return subprocess.run(CURL + [url], capture_output=True, check=True).stdout


def eia():
    """EIA history files are .xls with one series each, named <id>m or <id>w, dated mid-month or
    on the Friday that ends the week."""
    for freq, series in EIA.items():
        columns = []
        for code, column in series.items():
            url = f"https://www.eia.gov/dnav/pet/hist_xls/{code}{freq[0]}.xls"
            book = pd.read_excel(io.BytesIO(get(url)), sheet_name="Data 1", header=2)
            columns.append(book.set_axis(["date", column], axis=1).set_index("date")[column])
        pd.concat(columns, axis=1, sort=True).to_csv(c.JET / f"eia_{freq}.csv")


def jodi():
    """Total demand (TOTDEMO) of naphtha and LPG in Korea and Japan, in kb/d and in the
    kilotonnes they are reported in, and Japan's naphtha imports (TOTIMPSB), from JODI's world
    database, which republishes its whole history every month. Missing values come as '-', 'x'
    or 'N/A'."""
    archive = zipfile.ZipFile(io.BytesIO(get(JODI)))
    rows = []
    with archive.open(archive.namelist()[0]) as f:
        for chunk in pd.read_csv(f, dtype=str, chunksize=1_000_000):
            rows.append(chunk[chunk["REF_AREA"].isin(["KR", "JP"])
                              & chunk["ENERGY_PRODUCT"].isin(["NAPHTHA", "LPG"])
                              & chunk["FLOW_BREAKDOWN"].isin(["TOTDEMO", "TOTIMPSB"])
                              & chunk["UNIT_MEASURE"].isin(["KBD", "KTONS"])])
    df = pd.concat(rows)
    demand = (df["FLOW_BREAKDOWN"] == "TOTDEMO") | ((df["REF_AREA"] == "JP")
                                                   & (df["ENERGY_PRODUCT"] == "NAPHTHA")
                                                   & (df["UNIT_MEASURE"] == "KTONS"))
    df = df[demand].copy()
    df["series"] = (df["ENERGY_PRODUCT"].str.lower() + "_" + df["REF_AREA"].str.lower()
                    + df["FLOW_BREAKDOWN"].map({"TOTDEMO": "", "TOTIMPSB": "_imports"})
                    + df["UNIT_MEASURE"].map({"KBD": "", "KTONS": "_kt"}))
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    wide = df.pivot(index="TIME_PERIOD", columns="series", values="value")
    wide.rename_axis("month").to_csv(c.NAPHTHA / "jodi.csv")


def customs():
    """Japan's imports of naphtha, propane and butane by country and month, kg or tonnes and
    thousand yen, from the year files on e-Stat. Past years are the definitive figures, the
    current year the latest release: final months, then the newest as a preliminary 9-digit
    figure. Each file's publication date is kept next to it."""
    rows, releases = [], []
    for year in range(2009, pd.Timestamp.now().year + 1):
        page = get(ESTAT + f"{year}0").decode("utf-8")
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
        # the title of table 5 and its publication date
        pattern = (r"(\d{4}年(?:\d+月)?分 品別国別表 \([^)]*\) \S+ 25-27類) 調査年月 \S+ "
                   r"公開（更新）日 (\d{4}-\d{2}-\d{2})")
        title, published = re.search(pattern, text).groups()
        ids = re.findall(r"statInfId=(\d+)&(?:amp;)?fileKind=1", page)
        file_id = ids[4]  # the fifth table: chapters 25-27
        book = pd.read_csv(io.BytesIO(get(ESTAT_FILE + file_id)), dtype={"HS": str})
        book["HS"] = book["HS"].str.strip("'")
        book = book[book["HS"].isin(list(c.CUSTOMS))]
        # after the year totals, three columns a month: quantity 1, quantity 2, value. Read by
        # position: older files spell April "Apl"
        for month in range(1, 13):
            first = 9 + 3 * (month - 1)
            rows.append(pd.DataFrame({"month": f"{year}-{month:02d}", "code": book["HS"],
                                      "country": book["Country"], "unit": book["Unit2"],
                                      "quantity": book.iloc[:, first + 1],
                                      "value": book.iloc[:, first + 2]}))
        releases.append({"year": year, "title": title, "published": published,
                         "statInfId": file_id})
    df = pd.concat(rows)
    df = df[(df["quantity"] > 0) | (df["value"] > 0)]  # a month not yet out is all zeros
    df.to_csv(c.NAPHTHA / "customs.csv", index=False)
    pd.DataFrame(releases).to_csv(c.NAPHTHA / "customs_releases.csv", index=False)


def aircraft():
    """OpenSky's aircraft database, the rows of the commercial jet types: icao24 to type code.
    One snapshot, August 2025, used for every year."""
    df = pd.read_csv(io.BytesIO(get(AIRCRAFT)), quotechar="'", dtype=str,
                     usecols=["icao24", "typecode", "model", "operator", "registration"])
    df = df[df["typecode"].isin(list(c.COMMERCIAL_JETS))]
    df.to_csv(c.JET / "aircraft.csv", index=False)


def opensky():
    """The day's flights touching a US airport, by commercial type, one file per month. Needs a
    Trino account in pyopensky's settings file. Newest day first, so a rerun picks up where the
    last one stopped; it stops at the first day OpenSky returns no flight for, the edge of what
    it holds. The days of its documented outages are not asked for: they are dropped anyway."""
    from pyopensky.trino import Trino
    trino = Trino()
    aircraft = pd.read_csv(c.JET / "aircraft.csv", dtype=str)
    table = pd.read_csv(flights.FUEL_TABLE)
    folder = c.JET / "opensky"
    folder.mkdir(exist_ok=True)
    lost = {day for start, end in c.OPENSKY_OUTAGES
            for day in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())}
    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    for day in pd.date_range(c.OPENSKY_START, yesterday - pd.Timedelta(days=1))[::-1]:
        path = folder / f"{day:%Y-%m}.csv"
        if day in lost or (path.exists() and f"{day:%Y-%m-%d}" in pd.read_csv(path)["day"].values):
            continue
        rows = trino.query(flights.query(day), cached=False)
        if rows.empty:
            print(f"{day:%Y-%m-%d}: no flight returned, stopping")
            break
        df = flights.burn(rows, aircraft, table)
        df.insert(0, "day", f"{day:%Y-%m-%d}")
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"{day:%Y-%m-%d}: {len(rows):,} flights, {df['departures'].sum():,} departures",
              flush=True)


SOURCES = {"eia": eia, "jodi": jodi, "customs": customs, "aircraft": aircraft,
           "opensky": opensky}

if __name__ == "__main__":
    for folder in (c.JET, c.NAPHTHA):
        folder.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or SOURCES:
        print(name)
        SOURCES[name]()
