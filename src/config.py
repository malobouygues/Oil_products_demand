"""Paths, the products, every publication-lag assumption, and the physical assumptions behind the
kerosene burn index and the naphtha feedstock ratio."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "oil.duckdb"

# raw files, one folder per product
JET = DATA / "US_jet_fuel"
NAPHTHA = DATA / "KR_JP_naphtha"  # Korea and Japan share their sources

# --- when a number becomes readable -------------------------------------------
# Months from the end of the period a number describes to the day it is published, from each
# source's release calendar: available_at = period_end + LAG month ends.
# The targets: EIA's Petroleum Supply Monthly comes out at the end of the second month after;
# JODI updates its world database around the 20th of the second month after.
DEMAND_LAG = {"jet": 2, "naphtha_kr": 2, "naphtha_jp": 2}
LAG = {
    "propane_naphtha": 1,  # Japanese customs, 9-digit import figures at the end of the next month
}
# EIA's Weekly Petroleum Status Report: the Wednesday after the Friday that ends the week,
# Thursday in holiday weeks
WEEKLY_LAG_DAYS = 6
# The burn index of day D is computed from the flights OpenSky closes that day: readable on D+1
BURN_LAG_DAYS = 1

# Past this a value is stale rather than late: it goes missing, which XGBoost handles and which
# is honest about a series that has stopped publishing. The slowest driver, the customs ratio,
# is a month late, 62 days at most: 125 days leaves room for two skipped releases and no more.
STALE_DAYS = 125

# Drivers kept in the database as context and not fed to the model: the weekly EIA estimate (the
# jet target a week after the fact, the yardstick of the burn index), each country's LPG demand
# and the Gulf's share of Japan's naphtha and propane imports
CONTEXT = ["product_supplied_weekly", "lpg", "gulf_naphtha", "gulf_propane", "flights",
           "unknown_airport"]

# --- the sample and the experiment ---------------------------------------------
SAMPLE_START = "2000-01-01"
# Training rows start in 2001, the first year with a year-earlier value for the year-on-year
# features and the naive reference. JODI's naphtha starts in 2009: its rows start there.
TRAIN_START = "2001-01-01"
# The backtest forecasts from every month end since 2015
BACKTEST_START = "2015-01-01"
# Shocks, not patterns to learn: no training row has its origin, target or naive reference
# month in them. The backtest still forecasts through them; the headline scores leave those
# forecasts out and notebooks 02 show them separately. 2020 for every product. For naphtha also
# the closure of Hormuz, from March 2026: the strait closed in the first days of the month and
# Japan's naphtha imports from the Gulf fell that month (notebook 01_data_naphtha).
COVID = ("2020-01-01", "2020-12-31")
HORMUZ = ("2026-03-01", "2099-12-31")  # not over at the last download
EXCLUDE = {"jet": [COVID], "naphtha_kr": [COVID, HORMUZ], "naphtha_jp": [COVID, HORMUZ]}

# --- the kerosene burn index -----------------------------------------------------
# OpenSky's flights_data4 from 1 January 2016, one UTC day per query. US airports by ICAO prefix:
# the contiguous states, Alaska, Hawaii.
OPENSKY_START = "2016-01-01"
US_AIRPORTS = ("K", "PA", "PH")
# The commercial jets counted, by ICAO type code -> the OpenAP model that burns for it: the type
# itself where OpenAP has it, through its synonyms for the 767-200, 777-200LR, MD-11 and CRJ200,
# and the nearest model by size otherwise (marked). OpenAP's synonyms for the A300 and A310 burn
# implausibly (10 and 2.8 kg/km): both go to the 767-300, the nearest by take-off weight.
COMMERCIAL_JETS = {
    "A19N": "a19n", "A20N": "a20n", "A21N": "a21n", "A318": "a318", "A319": "a319",
    "A320": "a320", "A321": "a321", "A332": "a332", "A333": "a333", "A339": "a333",  # A330-900
    "A343": "a343", "A346": "a343", "A359": "a359", "A35K": "a359", "A388": "a388",  # A350-1000
    "A306": "b763", "A310": "b763",  # A300-600 and A310
    "BCS1": "e195", "BCS3": "a319",  # A220-100 and -300, by take-off weight
    "B712": "e195",                  # 717
    "B733": "b733", "B734": "b734", "B735": "b735", "B736": "b737", "B737": "b737",
    "B738": "b738", "B739": "b739", "B37M": "b37m", "B38M": "b38m", "B39M": "b39m",
    "B3XM": "b3xm", "B744": "b744", "B748": "b748", "B752": "b752", "B753": "b752",
    "B762": "b762", "B763": "b763", "B764": "b763", "B772": "b772", "B77L": "b77l",
    "B773": "b773", "B77W": "b77w", "B788": "b788", "B789": "b789", "B78X": "b789",
    "MD11": "md11", "MD88": "b738", "MD90": "b738",  # MD-80/90 by size
    "CRJ2": "crj2", "CRJ7": "e170", "CRJ9": "crj9", "CRJX": "crj9",
    "E135": "e145", "E145": "e145", "E170": "e170", "E75L": "e75l", "E75S": "e75l",
    "E190": "e190", "E195": "e195", "E290": "e290", "E295": "e195",
}
# Fuel against distance, computed once per model with OpenAP: a complete flight (climb, cruise,
# descent) for each cruise range, the mass falling with the fuel burned. Take-off at 85% of
# MTOW, assumed: a full flight, neither empty nor at the limit.
TAKEOFF_MASS = 0.85
CRUISE_KM = [100, 300, 600, 1000, 1500, 2500, 4000, 6000, 8000, 10000, 12000]
# Jet A at 0.80 kg per litre, 158.987 litres per barrel
KG_PER_BARREL = 127.2
# Data OpenSky lost for good, UTC, from its Trino documentation: a day that overlaps one is
# missing, not zero
OPENSKY_OUTAGES = [
    ("2023-01-02 23:00", "2023-01-03 10:00"), ("2023-01-18 11:00", "2023-01-23 07:00"),
    ("2023-06-21 13:00", "2023-06-21 22:00"), ("2023-11-15 06:00", "2023-11-16 08:00"),
    ("2023-11-20 01:00", "2023-11-20 03:00"), ("2023-12-02 08:00", "2023-12-05 03:00"),
    ("2024-05-20 10:00", "2024-05-21 05:00"), ("2026-01-02 19:00", "2026-01-03 18:00"),
    ("2026-01-24 08:00", "2026-01-24 17:00")]
# and the days it degraded without saying so, read from the coverage of the day itself: fewer
# than half the departures of the four weeks before, or more than 40% of its flights left
# without an airport (22% at the median, 39% at the 99th percentile, 2023-2026)
LOST_FLIGHTS = 0.5
LOST_UNKNOWN = 0.4
# OpenSky reprocessed its data from this date (decoder fixes): the airports it estimates may shift
OPENSKY_REPROCESSED = "2026-08-13"
# a month's index is a 28-day mean of the daily burn; it needs three weeks of days
BURN_MIN_DAYS = 21

# --- the naphtha feedstock -------------------------------------------------------
# Japanese customs, 9-digit statistical codes: petroleum spirits for petrochemical manufacture
# (the naphtha crackers burn), propane, and butanes for ethylene and other uses. The codes moved
# twice: HS 2012 renumbered 2710.11 as 2710.12, and propane's two lines became one in 2016.
CUSTOMS = {"271011181": "naphtha", "271012181": "naphtha",   # to 2011, from 2012
           "271112010": "propane", "271112020": "propane",   # to 2015
           "271112000": "propane",                           # from 2016
           "271113010": "butane", "271113020": "butane"}
# The Gulf, in the customs country codes: Iran, Iraq, Bahrain, Saudi Arabia, Kuwait, Qatar, Oman,
# the United Arab Emirates
GULF = [133, 134, 135, 137, 138, 140, 141, 147]
# Korea's cuts to naphtha cracking capacity, 2.7 to 3.7 Mt a year in all: dates marked on the
# charts, nothing the model can see
KOREA_CUTS = {"2026-02-01": "Daesan restructuring approved", "2026-07-01": "Yeosu approved"}
