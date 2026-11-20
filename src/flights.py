"""The kerosene burn index: what the commercial jets leaving US airports burn, flight by flight,
from the ADS-B record of the OpenSky Network.

    python -m src.flights   ->  data/US_jet_fuel/fuel_table.csv: fuel against distance, for every
                                OpenAP model the commercial types map to

Each flight OpenSky closes on a day comes with its estimated departure and arrival airports and
the aircraft's transponder address, icao24. The type is joined on icao24 from OpenSky's aircraft
database, and only the commercial jets of config.COMMERCIAL_JETS are kept. For each flight:

    distance = great circle between the two airports
    fuel     = the type's fuel against distance, from a table OpenAP computes once at a typical
               take-off mass

Summed by day over the flights that leave a US airport, and turned into kb/d. A flight that only
arrives in the US is counted apart: its fuel was mostly bought abroad. A flight OpenSky could
not place, no departure airport estimated or a US departure with no arrival, has no distance: it
is counted, not burned, which measures coverage.

The level is not EIA's: ADS-B misses flights, a great circle is shorter than the route flown,
and product supplied also counts military and general aviation. What the index needs is a stable
ratio to EIA, checked in notebook 01_data_jet.
"""

import numpy as np
import pandas as pd
from openap import FuelFlow, aero, nav, prop
from openap.gen import FlightGenerator

from src import config as c

FUEL_TABLE = c.JET / "fuel_table.csv"
STEP_SECONDS = 30  # OpenAP's flight generator samples a complete flight every 30 seconds
# OpenAP's airports, read once: nav.airport() reads the whole file again at every call
AIRPORTS = pd.read_csv(nav.db_airport).drop_duplicates("icao").set_index("icao")[["lat", "lon"]]


def fuel_curve(model):
    """Fuel burned, kg, against distance flown, km, for one OpenAP model: a complete flight for
    each cruise range, integrated with the mass falling as the fuel is burned."""
    flight = FlightGenerator(model, use_synonym=True)
    flow = FuelFlow(model, use_synonym=True)
    takeoff = c.TAKEOFF_MASS * prop.aircraft(model, use_synonym=True)["mtow"]
    rows = []
    for km in c.CRUISE_KM:
        path = flight.complete(dt=STEP_SECONDS, random=False, range_cr=km * 1000)
        path = path[path["v"] > 0]  # the first sample is the aircraft at rest
        mass, fuel = takeoff, 0.0
        for v, h, vs in zip(path["v"], path["h"], path["vs"]):
            # OpenAP has no flow for the last sample, the aircraft rolling to a stop: zero
            kg = np.nan_to_num(flow.enroute(mass=mass, tas=v / aero.kts, alt=h / aero.ft,
                                            vs=vs / aero.fpm)) * STEP_SECONDS
            fuel, mass = fuel + kg, mass - kg
        rows.append({"model": model, "km": path["s"].iloc[-1] / 1000, "fuel_kg": fuel})
    return pd.DataFrame(rows)


def distance(departure, arrival):
    """Great-circle distance between two lists of ICAO airports, km; missing if either is
    unknown."""
    lat1, lon1 = np.radians(AIRPORTS.reindex(departure.astype(object)).to_numpy()).T
    lat2, lon2 = np.radians(AIRPORTS.reindex(arrival.astype(object)).to_numpy()).T
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))  # mean Earth radius, km


def query(day):
    """The flights OpenSky closed on one UTC day that touch a US airport. day is the partition:
    a Unix timestamp at 00:00 UTC, which every query must filter on."""
    us = " OR ".join(f"{side} LIKE '{prefix}%'" for side in ("estdepartureairport",
                                                             "estarrivalairport")
                     for prefix in c.US_AIRPORTS)
    return (f"SELECT icao24, estdepartureairport, estarrivalairport FROM flights_data4 "
            f"WHERE day = {int(pd.Timestamp(day, tz='UTC').timestamp())} AND ({us})")


def burn(flights, aircraft, table):
    """One day's flights by commercial type: departures from a US airport with a known arrival
    and the fuel they burn, kg; arrivals from abroad; flights with an airport not estimated."""
    df = flights.merge(aircraft[["icao24", "typecode"]], on="icao24")
    df = df[df["typecode"].isin(list(c.COMMERCIAL_JETS))].copy()
    us = tuple(c.US_AIRPORTS)
    leaves = df["estdepartureairport"].fillna("").str.startswith(us)
    df["km"] = distance(df["estdepartureairport"], df["estarrivalairport"])
    df["fuel_kg"] = np.nan
    for model, curve in table.groupby("model"):
        rows = df["typecode"].map(c.COMMERCIAL_JETS) == model
        # a flight shorter than the shortest complete one is read on a line from zero
        df.loc[rows, "fuel_kg"] = np.interp(df.loc[rows, "km"], [0, *curve["km"]],
                                            [0, *curve["fuel_kg"]])
    df["departures"] = leaves & df["km"].notna()
    df["arrivals_only"] = ~leaves & df["estdepartureairport"].notna()
    # no departure estimated, or a US departure with no arrival, or one OpenAP cannot place
    df["unknown_airport"] = df["estdepartureairport"].isna() | (leaves & df["km"].isna())
    df["fuel_kg"] = df["fuel_kg"].where(df["departures"], 0.0)
    return df.groupby("typecode")[["departures", "arrivals_only", "unknown_airport",
                                   "fuel_kg"]].sum().reset_index()


if __name__ == "__main__":
    table = pd.concat([fuel_curve(model) for model in sorted(set(c.COMMERCIAL_JETS.values()))])
    table.to_csv(FUEL_TABLE, index=False)
    print(table.pivot(index="model", columns="km", values="fuel_kg").round(-2).iloc[:, :4])
