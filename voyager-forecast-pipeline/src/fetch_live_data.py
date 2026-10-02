"""
fetch_live_data.py
===================
Pulls genuinely LIVE, real-time data from two public APIs (both confirmed
reachable and returning real values when queried from an unrestricted network
-- e.g. a GitHub Actions runner; NOT reachable from the Claude sandbox this
pipeline was assembled in, same restriction as fetch_nasa_data.py):

1. JPL Horizons System (https://ssd.jpl.nasa.gov/api/horizons.api)
   Real-time ephemeris for Voyager 1 (NAIF ID -31) and Voyager 2 (-32):
   current heliocentric distance and radial velocity, computed from JPL's
   own navigation solution for each spacecraft -- this is the real number,
   not the merged historical file, and it is current as of whenever this
   script runs.

2. NOAA Space Weather Prediction Center real-time solar wind
   (https://services.swpc.noaa.gov/json/rtsw/)
   Live plasma and magnetic-field readings from whichever spacecraft is
   currently NOAA's primary real-time-solar-wind (RTSW) source at the
   Sun-Earth L1 point (DSCOVR, ACE, or newer additions like IMAP/SOLAR1 --
   NOAA switches the active feed between them), updated roughly every
   minute. This is a genuinely different vantage point from Voyager (near
   Earth, inside the heliosphere, vs. Voyager far outside it) -- useful as
   a "conditions near Earth right now" comparison panel, not a Voyager
   substitute.

   NOAA retired the old `products/solar-wind/plasma-2-hour.json` /
   `mag-2-hour.json` endpoints in 2026 -- they now 404 unconditionally, from
   ANY client (confirmed directly: both a from-the-sandbox fetch and a
   separate run from GitHub Actions hit the identical 404 on the identical
   URL). This was NOT a User-Agent/bot-blocking problem despite the header
   workaround that used to live in this file -- the endpoint itself is
   gone. The real replacement is the RTSW family
   (services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json and rtsw_mag_1m.json,
   see https://github.com/qso-graph/solar-mcp/pull/5 for independent
   confirmation of the same migration), which returns a JSON array of
   per-instrument-source records (newest first, each tagged "active": true
   for whichever source NOAA currently trusts) rather than the old
   header-row + data-rows array, and uses the sentinel -9999 for missing
   numeric fields instead of null/empty-string.

Output: a small live.json consumed by the web page's "right now" panel.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
# 2026 RTSW endpoints -- see the module docstring for why these replaced
# the old products/solar-wind/*-2-hour.json URLs (those now 404 for everyone).
SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
SWPC_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
NOAA_MISSING_SENTINEL = -9999

KM_PER_AU = 1.495978707e8

VOYAGER_NAIF_IDS = {"voyager1": "-31", "voyager2": "-32"}


def fetch_voyager_position(which: str) -> dict | None:
    import requests

    naif_id = VOYAGER_NAIF_IDS[which]
    today = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=1)

    params = {
        "format": "text",
        "COMMAND": f"'{naif_id}'",
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",  # Sun body center
        "START_TIME": f"'{today.isoformat()}'",
        "STOP_TIME": f"'{tomorrow.isoformat()}'",
        "STEP_SIZE": "'1 d'",
    }
    try:
        resp = requests.get(HORIZONS_URL, params=params, timeout=30)
        resp.raise_for_status()
        text = resp.text
        soe = text.split("$$SOE")[1].split("$$EOE")[0]
        lines = [l for l in soe.strip().splitlines() if l.strip()]
        # lines[0] = date line, lines[1] = X Y Z, lines[2] = VX VY VZ, lines[3] = LT RG RR
        vec_line = lines[-1]
        rg_km = float(vec_line.split("RG=")[1].split("RR=")[0].strip())
        rr_kms = float(vec_line.split("RR=")[1].strip())
        return {
            "distance_au": round(rg_km / KM_PER_AU, 3),
            "radial_velocity_km_s": round(rr_kms, 3),
            "as_of": datetime.now(timezone.utc).isoformat(),
            "source": "JPL Horizons System (live ephemeris)",
        }
    except Exception as e:  # noqa: BLE001
        print(f"  Horizons fetch failed for {which}: {e}")
        return None


SWPC_HEADERS = {
    # Kept defensively (costs nothing) even though the 2026 404s turned out
    # to be a retired endpoint, not a bot-blocking header check -- NOAA's
    # other products/* paths DID previously 404 non-browser clients, so a
    # real UA is cheap insurance against that happening here too.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _safe_float(value) -> float | None:
    """Parse a numeric field, treating NOAA's -9999 missing-data sentinel
    (and any other non-finite/unparseable value) the same as a real null."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f <= NOAA_MISSING_SENTINEL:
        return None
    return f


def _latest_usable_record(records: list[dict], primary_key: str) -> dict | None:
    """RTSW returns one record per *source* per timestamp (DSCOVR/ACE/IMAP/
    SOLAR1/...), newest first, each flagged "active": true for whichever
    source NOAA currently trusts as the real-time feed. Sort by time_tag
    defensively (don't assume the API's own ordering holds forever), then
    prefer the most recent record that is both flagged active and has a
    real (non-sentinel) value for `primary_key`; fall back to the most
    recent record with a real value at all if nothing is flagged active.
    """
    usable = [r for r in records if _safe_float(r.get(primary_key)) is not None]
    if not usable:
        return None
    usable.sort(key=lambda r: r.get("time_tag", ""), reverse=True)
    active = [r for r in usable if r.get("active")]
    return active[0] if active else usable[0]


def fetch_solar_wind_now() -> dict | None:
    import requests

    try:
        plasma = requests.get(SWPC_PLASMA_URL, headers=SWPC_HEADERS, timeout=20)
        plasma.raise_for_status()
        plasma = plasma.json()
        mag = requests.get(SWPC_MAG_URL, headers=SWPC_HEADERS, timeout=20)
        mag.raise_for_status()
        mag = mag.json()

        p = _latest_usable_record(plasma, "proton_speed")
        m = _latest_usable_record(mag, "bt")
        if p is None or m is None:
            raise ValueError(
                f"no usable record in RTSW response (plasma_records={len(plasma)}, "
                f"mag_records={len(mag)}) -- NOAA may have changed the schema again"
            )

        return {
            "speed_km_s": _safe_float(p.get("proton_speed")),
            "density_p_cm3": _safe_float(p.get("proton_density")),
            "temperature_k": _safe_float(p.get("proton_temperature")),
            "bt_nT": _safe_float(m.get("bt")),
            "time_tag": p.get("time_tag"),
            "source": f"NOAA SWPC real-time solar wind (RTSW, {p.get('source', 'L1')})",
        }
    except Exception as e:  # noqa: BLE001
        print(f"  SWPC fetch failed: {e}")
        # Surface the failure in the output itself, not just the Actions
        # log -- the job exits 0 either way, so this is the only place
        # future failures (NOAA changing the endpoint again, a timeout,
        # etc.) will be visible from the deployed page/README.
        return {"error": str(e), "source": "NOAA SWPC real-time solar wind (RTSW)"}


def main(out_path: str) -> None:
    result = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "voyager1": fetch_voyager_position("voyager1"),
        "voyager2": fetch_voyager_position("voyager2"),
        "solar_wind_near_earth": fetch_solar_wind_now(),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote live stats -> {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="web/live.json")
    args = p.parse_args()
    main(args.out)
