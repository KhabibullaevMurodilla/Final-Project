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
   (https://services.swpc.noaa.gov/products/solar-wind/)
   Live plasma and magnetic-field readings from the DSCOVR satellite at the
   Sun-Earth L1 point, updated roughly every minute. This is a genuinely
   different vantage point from Voyager (near Earth, inside the heliosphere,
   vs. Voyager far outside it) -- useful as a "conditions near Earth right
   now" comparison panel, not a Voyager substitute.

Output: a small live.json consumed by the web page's "right now" panel.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"

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
    # NOAA SWPC's services subdomain returns 404 to requests that don't look
    # like a normal browser (confirmed: the directory listing shows these
    # files exist, but a plain python-requests/urllib fetch of the file
    # itself 404s). A browser User-Agent clears it.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def fetch_solar_wind_now() -> dict | None:
    import requests

    try:
        plasma = requests.get(SWPC_PLASMA_URL, headers=SWPC_HEADERS, timeout=20)
        plasma.raise_for_status()
        plasma = plasma.json()
        mag = requests.get(SWPC_MAG_URL, headers=SWPC_HEADERS, timeout=20)
        mag.raise_for_status()
        mag = mag.json()

        p_header, p_rows = plasma[0], plasma[1:]
        m_header, m_rows = mag[0], mag[1:]
        p_last = [r for r in reversed(p_rows) if all(v not in (None, "") for v in r)][0]
        m_last = [r for r in reversed(m_rows) if all(v not in (None, "") for v in r)][0]

        p = dict(zip(p_header, p_last))
        m = dict(zip(m_header, m_last))

        def safe_float(d: dict, key: str):
            try:
                return float(d.get(key))
            except (TypeError, ValueError):
                return None

        return {
            "speed_km_s": safe_float(p, "speed"),
            "density_p_cm3": safe_float(p, "density"),
            "temperature_k": safe_float(p, "temperature"),
            "bt_nT": safe_float(m, "bt"),
            "time_tag": p.get("time_tag"),
            "source": "NOAA SWPC real-time solar wind (DSCOVR, L1)",
            "_raw_header_check": {"plasma_fields": p_header, "mag_fields": m_header},
        }
    except Exception as e:  # noqa: BLE001
        print(f"  SWPC fetch failed: {e}")
        # Surface the failure in the output itself, not just the Actions
        # log -- the job exits 0 either way, so this is the only place
        # future failures (NOAA changing the endpoint again, a timeout,
        # etc.) will be visible from the deployed page/README.
        return {"error": str(e), "source": "NOAA SWPC real-time solar wind (DSCOVR, L1)"}


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
