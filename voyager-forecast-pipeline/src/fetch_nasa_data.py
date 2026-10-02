"""
fetch_nasa_data.py
===================
Downloads real Voyager merged hourly telemetry directly from NASA's public
Space Physics Data Facility.

This CANNOT run inside the Claude sandbox this pipeline was built in (that
sandbox's network policy blocks spdf.gsfc.nasa.gov outright). It WILL work
from an ordinary machine, and from a GitHub Actions runner -- GitHub-hosted
runners have normal outbound internet access, which is exactly why this
script is wired into .github/workflows/update-forecast.yml: the workflow
runs on GitHub's infrastructure, not inside any sandbox, so the fetch that's
blocked here goes through fine there.

Source: https://spdf.gsfc.nasa.gov/pub/data/voyager/{voyager1,voyager2}/merged/
File naming: vy1_YYYY.asc / vy2_YYYY.asc, one file per year.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/voyager/{sat_dir}/merged/{prefix}_{year}.asc"

SAT_DIRS = {"vy1": "voyager1", "vy2": "voyager2"}

# A real browser UA, defensively -- NOAA's SWPC subdomain turned out to 404
# non-browser clients on *some* of its endpoints (see fetch_live_data.py),
# so the same trick costs nothing here even though spdf.gsfc.nasa.gov has
# not been observed doing that.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Delay between successive year requests within one run. Fetching ~48 years
# sequentially with zero gap looks exactly like scraping to a server-side
# rate limiter, and a run that silently loses most years to connection
# resets/timeouts (NOT real 404s) is indistinguishable downstream from one
# where NASA genuinely has no data -- both just produce an empty frame for
# that year. The original notebook never hit this because it read files
# that had already been downloaded once into Google Drive; this script is
# the first thing in this pipeline to actually do 48 back-to-back live HTTP
# fetches against spdf.gsfc.nasa.gov in a single run.
REQUEST_DELAY_SECONDS = 0.5


def fetch_year(satellite: str, year: int, out_dir: str, retries: int = 4, timeout: int = 45) -> tuple[str | None, str]:
    """Returns (path_or_None, status) where status is one of:
    "fetched", "cached", "no_data" (a real 404 -- year genuinely doesn't
    exist yet), "failed" (retries exhausted on something other than a
    clean 404 -- network issue, rate limiting, 5xx, timeout). Distinguishing
    "no_data" from "failed" matters: silently treating a rate-limit/timeout
    the same as a genuine "this year isn't published" 404 is exactly how a
    bad run can quietly end up training/forecasting on far less real data
    than requested without anyone noticing.
    """
    import requests  # imported lazily so this file can be inspected without the dependency installed

    sat_dir = SAT_DIRS[satellite]
    url = BASE_URL.format(sat_dir=sat_dir, prefix=satellite, year=year)
    out_path = os.path.join(out_dir, f"{satellite}_{year}.asc")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  {satellite}_{year}.asc already present, skipping")
        return out_path, "cached"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            if resp.status_code == 404:
                print(f"  {satellite}_{year}.asc -> 404 (no data for this year yet), skipping")
                return None, "no_data"
            resp.raise_for_status()
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f"  {satellite}_{year}.asc -> {len(resp.content)/1024:.0f} KB")
            return out_path, "fetched"
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed for {year}: {e}")
            time.sleep(2 * attempt)
    print(f"  giving up on {satellite}_{year}.asc after {retries} attempts (network/rate-limit failure, NOT a confirmed 404)")
    return None, "failed"


def fetch_range(satellite: str, start_year: int, end_year: int, out_dir: str) -> list[str]:
    fetched = []
    no_data_years: list[int] = []
    failed_years: list[int] = []
    years = list(range(start_year, end_year + 1))
    for i, year in enumerate(years):
        path, status = fetch_year(satellite, year, out_dir)
        if path:
            fetched.append(path)
        elif status == "no_data":
            no_data_years.append(year)
        elif status == "failed":
            failed_years.append(year)
        # Only pace the *next* request if we actually hit the network just
        # now (skip the delay after a cache hit, and after the last year).
        if status != "cached" and i < len(years) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    print(
        f"  fetch_range({satellite}, {start_year}-{end_year}): "
        f"{len(fetched)} fetched, {len(no_data_years)} genuinely absent (404), "
        f"{len(failed_years)} failed after retries (network/rate-limit, NOT 404)"
    )
    if failed_years:
        print(
            f"  WARNING: {len(failed_years)} year(s) failed for reasons other than a "
            f"real 404 and were silently excluded from the merge: {failed_years}. "
            f"This is very likely NASA/network throttling under rapid sequential "
            f"requests, not an absence of real data for those years -- re-running, "
            f"or increasing REQUEST_DELAY_SECONDS/retries, may recover them."
        )
    return fetched


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--satellite", choices=["vy1", "vy2"], default="vy2")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--out-dir", default="data/raw")
    args = p.parse_args()

    print(f"Fetching {args.satellite} merged data, {args.start_year}-{args.end_year} -> {args.out_dir}")
    results = fetch_range(args.satellite, args.start_year, args.end_year, args.out_dir)
    print(f"Fetched {len(results)} file(s).")
    if not results:
        print("No files fetched -- check network access to spdf.gsfc.nasa.gov from this environment.")
        sys.exit(1)
