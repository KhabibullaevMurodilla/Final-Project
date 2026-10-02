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


def fetch_year(satellite: str, year: int, out_dir: str, retries: int = 3, timeout: int = 30) -> str | None:
    import requests  # imported lazily so this file can be inspected without the dependency installed

    sat_dir = SAT_DIRS[satellite]
    url = BASE_URL.format(sat_dir=sat_dir, prefix=satellite, year=year)
    out_path = os.path.join(out_dir, f"{satellite}_{year}.asc")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  {satellite}_{year}.asc already present, skipping")
        return out_path

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                print(f"  {satellite}_{year}.asc -> 404 (no data for this year yet), skipping")
                return None
            resp.raise_for_status()
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f"  {satellite}_{year}.asc -> {len(resp.content)/1024:.0f} KB")
            return out_path
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed for {year}: {e}")
            time.sleep(2 * attempt)
    print(f"  giving up on {satellite}_{year}.asc")
    return None


def fetch_range(satellite: str, start_year: int, end_year: int, out_dir: str) -> list[str]:
    fetched = []
    for year in range(start_year, end_year + 1):
        path = fetch_year(satellite, year, out_dir)
        if path:
            fetched.append(path)
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
