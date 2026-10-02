"""
run_pipeline.py
================
End-to-end driver: fetch real NASA data -> clean -> forecast -> write the
JSON the web page reads. This is what .github/workflows/update-forecast.yml
runs on GitHub's own infrastructure (real internet access, unlike the
sandbox this pipeline was built in).

Usage:
    python src/run_pipeline.py \
        --satellite vy2 --start-year 1977 --end-year 2025 \
        --future-hours 240 --out web/forecast.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fetch_nasa_data import fetch_range
from preprocess import load_data_dir
from predict import run as run_predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--satellite", choices=["vy1", "vy2"], default="vy2")
    # The original notebook trained location_model.h5/param_model.h5 with a
    # MinMaxScaler fit on 1977-2015 data (cell 18/25: df_train = load_data(
    # ..., 1977, 2015); scaler.fit_transform(df_train[all_columns])). This
    # pipeline refits a fresh scaler every run on whatever range it fetches
    # -- if that range doesn't overlap 1977-2015, the refit scaler's min/max
    # has nothing to do with the scaling the frozen model weights actually
    # expect, and feeding mis-scaled inputs into a 240-step autoregressive
    # rollout compounds the mismatch until it diverges. Defaulting to 1977
    # keeps the refit scaler close to the one the models were really trained
    # under, instead of an arbitrary disjoint window.
    p.add_argument("--start-year", type=int, default=1977)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--csv-out", default="data/real_merged.csv")
    p.add_argument("--loc-model", default="models/location_model.h5")
    p.add_argument("--param-model", default="models/param_model.h5")
    p.add_argument("--future-hours", type=int, default=240)
    p.add_argument("--backtest-hours", type=int, default=72)
    p.add_argument("--out", default="web/forecast.json")
    p.add_argument("--allow-demo-fallback", action="store_true",
                    help="If the real NASA fetch fails (e.g. running locally in a "
                         "sandboxed environment), fall back to generated demo data "
                         "instead of failing the run.")
    args = p.parse_args()

    print(f"Step 1/3: fetching real {args.satellite} data, {args.start_year}-{args.end_year} ...")
    fetched = fetch_range(args.satellite, args.start_year, args.end_year, args.raw_dir)

    if not fetched:
        if args.allow_demo_fallback:
            print("No real files fetched. Falling back to calibrated demo data.")
            from generate_demo_data import generate_demo_dataframe
            df = generate_demo_dataframe()
            os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
            df.to_csv(args.csv_out)
        else:
            print("No real files fetched and --allow-demo-fallback not set. Aborting.")
            sys.exit(1)
    else:
        print("Step 2/3: cleaning real data ...")
        df = load_data_dir(args.raw_dir, satellite_prefix=f"{args.satellite}_")
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        df.to_csv(args.csv_out)
        print(f"  {df.shape[0]} clean hourly rows -> {args.csv_out}")

    print("Step 3/3: forecasting with the real trained models ...")
    run_predict(args.csv_out, args.loc_model, args.param_model, args.future_hours, args.out, args.backtest_hours)
    print("Done.")


if __name__ == "__main__":
    main()
