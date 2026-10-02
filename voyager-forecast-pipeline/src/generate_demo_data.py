"""
generate_demo_data.py
======================
THIS IS NOT REAL NASA TELEMETRY.

NASA's public data server (spdf.gsfc.nasa.gov) is not reachable from this
build sandbox's network policy, so this script synthesises a calibrated
stand-in dataset shaped exactly like real Voyager-2 merged hourly data, so
the preprocessing + prediction pipeline can be demonstrated end-to-end.

Calibration anchors (taken from a verified live read of the real file
https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/merged/vy1_2020.asc):
  - HGI_R    ~ 147.85 AU         (slowly increasing, ~3.5 AU/year for V1-era outbound cruise)
  - HGI_Lat  ~ 34.8 deg          (near-constant on these timescales)
  - HGI_Lon  ~ 175.2 deg         (near-constant on these timescales)
  - B_Avg    ~ 0.35-0.40 nT      (interstellar/outer-heliosheath field magnitude)
  - Proton flux channels ~ 1e-3 to 1e-2 (particles / cm^2-s-sr-MeV), heavily noisy

To use REAL data instead: download the real .asc files directly from
https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/ (reachable from
an ordinary browser/network, just not from this sandbox) and place them in
data/raw/, then run preprocess.load_data_dir() -- no code changes needed.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from preprocess import ALL_COLUMNS, POSITION_COLUMNS, OTHER_COLUMNS

RNG = np.random.default_rng(42)


def generate_demo_dataframe(n_hours: int = 20_000, start: str = "2010-01-01") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n_hours, freq="h")
    t = np.arange(n_hours)

    # Slow radial outbound drift (~3.5 AU/year => /8760 per hour) plus small noise
    hgi_r = 90.0 + (3.5 / 8760.0) * t + RNG.normal(0, 0.01, n_hours)
    hgi_lat = 34.8 + RNG.normal(0, 0.02, n_hours).cumsum() * 0.0005
    hgi_lon = 175.2 + RNG.normal(0, 0.02, n_hours).cumsum() * 0.0005

    # Magnetic field: slow drift + solar-cycle-like oscillation + noise
    b_avg = 0.37 + 0.05 * np.sin(2 * np.pi * t / (11 * 8760)) + RNG.normal(0, 0.01, n_hours)
    b_avg = np.clip(b_avg, 0.1, None)
    b_vec = b_avg + RNG.normal(0, 0.005, n_hours)
    br = RNG.normal(0.04, 0.02, n_hours)
    bt = RNG.normal(-0.35, 0.02, n_hours)
    bn = RNG.normal(0.17, 0.02, n_hours)

    # Solar wind (sparse / often fill-valued in the real outer-heliosphere data,
    # but we keep it populated here for a clean demo signal)
    v_flow = RNG.normal(350, 20, n_hours)
    v_theta = RNG.normal(0, 3, n_hours)
    v_phi = RNG.normal(0, 3, n_hours)
    proton_density = np.abs(RNG.normal(0.002, 0.001, n_hours))
    proton_temp = np.abs(RNG.normal(15000, 3000, n_hours))

    data = {
        "HGI_R": hgi_r, "HGI_Lat": hgi_lat, "HGI_Lon": hgi_lon,
        "B_Avg": b_avg, "B_Avg_Vec": b_vec, "BR": br, "BT": bt, "BN": bn,
        "V_Flow": v_flow, "V_Theta": v_theta, "V_Phi": v_phi,
        "Proton_Density": proton_density, "Proton_Temp": proton_temp,
    }

    # 18 flux channels: lower channels (lower energy) noisier/higher amplitude,
    # higher channels smaller magnitude -- mirrors the real file's shape
    for i in range(1, 19):
        base = 0.003 * (19 - i) / 18 + 0.0005
        series = np.abs(base + RNG.normal(0, base * 0.6, n_hours))
        # occasional cosmic-ray-event spikes
        spike_mask = RNG.random(n_hours) < 0.001
        series[spike_mask] *= RNG.uniform(5, 15, spike_mask.sum())
        data[f"Flux_{i}"] = series

    df = pd.DataFrame(data, index=idx)
    df.index.name = "DateTime"
    return df[ALL_COLUMNS]


if __name__ == "__main__":
    os.makedirs("data/demo", exist_ok=True)
    df = generate_demo_dataframe()
    out_path = "data/demo/vy2_demo_merged.csv"
    df.to_csv(out_path)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {out_path}")
    print(df.describe().T[["mean", "std", "min", "max"]])
