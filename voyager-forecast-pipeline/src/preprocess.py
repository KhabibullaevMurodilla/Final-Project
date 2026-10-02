"""
preprocess.py
=============
Faithful, modular re-implementation of the data-loading and cleaning logic
from the original Final_Project.ipynb notebook (KhabibullaevMurodilla/Final-Project).

Source data format: NASA Voyager 1/2 "merged" hourly telemetry files (.asc),
whitespace-delimited, one row per hour. See:
https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/merged/00readme_v1.txt
https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/

Real column layout (34 columns in the raw file):
  1-3   : Year, Day-of-year, Hour
  4-6   : Heliographic distance (AU), latitude, longitude  <- "position" columns
  7-11  : Interplanetary magnetic field: |B| avg, avg-vector, BR, BT, BN
  12-16 : Solar wind: proton flow speed, elevation angle, azimuth angle,
          proton density, proton temperature
  17-34 : 18 proton-flux energy channels (LECP + CRS instruments,
          <2 MeV up to >300 MeV)

Missing/fill values in the raw files are encoded as repeating-9 patterns
(e.g. 999.999, 9999.9, 9.999e+05) -- this mirrors the notebook's `is_missing`
regex-based detection exactly.
"""

from __future__ import annotations

import re
import glob
import os
from typing import List

import numpy as np
import pandas as pd

# --- Exact column schema from the notebook -------------------------------

POSITION_COLUMNS: List[str] = ["HGI_R", "HGI_Lat", "HGI_Lon"]

FIELD_AND_WIND_COLUMNS: List[str] = [
    "B_Avg", "B_Avg_Vec", "BR", "BT", "BN",
    "V_Flow", "V_Theta", "V_Phi", "Proton_Density", "Proton_Temp",
]

FLUX_COLUMNS: List[str] = [f"Flux_{i}" for i in range(1, 19)]

COLUMN_NAMES: List[str] = (
    ["Year", "Day", "Hour"] + POSITION_COLUMNS + FIELD_AND_WIND_COLUMNS + FLUX_COLUMNS
)

OTHER_COLUMNS: List[str] = [
    c for c in COLUMN_NAMES if c not in POSITION_COLUMNS + ["Year", "Day", "Hour"]
]

ALL_COLUMNS: List[str] = POSITION_COLUMNS + OTHER_COLUMNS  # 31 columns total

SEQ_LENGTH = 120  # hours of history used per prediction step (matches notebook)


def is_missing(value) -> bool:
    """Detect NASA's repeating-9 fill-value convention (e.g. 999.999, 9.999e+05)."""
    value_str = str(value).strip()
    if re.match(r"^-?9+\.?9*$", value_str):
        return True
    if re.match(r"^-?9+0*\.?0*$", value_str):
        return True
    return False


def load_asc_file(path: str) -> pd.DataFrame:
    """Load one raw .asc file with the real 34-column schema."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES, index_col=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Build DateTime index, drop raw time columns, null out fill values, drop NaNs."""
    df = df.copy()
    df["DateTime"] = (
        pd.to_datetime(
            df["Year"].astype(str) + " " + df["Day"].astype(str) + " " + df["Hour"].astype(str),
            format="%Y %j %H",
        )
        - pd.Timedelta(hours=1)
    )
    df = df.drop(["Year", "Day", "Hour"], axis=1)
    df = df.set_index("DateTime")
    df = df.map(lambda x: np.nan if is_missing(x) else x)
    df = df.dropna()
    return df


def load_data(file_prefix: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load and clean a contiguous range of yearly .asc files, e.g.:
        load_data('data/raw/vy2_', 1977, 2015)
    expects files data/raw/vy2_1977.asc ... data/raw/vy2_2014.asc
    """
    frames = []
    for year in range(start_year, end_year):
        path = f"{file_prefix}{year}.asc"
        if not os.path.exists(path):
            continue
        frames.append(load_asc_file(path))
    if not frames:
        raise FileNotFoundError(
            f"No .asc files found matching {file_prefix}{{{start_year}..{end_year-1}}}.asc"
        )
    df = pd.concat(frames)
    return clean(df)


def load_data_dir(directory: str, satellite_prefix: str = "vy2_") -> pd.DataFrame:
    """Load every matching .asc file found in a directory, in year order."""
    pattern = os.path.join(directory, f"{satellite_prefix}*.asc")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern}")
    frames = [load_asc_file(p) for p in paths]
    df = pd.concat(frames)
    return clean(df)


if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = load_data_dir(directory)
    print(df.shape)
    print(df.head())
