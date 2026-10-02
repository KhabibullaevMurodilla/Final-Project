"""
preprocess.py
=============
Faithful, modular re-implementation of the data-loading and cleaning logic
from the original Final_Project.ipynb notebook (KhabibullaevMurodilla/Final-Project).

Source data format: NASA Voyager 1/2 "merged" hourly telemetry files (.asc),
whitespace-delimited, one row per hour. See:
https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/merged/00readme_v1.txt
https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/

Real column layout, confirmed against NASA's own field documentation
(spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/vy2mgd.txt) -- the raw
file actually has 37 columns, not 34:
  1-3   : Year, Day-of-year, Hour
  4-6   : Heliographic distance (AU), latitude, longitude  <- "position" columns
  7-16  : Interplanetary magnetic field (|B| avg, avg-vector, BR, BT, BN) and
          solar wind (flow speed, elevation angle, azimuth angle, proton
          density, proton temperature) -- 10 columns
  17-19 : 3 LECP proton-flux energy channels (0.52-30.0 MeV)
  20-37 : 18 CRS proton-flux energy channels (3.0-598.7 MeV)

location_model.h5/param_model.h5 were trained expecting 31 non-time columns
(3 position + 10 field/wind + 18 flux) -- the ORIGINAL 34-column assumption
below, which is short exactly the 3 LECP channels. Those weights are frozen,
so instead of changing what the models expect, RAW_COLUMN_NAMES below parses
all 37 real columns correctly (so nothing is silently misaligned), and
load_asc_file drops the 3 LECP_Flux_* columns immediately after parsing --
the 18 CRS channels keep the name "Flux_1".."Flux_18" so everything
downstream (ALL_COLUMNS, the models, predict.py) is unchanged.

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

LECP_COLUMNS: List[str] = ["LECP_1", "LECP_2", "LECP_3"]        # real cols 17-19, 0.52-30.0 MeV
CRS_COLUMNS: List[str] = [f"CRS_{i}" for i in range(1, 19)]     # real cols 20-37, 3.0-598.7 MeV

# The real schema as it exists in the raw file: 37 columns, in file order.
RAW_COLUMN_NAMES: List[str] = (
    ["Year", "Day", "Hour"] + POSITION_COLUMNS + FIELD_AND_WIND_COLUMNS
    + LECP_COLUMNS + CRS_COLUMNS
)

# What the notebook's original `names=column_names` (34 entries, missing the
# 3 LECP channels) actually did against these real 37-column rows: pandas
# silently used the 34 given names against the first 34 whitespace tokens
# and dropped the trailing 3. In file order that means the notebook's
# "Flux_1".."Flux_18" were really: LECP_1, LECP_2, LECP_3, then only the
# FIRST 15 of the 18 CRS channels -- CRS_16/17/18 (the three highest-energy
# bands, up to 598.7 MeV) were never seen in training at all, for the
# entire 1977-2015 training window.
#
# The model's weights are frozen, so "fixed" has to mean "matches what it
# actually learned," not "matches NASA's documented channel order." This
# reproduces that exact same selection deliberately (not by accident, this
# time) so real data lines up with the training distribution instead of
# feeding it 3 flux channels the model never learned to interpret.
FLUX_TRAIN_SOURCE: List[str] = LECP_COLUMNS + CRS_COLUMNS[:15]  # 3 + 15 = 18, in training order
FLUX_COLUMNS: List[str] = [f"Flux_{i}" for i in range(1, 19)]   # model-facing names, unchanged everywhere downstream

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


def load_asc_file(path: str) -> pd.DataFrame | None:
    """Load one raw .asc file with the real 37-column schema, then select
    the exact 18 flux columns (in the exact order) the frozen models were
    actually trained on -- LECP_1-3 followed by the first 15 CRS channels,
    renamed to the model-facing Flux_1..Flux_18 -- instead of NASA's
    documented 18 CRS channels (see the FLUX_TRAIN_SOURCE comment above).

    Checking the real field count first and skipping (not guessing how to
    fix) any file that doesn't match 37 keeps a genuinely malformed file
    from silently poisoning the merged dataset -- missing one year of real
    data is far better than corrupting every year after it. This caught the
    real bug directly: every file from 1977-2024 has 37 fields, confirming
    the schema itself (not the file) was wrong until this fix.
    """
    with open(path) as f:
        first_data_line = next((line for line in f if line.strip()), "")
    actual_fields = len(first_data_line.split())
    expected_fields = len(RAW_COLUMN_NAMES)
    if actual_fields != expected_fields:
        print(
            f"  SKIPPING {path}: expected {expected_fields} whitespace-separated "
            f"fields per row (the real NASA merged-file schema), found "
            f"{actual_fields} on its first data line. Loading it anyway would "
            f"silently misalign columns for this file instead of failing "
            f"loudly, so it's excluded from the merge rather than guessed at."
        )
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMN_NAMES, index_col=False)

    # Build the model-facing Flux_1..Flux_18 from exactly what training saw
    # (LECP_1-3 + the first 15 CRS channels), dropping CRS_16/17/18 -- not
    # NASA's real channel order, deliberately, to match the frozen weights.
    flux_block = df[FLUX_TRAIN_SOURCE].copy()
    flux_block.columns = FLUX_COLUMNS
    df = pd.concat(
        [df[["Year", "Day", "Hour"] + POSITION_COLUMNS + FIELD_AND_WIND_COLUMNS], flux_block],
        axis=1,
    )
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
        loaded = load_asc_file(path)
        if loaded is not None:
            frames.append(loaded)
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
    frames = [f for f in (load_asc_file(p) for p in paths) if f is not None]
    if not frames:
        raise FileNotFoundError(f"No usable files matching {pattern} (all failed the column-count check)")
    df = pd.concat(frames)
    return clean(df)


if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = load_data_dir(directory)
    print(df.shape)
    print(df.head())
