"""
predict.py
==========
Loads the two REAL trained models from the original project
(KhabibullaevMurodilla/Final-Project -> location_model.h5, param_model.h5)
and runs the same recursive multi-step forecasting procedure used in the
notebook:

  1. Fit a MinMaxScaler over the input window (same preprocessing the
     models were trained under).
  2. location_model: LSTM(64) -> Dense(3), takes the last 120 hours of
     [HGI_R, HGI_Lat, HGI_Lon] and predicts the next hour's position.
     Rolled forward autoregressively for `future_steps` hours.
  3. param_model: LSTM(128)->LSTM(64)->LSTM(32)->Dense(28), takes the last
     120 hours of all 31 columns and predicts the next hour's 28
     non-position parameters (field, solar wind, 18 flux channels).
     Rolled forward using the location model's own predictions as the
     position input for each step (exactly as in the notebook).
  4. Inverse-transform both forecasts back to physical units and export to
     JSON for the web page.

Run:
    python predict.py --input ../data/demo/vy2_demo_merged.csv \
                       --future-hours 720 \
                       --out ../outputs/forecast.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from preprocess import ALL_COLUMNS, POSITION_COLUMNS, OTHER_COLUMNS, SEQ_LENGTH


def create_sequences(data: np.ndarray, seq_length: int):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


def predict_future_locations(model, last_sequence: np.ndarray, steps: int) -> np.ndarray:
    """Autoregressive rollout of the location model (notebook logic, unchanged
    except for one addition -- see the clip below)."""
    future_predictions = []
    current_sequence = last_sequence.copy()
    n_pos = len(POSITION_COLUMNS)
    for _ in range(steps):
        pred = model.predict(current_sequence.reshape(1, SEQ_LENGTH, n_pos), verbose=0)
        # The model was trained against a MinMaxScaler fit on a different
        # (larger, differently-ranged) real-data window than whatever gets
        # fetched on a given run. Feeding its own output back in for 240
        # straight steps means any small scale mismatch compounds every
        # step -- confirmed: without this clip, HGI_R was already in the
        # tens of millions of AU by hour ~130 and climbing exponentially,
        # eventually overflowing to Infinity (which breaks JSON.parse on
        # the page entirely, not just that one chart). Clamping each
        # prediction back into the scaler's valid [0, 1] range before it
        # becomes next step's input is the standard fix for this failure
        # mode and keeps every inverse-transformed value inside the
        # physical range the scaler was actually fit on.
        pred = np.clip(pred, 0.0, 1.0)
        future_predictions.append(pred[0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred
    return np.array(future_predictions)


def predict_future_params(model, last_sequence: np.ndarray, future_locations: np.ndarray, steps: int) -> np.ndarray:
    """Autoregressive rollout of the parameter model, fed by the location model's
    own forecasted positions each step (notebook logic, unchanged except for
    the same clip as predict_future_locations, for the same reason)."""
    n_pos = len(POSITION_COLUMNS)
    future_predictions = []
    current_sequence = last_sequence.copy()
    for i in range(steps):
        pred = model.predict(current_sequence.reshape(1, SEQ_LENGTH, len(ALL_COLUMNS)), verbose=0)
        pred = np.clip(pred, 0.0, 1.0)
        future_predictions.append(pred[0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = np.concatenate([future_locations[i], pred[0]])
    return np.array(future_predictions)


def backtest(location_model, param_model, scaler, df: pd.DataFrame, backtest_hours: int) -> dict | None:
    """
    Honesty check on the forecast: hold out the last `backtest_hours` of
    ALREADY-OBSERVED data, forecast that same window using only the data
    before it, and compare against what actually happened. This is the one
    chart that shows whether the autoregressive rollout is trustworthy, and
    how fast its error grows with forecast horizon -- an LSTM that feeds its
    own prediction back in as input for hundreds of steps compounds error,
    and a forecast page that never shows this is hiding the model's weakest
    point.
    """
    n_pos = len(POSITION_COLUMNS)
    n_other = len(OTHER_COLUMNS)

    if len(df) < SEQ_LENGTH + backtest_hours + 1:
        print(f"  Not enough data for a {backtest_hours}h backtest (need {SEQ_LENGTH + backtest_hours + 1} rows, have {len(df)}); skipping.")
        return None

    scaled_all = scaler.transform(df[ALL_COLUMNS])
    cutoff = len(scaled_all) - backtest_hours

    context = scaled_all[cutoff - SEQ_LENGTH:cutoff]
    actual_future_scaled = scaled_all[cutoff:cutoff + backtest_hours]

    pred_locations = predict_future_locations(location_model, context[:, :n_pos], backtest_hours)
    pred_params = predict_future_params(param_model, context, pred_locations, backtest_hours)

    loc_padded = np.column_stack([pred_locations, np.zeros((backtest_hours, n_other))])
    param_padded = np.column_stack([np.zeros((backtest_hours, n_pos)), pred_params])
    pred_locations_orig = scaler.inverse_transform(loc_padded)[:, :n_pos]
    pred_params_orig = scaler.inverse_transform(param_padded)[:, n_pos:]

    actual_df = df.iloc[cutoff:cutoff + backtest_hours]
    actual_hgi_r = actual_df["HGI_R"].to_numpy()
    actual_flux1 = actual_df["Flux_1"].to_numpy()
    pred_hgi_r = pred_locations_orig[:, 0]
    pred_flux1 = pred_params_orig[:, OTHER_COLUMNS.index("Flux_1")]

    hgi_r_range = max(df["HGI_R"].max() - df["HGI_R"].min(), 1e-9)
    flux1_range = max(df["Flux_1"].max() - df["Flux_1"].min(), 1e-9)

    return {
        "backtest_hours": backtest_hours,
        "horizon_hours": list(range(1, backtest_hours + 1)),
        "HGI_R": {
            "actual": actual_hgi_r.round(4).tolist(),
            "predicted": pred_hgi_r.round(4).tolist(),
            "abs_error_pct_of_range": (np.abs(actual_hgi_r - pred_hgi_r) / hgi_r_range * 100).round(3).tolist(),
        },
        "Flux_1": {
            "actual": actual_flux1.round(6).tolist(),
            "predicted": pred_flux1.round(6).tolist(),
            "abs_error_pct_of_range": (np.abs(actual_flux1 - pred_flux1) / flux1_range * 100).round(3).tolist(),
        },
    }


def run(input_csv: str, loc_model_path: str, param_model_path: str,
        future_hours: int, out_path: str, backtest_hours: int = 72) -> None:

    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    df = df[ALL_COLUMNS]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[ALL_COLUMNS])

    X, _ = create_sequences(scaled, SEQ_LENGTH)
    if len(X) == 0:
        raise ValueError(f"Not enough rows ({len(df)}) for a {SEQ_LENGTH}-hour sequence window.")

    print("Loading real trained models ...")
    location_model = keras.models.load_model(loc_model_path, compile=False)
    param_model = keras.models.load_model(param_model_path, compile=False)

    print(f"Backtesting on the last {backtest_hours}h of observed data ...")
    backtest_result = backtest(location_model, param_model, scaler, df, backtest_hours)

    n_pos = len(POSITION_COLUMNS)
    last_location_seq = X[-1, :, :n_pos]
    last_full_seq = X[-1]

    print(f"Forecasting {future_hours} hours ahead (autoregressive rollout) ...")
    future_locations = predict_future_locations(location_model, last_location_seq, future_hours)
    future_params = predict_future_params(param_model, last_full_seq, future_locations, future_hours)

    # Inverse-transform back to physical units
    n_other = len(OTHER_COLUMNS)
    loc_padded = np.column_stack([future_locations, np.zeros((future_hours, n_other))])
    param_padded = np.column_stack([np.zeros((future_hours, n_pos)), future_params])

    future_locations_original = scaler.inverse_transform(loc_padded)[:, :n_pos]
    future_params_original = scaler.inverse_transform(param_padded)[:, n_pos:]

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_hours + 1, freq="h")[1:]

    result = {
        "meta": {
            "seq_length_hours": SEQ_LENGTH,
            "future_hours": future_hours,
            "input_rows": int(len(df)),
            "input_source": os.path.basename(input_csv),
            "position_columns": POSITION_COLUMNS,
            "other_columns": OTHER_COLUMNS,
            "last_observed_date": str(last_date),
        },
        "history": {
            "dates": [str(d) for d in df.index[-500:]],
            "HGI_R": df["HGI_R"].iloc[-500:].round(4).tolist(),
            "HGI_Lat": df["HGI_Lat"].iloc[-500:].round(4).tolist(),
            "HGI_Lon": df["HGI_Lon"].iloc[-500:].round(4).tolist(),
            "B_Avg": df["B_Avg"].iloc[-500:].round(5).tolist(),
            "Flux_1": df["Flux_1"].iloc[-500:].round(6).tolist(),
        },
        "forecast": {
            "dates": [str(d) for d in future_dates],
            "HGI_R": future_locations_original[:, 0].round(4).tolist(),
            "HGI_Lat": future_locations_original[:, 1].round(4).tolist(),
            "HGI_Lon": future_locations_original[:, 2].round(4).tolist(),
            "B_Avg": future_params_original[:, OTHER_COLUMNS.index("B_Avg")].round(5).tolist(),
            "Flux_1": future_params_original[:, OTHER_COLUMNS.index("Flux_1")].round(6).tolist(),
            "Proton_Density": future_params_original[:, OTHER_COLUMNS.index("Proton_Density")].round(6).tolist(),
        },
        "backtest": backtest_result,
    }

    def _finite_or_none(obj):
        """Safety net: json.dump() happily writes the literal tokens NaN /
        Infinity for non-finite floats, which is NOT valid JSON -- a
        browser's JSON.parse throws on them and, because this page runs
        everything in one sequential script, that one throw blanks the
        entire page (forecast chart, backtest chart, live panel, all of
        it), not just the one bad number. The clip in predict_future_*
        above should prevent non-finite values from ever reaching here;
        this just guarantees the file is always valid JSON even if a
        future change to the model/data reintroduces the failure mode.
        """
        if isinstance(obj, dict):
            return {k: _finite_or_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_finite_or_none(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    result = _finite_or_none(result)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"Wrote forecast -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="../data/demo/vy2_demo_merged.csv")
    p.add_argument("--loc-model", default="../models/location_model.h5")
    p.add_argument("--param-model", default="../models/param_model.h5")
    p.add_argument("--future-hours", type=int, default=720)
    p.add_argument("--backtest-hours", type=int, default=72)
    p.add_argument("--out", default="../outputs/forecast.json")
    args = p.parse_args()
    run(args.input, args.loc_model, args.param_model, args.future_hours, args.out, args.backtest_hours)
