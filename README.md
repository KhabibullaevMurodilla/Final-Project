# Voyager Telemetry Forecast

Forecasting Voyager spacecraft trajectory and local interstellar-medium conditions with LSTMs, trained on NASA's own merged hourly telemetry.

**Results page:** https://khabibullaevmurodilla.github.io/Final-Project/ (live once GitHub Pages is turned on for this repo — see setup below; nothing to run until then)

This started as MSc coursework (`Final_Project.ipynb`) and has since been repackaged as a runnable pipeline with a results page — see the `voyager-forecast-pipeline/` folder for the production version. This file documents the project as a whole; the add-on folder has its own README with setup details.

### One-time setup to make the results page public

The page lives at `voyager-forecast-pipeline/web/index.html` in this repo. To serve it at the URL above:

1. Repo **Settings → Pages** → Source: **Deploy from a branch** → Branch `main`, folder `/voyager-forecast-pipeline/web` → Save.
2. Wait a minute or two for the first build.

That's it — anyone can open that link afterward, no GitHub account or Claude account needed. The two GitHub Actions workflows (below) keep it updated automatically from there.

## What this is

Two LSTM networks, trained on NASA's Voyager-2 merged hourly dataset (34 columns: position, interplanetary magnetic field, solar wind, and 18 proton-flux energy channels spanning <2 MeV to >300 MeV), forecast the spacecraft's position and local space-environment readings hour by hour, autoregressively, for up to 10 days past the last observation.

| Model | Architecture | Input | Output | Loss (scaled MSE) |
|---|---|---|---|---|
| `location_model.h5` | LSTM(64) → Dense(3) | 120h × 3 cols (position) | next-hour position | 6.09 × 10⁻⁷ (test) |
| `param_model.h5` | LSTM(128) → LSTM(64) → LSTM(32) → Dense(28) | 120h × 31 cols (all) | next-hour 28 params | 1.29 × 10⁻⁴ (train) |

Both figures are read directly from the original training run, not re-measured.

## Is the forecast any good?

The honest answer needs its own check, not just a confident-looking chart: the pipeline holds out the last 72 hours of already-observed data, forecasts that window blind, and compares it to what actually happened. Error on distance climbs from roughly 8% of the observed range at 1 hour out to over 50% by 72 hours — expected for an autoregressive LSTM that feeds its own prediction back in as input hundreds of times in a row, but worth seeing rather than hiding. The results page shows this chart alongside the forecast itself, not as a footnote.

## Live data, not just history

Beyond the historical merged file the models trained on, the pipeline also pulls two things that are live right now:

- **JPL Horizons System** — Voyager 1 and 2's actual current distance from the Sun and velocity, from JPL's own navigation solution.
- **NOAA SWPC real-time solar wind** — live plasma and magnetic-field readings near Earth, for contrast with conditions far outside the heliosphere.

Both refresh hourly via a scheduled GitHub Action and feed the "Right now" panel on the results page.

## Data source and an honest caveat

Real data: [NASA SPDF Voyager archive](https://spdf.gsfc.nasa.gov/pub/data/voyager/). The two trained models in this repo were trained on this real data and are used as-is throughout — not retrained, not reconstructed.

The pipeline itself (fetch → clean → forecast) is built to run against that real archive directly, and does so automatically through the GitHub Actions workflows described in `voyager-forecast-pipeline/README.md`. The copy of `forecast.json` committed in this repo at any given moment reflects whatever that workflow last fetched — real NASA data once the workflow has run at least once in your environment, or a clearly-labeled calibrated synthetic fallback before that. The results page states which one it's currently showing.

## Repo layout

```
Final_Project.ipynb              original coursework notebook
location_model.h5                 trained weights (real, unmodified)
param_model.h5                     trained weights (real, unmodified)
voyager-forecast-pipeline/        production pipeline + results page (add-on, see its own README)
  src/
    preprocess.py                  real NASA .asc parsing + cleaning
    predict.py                     forecasting + held-out backtest
    fetch_nasa_data.py             downloads real .asc files (needs normal internet)
    fetch_live_data.py             live Voyager position + solar wind
    run_pipeline.py                ties the above together
    generate_demo_data.py          calibrated synthetic fallback data
  web/index.html                  the results page
.github/workflows/
  update-forecast.yml              weekly: real data -> forecast -> commit
  update-live-stats.yml            hourly: live position + solar wind -> commit
```

## Running it

```bash
cd voyager-forecast-pipeline
pip install tensorflow-cpu pandas numpy scikit-learn requests
python src/run_pipeline.py --satellite vy2 --start-year 2015 --end-year 2025 \
  --future-hours 240 --out web/forecast.json --allow-demo-fallback
```

Full setup, including the GitHub Actions automation, is in `voyager-forecast-pipeline/README.md`.
