# voyager-forecast-pipeline (add-on)

This is a **standalone add-on folder** — drop it into your `Final-Project` repo as
a new top-level folder. It does not read, write, or require changes to anything
already in the repo. It carries its own copies of the two trained models, so it
works the moment it's added.

```
your-repo/
  Final_Project.ipynb          <- untouched
  location_model.h5            <- untouched
  param_model.h5                <- untouched
  .github/workflows/
    update-forecast.yml        <- NEW, goes at repo root (see "Automatic updates" below)
  voyager-forecast-pipeline/   <- new folder, this upload
    src/
    models/                    <- copies of the same two files, used internally
    outputs/forecast.json      <- precomputed real forecast (manual-run output)
    web/
      index.html                <- the results page, same as the one published here
      forecast.json             <- what the page actually reads (GitHub Pages serves this)
    README.md                  <- this file
```

## What's real vs. demo right now

- `models/*.h5` — your actual trained weights, copied in, not modified.
- `src/preprocess.py` — real NASA `.asc` parsing logic, lifted from your notebook.
- `src/predict.py` — real autoregressive forecasting logic, lifted from your
  notebook, running on your real models.
- `web/forecast.json` — the actual output of running `predict.py` against your
  two real models. The *model inference is real*; the *input data it ran on*
  is a calibrated synthetic stand-in, because the sandbox this was assembled
  in cannot reach NASA's data server. This is disclosed on the page itself.
- `web/index.html` — reads `forecast.json` (same folder) and renders it. Same
  file already published at: https://claude.ai/artifact/33sv5DbTZayyBVzeRdt6nH

## Two new additions: forecast honesty check, and genuinely live data

**1. Backtest chart.** The forecast chart on its own can't tell you whether the
model is any good -- it's just extrapolating. `src/predict.py` now also holds
out the last 72 hours of *already-observed* data, forecasts that same window
blind, and compares it to what actually happened. The page shows both: the
actual-vs-predicted overlay, and how the error grows with forecast horizon.
On an LSTM that feeds its own output back in as input for hundreds of steps,
that error growth is real and worth seeing -- this is the chart that keeps
the page honest about how far out to trust it.

**2. Live data, not just the historical archive.** `src/fetch_live_data.py`
pulls two things that are genuinely live, right now, not from a static file:

- **JPL Horizons System** -- Voyager 1 and Voyager 2's *current* distance
  from the Sun and radial velocity, straight from JPL's own navigation
  solution (not the merged historical file the models trained on).
- **NOAA SWPC real-time solar wind** -- live plasma and magnetic-field
  readings from the DSCOVR satellite at the Sun-Earth L1 point, for a "what's
  happening near Earth right now" comparison against conditions far outside
  the heliosphere.

Both run the same way as the NASA archive fetch: blocked from the sandbox
this was assembled in, but normal HTTP calls that work fine from a GitHub
Actions runner. See `.github/workflows/update-live-stats.yml` below.

## Automatic updates from real NASA data (GitHub Actions)

`fetch_nasa_data.py` and `run_pipeline.py` can't reach NASA from the sandbox this
was built in, but a **GitHub Actions runner is not sandboxed the same way** — it
has ordinary internet access. So the fetch that fails here runs fine there.

One-time setup:

1. Move both files from `github-workflow/.github/workflows/` (shipped alongside
   this folder, not inside it) to your repo root, i.e.
   `your-repo/.github/workflows/update-forecast.yml` and
   `your-repo/.github/workflows/update-live-stats.yml`. GitHub only reads
   workflows from that exact path, which is why they aren't nested inside this
   add-on folder.
2. Commit and push this folder and both workflow files.
3. In your repo's **Settings → Actions → General**, confirm Actions are enabled
   (they are by default). Then run each once by hand from the **Actions** tab
   → pick the workflow → **Run workflow** -- or just wait for their schedules:
   `update-forecast.yml` weekly (Monday 06:00 UTC), `update-live-stats.yml`
   hourly.

What it does, each run:

1. Downloads real `vy2_{year}.asc` files for the year range you give it (default
   2015-2025) straight from `spdf.gsfc.nasa.gov`.
2. Cleans them with the exact logic from your notebook (`src/preprocess.py`).
3. Runs both real trained models over them (`src/predict.py`).
4. Commits the refreshed `voyager-forecast-pipeline/web/forecast.json` straight
   back to the repo.

From that point on, every run replaces the synthetic numbers with a forecast
built on genuine Voyager-2 telemetry — no manual download, no code changes.

## Showing results straight away (GitHub Pages)

**Use the "GitHub Actions" Pages source, not "Deploy from a branch".** The
branch-deploy method only lets you serve `/` (repo root) or `/docs` — it can't
point at an arbitrary folder like `voyager-forecast-pipeline/web`, and it runs
everything through Jekyll by default, which is why it may fail looking for a
`/docs` folder and a Jekyll theme this plain HTML page doesn't use or need.
Deploying through Actions serves the folder as-is, no Jekyll involved.

1. Move `deploy-pages.yml` (shipped in `github-workflow/.github/workflows/`
   alongside the other two) to your repo root, same as the others:
   `your-repo/.github/workflows/deploy-pages.yml`.
2. Repo **Settings → Pages** → Source: **GitHub Actions**. That's the only
   setting needed — don't pick a branch or folder here.
3. Commit and push. The workflow runs automatically on that push (and on every
   future push to `voyager-forecast-pipeline/web/`), and GitHub shows the live
   URL both in the Actions run's summary and under Settings → Pages once it's
   deployed — typically `https://<your-username>.github.io/Final-Project/`.
4. From then on, every time `update-forecast.yml` or `update-live-stats.yml`
   commits a change to that folder, `deploy-pages.yml` fires automatically and
   redeploys — nothing further to run by hand.

## Running it yourself locally (no GitHub Actions)

```bash
pip install tensorflow-cpu pandas numpy scikit-learn requests
python src/run_pipeline.py --satellite vy2 --start-year 2015 --end-year 2025 \
  --future-hours 240 --out web/forecast.json --allow-demo-fallback
```

`--allow-demo-fallback` means: if the real NASA fetch fails (e.g. you're also
running this from a restricted network), it falls back to the same calibrated
demo data this bundle ships with, instead of erroring out.

## Running the bundled demo only (fastest way to see it work)

```bash
pip install tensorflow-cpu pandas numpy scikit-learn
python src/generate_demo_data.py
python src/predict.py --input data/demo/vy2_demo_merged.csv \
  --future-hours 240 --out web/forecast.json
```

Then open `web/index.html`.

Nothing in any of the above touches the original repo files — everything reads
and writes only inside this add-on folder (plus the one workflow file, which is
new, not an edit to an existing one).
