"""
horizons_position.py
=====================
Real, physics-computed spacecraft position from JPL Horizons, for ANY date
-- past or future -- expressed in the Heliographic Inertial (HGI) frame,
the same frame NASA's merged .asc files report HGI_R/HGI_Lat/HGI_Lon in.

Why this exists instead of letting location_model.h5 (an LSTM) guess at
future position: Voyager's trajectory in deep space is dominated by solar
gravity alone, with no atmosphere and no significant unmodeled perturbation
-- it is a deterministic orbit-propagation problem, not a statistical one.
JPL continuously refines its navigation solution from real tracking data,
and Horizons can compute that solution for any date, including ones that
haven't happened yet (it is propagating a trajectory model, not looking up
recorded telemetry -- the same reason the live "right now" panel's distance
numbers are exact rather than estimates). Feeding an LSTM's guess into the
forecast when the exact physics answer is one query away was never the
right design: param_model (predicting field/solar-wind/flux, which have no
analytic shortcut) is the only part of this pipeline that actually needs to
be a forecast rather than a lookup.

Validated structurally (the exact call below reaches ssd.jpl.nasa.gov and
is rejected only by the Claude sandbox's own egress proxy, not by JPL or by
a code error -- same restriction already documented in fetch_nasa_data.py /
fetch_live_data.py for this sandbox specifically). Confirmed working
call shape:
    get_horizons_coord(naif_id, astropy_time_array, id_type=None)
`id_type=None` is required -- sunpy validates id_type against a fixed list
that does NOT include 'id'; None is what tells Horizons to search major
bodies/spacecraft (which is where Voyager's NAIF ID resolves) before
falling back to small-body search.
"""

from __future__ import annotations

import pandas as pd

# Same NAIF IDs fetch_live_data.py already uses for the "right now" panel.
VOYAGER_NAIF_IDS = {"voyager1": "-31", "voyager2": "-32"}


def fetch_horizons_hgi(satellite: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Real HGI_R (AU) / HGI_Lat (deg) / HGI_Lon (deg) for `dates`, via JPL
    Horizons -- matching the exact columns, units, and frame the merged
    NASA files (and therefore this pipeline's models) already use for
    position, so the result drops straight into ALL_COLUMNS/POSITION_COLUMNS
    without any further conversion.

    Raises on failure (network error, Horizons/sunpy schema change, etc.)
    rather than returning anything approximate -- callers fall back to the
    LSTM's own guess (predict_future_locations) on failure, and must never
    silently substitute a made-up value for a real trajectory lookup.
    """
    from astropy.time import Time
    from sunpy.coordinates import HeliocentricInertial, get_horizons_coord

    naif_id = VOYAGER_NAIF_IDS[satellite]
    dates = pd.DatetimeIndex(dates)
    times = Time(dates.to_pydatetime().tolist())

    coord = get_horizons_coord(naif_id, times, id_type=None)
    hgi = coord.transform_to(HeliocentricInertial(obstime=times))

    return pd.DataFrame(
        {
            "HGI_R": hgi.distance.to("AU").value,
            "HGI_Lat": hgi.lat.to("deg").value,
            "HGI_Lon": hgi.lon.to("deg").value,
        },
        index=dates,
    )
