"""
app/domain/special/wf.py
Wildfire composite scoring with intermediate zarr checkpoint to avoid large dask graphs.

Algorithm:
  1. ds['burnability'] * ds['fwi']  → tmp/WF_composite.zarr  (flushes graph)
  2. open composite → score(...)    → returned Dataset per threshold scale
  3. For scale '100': min-max normalise the composite to 0-100 float16.
Then concatenate across scales along 'scoring' dim.
"""

from pathlib import Path
import logging

import gcsfs
import xarray as xr
import yaml

from app.domain.scoring import score_zarr, score_zarr_minmax
from app.utils.scoring_config import ScoringConfig

logger = logging.getLogger(__name__)

_PIPELINE_CFG = Path(__file__).parent.parent.parent / "config" / "pipeline.yaml"

_cfg_cache: dict = {}

def _tmp_base() -> str:
    if not _cfg_cache:
        with open(_PIPELINE_CFG) as f:
            _cfg_cache.update(yaml.safe_load(f))
    return _cfg_cache["gcs"]["tmp_base"]


def _tmp_path(stage: str) -> str:
    return f"{_tmp_base()}/WF_{stage}.zarr"


def score_wf(
    ds: xr.Dataset,
    scales: list = None,
    ref_scenario: str = None,
    ref_time: str = None,
) -> xr.Dataset:
    """
    Produce multi-scale Wildfire composite scores.

    Args:
        ds: Input Dataset with variables 'burnability' and 'fwi'
        scales: List of scale strings, e.g. ['5', '10', '100'].
                '100' triggers min-max normalization of the composite.
        ref_scenario: Scenario to use for min-max calibration (e.g. 'RCP85').
        ref_time: Time period to use for min-max calibration (e.g. 'Lt', 'St').

    Returns:
        xr.Dataset with variable 'score' and coordinate dim 'scoring'
    """
    if scales is None:
        scales = ["5", "10"]

    config = ScoringConfig()
    threshold_scales = [s for s in scales if s != "100"]
    include_minmax = "100" in scales

    logger.info(f"[WF] Starting composite scoring — scales={scales}")

    # Stage 1: multiply — flush graph before scoring loop (always needed)
    composite_path = _tmp_path("composite")
    fs = gcsfs.GCSFileSystem()
    if fs.exists(f"{composite_path.removeprefix('gs://')}/.zmetadata"):
        logger.info(f"  [WF] Stage 1 skipped — reusing existing tmp composite: {composite_path}")
    else:
        logger.info(f"  [WF] Stage 1: multiplying burnability × fwi → {composite_path}")
        # Fill FWI coastal gaps (25 km cells clipped at coast leave NaN at valid 1 km land pixels).
        # ffill+bfill with limit=30 covers gaps up to ~25 km without rechunking the full dimension.
        fwi_filled = (
            ds["fwi"]
            .ffill(dim="lon", limit=30).bfill(dim="lon", limit=30)
            .ffill(dim="lat", limit=30).bfill(dim="lat", limit=30)
        )
        # Mask to 1 km susceptibility footprint — removes FWI bleed into ocean pixels.
        composite = (ds["burnability"] * fwi_filled).where(ds["burnability"].notnull())
        composite.rename("score").to_dataset().to_zarr(composite_path, mode="w")
        logger.info("  [WF] Stage 1 done")

    comp_ds = xr.open_zarr(composite_path)
    results = []

    for scale in threshold_scales:
        logger.info(f"  [WF] Stage 2: scoring composite → final score (scale={scale})")
        thresh, scores = config.get_thresholds("WF", scale)
        final = score_zarr(comp_ds["score"], thresh, scores, metric="p95")
        final = final.expand_dims("scoring").assign_coords(scoring=[scale])
        results.append(final)
        logger.info(f"  [WF] Stage 2 done (scale={scale})")

    if include_minmax:
        logger.info("  [WF] Stage 2: min-max normalization (scale=100)")
        minmax_path = _tmp_path("minmax")
        if fs.exists(f"{minmax_path.removeprefix('gs://')}/.zmetadata"):
            logger.info(f"  [WF] Minmax checkpoint exists — reusing: {minmax_path}")
        else:
            minmax = score_zarr_minmax(comp_ds["score"], ref_scenario=ref_scenario, ref_time=ref_time)
            logger.info(f"  [WF] Checkpointing minmax → {minmax_path}")
            minmax["score"].to_zarr(minmax_path, mode="w")
        minmax = xr.open_zarr(minmax_path)
        minmax = minmax.expand_dims("scoring").assign_coords(scoring=["100"])
        results.append(minmax)
        logger.info("  [WF] Min-max normalization done")

    logger.info("[WF] All scales complete — concatenating results")
    return xr.concat(results, dim="scoring")


def cleanup_tmp() -> None:
    """Delete the WF composite and minmax temp zarrs (call after final write)."""
    paths = [_tmp_path("composite"), _tmp_path("minmax")]
    fs = gcsfs.GCSFileSystem()
    for path in paths:
        gcs_path = path.removeprefix("gs://")
        try:
            if fs.exists(gcs_path):
                fs.rm(gcs_path, recursive=True)
                logger.info(f"  Deleted tmp: {path}")
        except Exception as e:
            logger.warning(f"  Failed to delete tmp {path}: {e}")
