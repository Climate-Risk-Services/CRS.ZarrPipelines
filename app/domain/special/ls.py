"""
app/domain/special/ls.py
Landslide composite scoring with intermediate zarr checkpoints to avoid large dask graphs.

Algorithm per threshold scale:
  1. score(ds['ari'])                → tmp/LS_ari_{scale}.zarr
  2. score(ds['susceptibility'])     → tmp/LS_susc_{scale}.zarr
  3. open ari + susc → multiply      → tmp/LS_composite_{scale}.zarr
  4. score(composite)                → returned Dataset

For scale '100' (min-max normalization):
  Uses the composite from the first threshold scale computed (or computes a 5-pt
  reference composite if no threshold scale is in scales). Normalises to 0-100
  float16 using the global min/max of the composite.
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

def _load_cfg() -> dict:
    with open(_PIPELINE_CFG) as f:
        return yaml.safe_load(f)

_cfg_cache: dict = {}

def _tmp_base() -> str:
    if not _cfg_cache:
        _cfg_cache.update(_load_cfg())
    return _cfg_cache["gcs"]["tmp_base"]


def _tmp_path(stage: str, scale: str) -> str:
    return f"{_tmp_base()}/LS_{stage}_{scale}.zarr"


def _compute_composite(ds: xr.Dataset, scale: str, config: ScoringConfig) -> str:
    """Run stages 1-3 for a given scale and return the composite zarr path."""
    composite_path = _tmp_path("composite", scale)

    fs = gcsfs.GCSFileSystem()
    if fs.exists(f"{composite_path.removeprefix('gs://')}/.zmetadata"):
        logger.info(f"  [LS scale={scale}] Stage 1-3 skipped — reusing existing tmp composite: {composite_path}")
        return composite_path

    thresh_ari, scores_ari = config.get_thresholds(
        "LS", scale, threshold_type="thresholds_ari"
    )
    thresh_susc, scores_susc = config.get_thresholds(
        "LS", scale, threshold_type="thresholds_susceptibility"
    )

    ari_path = _tmp_path("ari", scale)
    logger.info(f"  [LS scale={scale}] Stage 1: scoring ARI → {ari_path}")
    score_zarr(ds["ari"], thresh_ari, scores_ari).to_zarr(ari_path, mode="w")
    logger.info(f"  [LS scale={scale}] Stage 1 done")

    susc_path = _tmp_path("susc", scale)
    logger.info(f"  [LS scale={scale}] Stage 2: scoring susceptibility → {susc_path}")
    score_zarr(ds["susceptibility"], thresh_susc, scores_susc).to_zarr(susc_path, mode="w")
    logger.info(f"  [LS scale={scale}] Stage 2 done")

    logger.info(f"  [LS scale={scale}] Stage 3: multiplying ARI × susceptibility → {composite_path}")
    ari_ds = xr.open_zarr(ari_path)
    susc_ds = xr.open_zarr(susc_path)
    (ari_ds["score"] * susc_ds["score"]).astype("float16").rename("score").to_dataset().to_zarr(
        composite_path, mode="w"
    )
    logger.info(f"  [LS scale={scale}] Stage 3 done")
    return composite_path


def score_ls(
    ds: xr.Dataset,
    scales: list = None,
    ref_scenario: str = None,
    ref_time: str = None,
) -> xr.Dataset:
    """
    Produce multi-scale Landslide composite scores using staged zarr writes.

    Args:
        ds: Input Dataset with variables 'ari' and 'susceptibility'
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

    logger.info(f"[LS] Starting composite scoring — scales={scales}")
    results = []
    composite_path_ref = None  # composite zarr for min-max normalization

    for scale in threshold_scales:
        logger.info(f"[LS] Processing threshold scale={scale}")
        thresh_final, scores_final = config.get_thresholds(
            "LS", scale, threshold_type="thresholds_final"
        )

        composite_path = _compute_composite(ds, scale, config)
        if composite_path_ref is None:
            composite_path_ref = composite_path  # use first computed composite for minmax

        # Stage 4: score final
        logger.info(f"  [LS scale={scale}] Stage 4: scoring composite → final score")
        comp_ds = xr.open_zarr(composite_path)
        final = score_zarr(comp_ds["score"], thresh_final, scores_final, metric="landslide_proxy")
        final = final.expand_dims("scoring").assign_coords(scoring=[scale])
        results.append(final)
        logger.info(f"  [LS scale={scale}] Stage 4 done")

    if include_minmax:
        logger.info("[LS] Processing min-max normalization (scale=100)")
        # If no threshold scale was computed, derive composite via 5-pt reference
        if composite_path_ref is None:
            logger.info("[LS] No threshold scale computed; deriving composite via scale=5 reference")
            composite_path_ref = _compute_composite(ds, "5", config)

        minmax_path = _tmp_path("minmax", "100")
        fs = gcsfs.GCSFileSystem()
        if fs.exists(f"{minmax_path.removeprefix('gs://')}/.zmetadata"):
            logger.info(f"  [LS] Minmax checkpoint exists — reusing: {minmax_path}")
        else:
            comp_ds = xr.open_zarr(composite_path_ref)
            minmax = score_zarr_minmax(comp_ds["score"], ref_scenario=ref_scenario, ref_time=ref_time)
            logger.info(f"  [LS] Checkpointing minmax → {minmax_path}")
            minmax["score"].to_zarr(minmax_path, mode="w")
        minmax = xr.open_zarr(minmax_path)
        minmax = minmax.expand_dims("scoring").assign_coords(scoring=["100"])
        results.append(minmax)
        logger.info("[LS] Min-max normalization done")

    logger.info("[LS] All scales complete — concatenating results")
    return xr.concat(results, dim="scoring")


def cleanup_tmp(scales: list) -> None:
    """Delete all LS temp zarrs for the given scales (call after final write)."""
    threshold_scales = [s for s in scales if s != "100"]
    # If only "100" was requested, a "5" reference composite was created
    if not threshold_scales and "100" in scales:
        threshold_scales = ["5"]
    paths = [
        _tmp_path(stage, scale)
        for scale in threshold_scales
        for stage in ("ari", "susc", "composite")
    ]
    if "100" in scales or not threshold_scales:
        paths.append(_tmp_path("minmax", "100"))
    fs = gcsfs.GCSFileSystem()
    for path in paths:
        gcs_path = path.removeprefix("gs://")
        try:
            if fs.exists(gcs_path):
                fs.rm(gcs_path, recursive=True)
                logger.info(f"  Deleted tmp: {path}")
        except Exception as e:
            logger.warning(f"  Failed to delete tmp {path}: {e}")
