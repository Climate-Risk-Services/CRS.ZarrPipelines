"""
app/domain/special/rf.py
River Flood scoring with flood protection subtraction.

Algorithm:
  1. Materialise ds['inundation_depth'] → tmp/RF_inundation.zarr  (breaks large dask graph)
     Skip if tmp already exists (crash recovery).
  2. Per threshold scale:
       a. Score inundation from tmp zarr
       b. Subtract protection × multiplier (1 for 5-pt, 2 for 10-pt)
       c. Clamp to 0
  3. Scale '100': min-max normalise raw inundation, subtract protection × 20, clamp to 0.
"""

from pathlib import Path
import logging

import gcsfs
import numpy as np
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
    return f"{_tmp_base()}/RF_{stage}.zarr"


_PROTECTION_MULTIPLIER = {"5": 1, "10": 2, "100": 20}


def score_rf(
    ds: xr.Dataset,
    scales: list = None,
    inundation_var: str = "return_period_0_5_m",
    protection_var: str = "flood_protection",
    ref_scenario: str = None,
    ref_time: str = None,
) -> xr.Dataset:
    """
    Produce multi-scale River Flood scores with flood protection subtraction.

    Args:
        ds: Input Dataset containing inundation_var and protection_var.
        scales: List of scale strings, e.g. ['5', '10', '100'].
        inundation_var: Name of the primary inundation variable in ds.
        protection_var: Name of the float64 1–5 flood protection variable in ds.
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

    logger.info(f"[RF] Starting flood-protection-adjusted scoring — scales={scales}")

    # Stage 1: materialise raw inundation to break large dask graph.
    # Collapse model dim → statistic='mean' to match the statistic-dim convention
    # used by all other hazards (HS, HP, etc.) so aggregation filters correctly.
    inundation_path = _tmp_path("inundation")
    fs = gcsfs.GCSFileSystem()
    if fs.exists(f"{inundation_path.removeprefix('gs://')}/.zmetadata"):
        logger.info(f"  [RF] Stage 1 skipped — reusing existing tmp: {inundation_path}")
    else:
        logger.info(f"  [RF] Stage 1: materialising {inundation_var} → {inundation_path}")
        inundation = ds[inundation_var]
        if "model" in inundation.dims:
            if inundation.sizes["model"] > 1:
                logger.info(f"  [RF] Computing ensemble mean across {inundation.sizes['model']} models")
                inundation = inundation.mean("model")
            else:
                inundation = inundation.squeeze("model", drop=True)
            inundation = inundation.expand_dims({"statistic": ["mean"]})

        # Fill RCP45/Cc and RCP85/Cc from HISTORICAL/Cc, then drop HISTORICAL.
        # In the RF dataset Cc (current climate) only exists under HISTORICAL;
        # RCP45/Cc and RCP85/Cc are NaN. This normalises the structure so all
        # downstream steps see a clean (RCP45, RCP85) × (Cc, St, Mt, Lt) grid.
        if "scenario" in inundation.dims and "HISTORICAL" in [str(s) for s in inundation.scenario.values]:
            logger.info("  [RF] Filling RCP*/Cc from HISTORICAL/Cc and dropping HISTORICAL scenario")
            hist_cc = inundation.sel(scenario="HISTORICAL", time="Cc", drop=True)
            rcp_scenarios = [s for s in inundation.scenario.values if str(s) != "HISTORICAL"]
            filled_parts = []
            for sc in rcp_scenarios:
                sc_data = inundation.sel(scenario=sc)
                times = []
                for t in inundation.time.values:
                    if str(t) == "Cc":
                        times.append(hist_cc.expand_dims({"time": ["Cc"]}))
                    else:
                        times.append(sc_data.sel(time=t).expand_dims({"time": [str(t)]}))
                filled_parts.append(
                    xr.concat(times, dim="time").expand_dims({"scenario": [sc]})
                )
            inundation = xr.concat(filled_parts, dim="scenario")

        ds_inundation = inundation.astype("float16").rename("inundation").to_dataset()
        # String coords from a reopened zarr are read-only; convert via tolist() so VLenUTF8 can encode them.
        # Handles both dtype('O') and fixed-length unicode dtype.kind=='U' (e.g. '<U6').
        ds_inundation = ds_inundation.assign_coords(
            {c: np.array(ds_inundation[c].values.tolist())
             for c in ds_inundation.coords if ds_inundation[c].dtype.kind in ("U", "O", "S")}
        )
        ds_inundation.to_zarr(inundation_path, mode="w")
        logger.info("  [RF] Stage 1 done")

    inundation_ds = xr.open_zarr(inundation_path)
    protection = ds[protection_var]  # float64 1–5
    results = []

    for scale in threshold_scales:
        multiplier = _PROTECTION_MULTIPLIER[scale]
        logger.info(f"  [RF] Stage 2: scoring inundation + subtracting protection×{multiplier} (scale={scale})")
        thresholds, scores = config.get_thresholds("RF", scale)
        raw_scored = score_zarr(inundation_ds["inundation"], thresholds, scores, metric=inundation_var)
        adjusted = (raw_scored["score"] - protection * multiplier).clip(min=0).astype("float16")
        final = adjusted.rename("score").to_dataset()
        final = final.expand_dims("scoring").assign_coords(scoring=[scale])
        results.append(final)
        logger.info(f"  [RF] Stage 2 done (scale={scale})")

    if include_minmax:
        logger.info("  [RF] Stage 3: min-max normalisation + subtracting protection×20 (scale=100)")
        minmax_path = _tmp_path("minmax100")
        if fs.exists(f"{minmax_path.removeprefix('gs://')}/.zmetadata"):
            logger.info(f"  [RF] Minmax100 checkpoint exists — reusing: {minmax_path}")
        else:
            raw_100 = score_zarr_minmax(inundation_ds["inundation"], ref_scenario=ref_scenario, ref_time=ref_time)
            adjusted_100 = (raw_100["score"] - protection * 20).clip(min=0).astype("float16")
            logger.info(f"  [RF] Checkpointing minmax100 → {minmax_path}")
            ds_100 = adjusted_100.rename("score").to_dataset()
            # String coords from a reopened zarr are read-only; convert via tolist() so VLenUTF8 can encode them.
            # Handles both dtype('O') and fixed-length unicode dtype.kind=='U' (e.g. '<U6').
            ds_100 = ds_100.assign_coords(
                {c: np.array(ds_100[c].values.tolist())
                 for c in ds_100.coords if ds_100[c].dtype.kind in ("U", "O", "S")}
            )
            ds_100.to_zarr(minmax_path, mode="w")
        final_100 = xr.open_zarr(minmax_path)
        final_100 = final_100.expand_dims("scoring").assign_coords(scoring=["100"])
        results.append(final_100)
        logger.info("  [RF] Stage 3 done")

    logger.info("[RF] All scales complete — concatenating results")
    return xr.concat(results, dim="scoring")


def cleanup_tmp() -> None:
    """Delete RF tmp zarrs (inundation + minmax100) after final write."""
    paths = [_tmp_path("inundation"), _tmp_path("minmax100")]
    fs = gcsfs.GCSFileSystem()
    for path in paths:
        gcs_path = path.removeprefix("gs://")
        try:
            if fs.exists(gcs_path):
                fs.rm(gcs_path, recursive=True)
                logger.info(f"  Deleted tmp: {path}")
        except Exception as e:
            logger.warning(f"  Failed to delete tmp {path}: {e}")
