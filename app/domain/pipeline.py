"""
app/domain/pipeline.py
Orchestration: score hazards and run GADM aggregations.
Called by the FastAPI background tasks.
"""

import logging
from typing import List, Optional

import dask
import xarray as xr

from app.domain.combine import combine_scores
from app.domain.gadm_aggregations import aggregate_gadm, write_csv
from app.domain.scoring import score_zarr, score_zarr_minmax, score_zarr_multi
from app.domain.special import score_ls, score_wf, score_rf, score_cf
from app.domain.special.ls import cleanup_tmp as _cleanup_ls
from app.domain.special.wf import cleanup_tmp as _cleanup_wf
from app.domain.special.rf import cleanup_tmp as _cleanup_rf
from app.domain.special.cf import cleanup_tmp as _cleanup_cf
from app.utils.pipeline_config import load_pipeline_config
from app.utils.scoring_config import ScoringConfig

logger = logging.getLogger(__name__)

ALL_HAZARD_CODES = [
    "ER", "CER", "CF", "RF", "CS", "HS", "DR", "HP",
    "LS", "SUB", "TC", "TS", "WF", "WS",
]


def _pipeline_cfg() -> dict:
    return load_pipeline_config()


def _open_zarr(input_path: str) -> xr.Dataset:
    return xr.open_zarr(input_path)


def _output_path(hazard_code: str) -> str:
    cfg = _pipeline_cfg()
    base = cfg["gcs"]["output_zarr_base"]
    return f"{base}/{hazard_code}.zarr"


def _csv_output_path(hazard_code: str, gadm_level: int) -> str:
    cfg = _pipeline_cfg()
    base = cfg["gcs"]["csv_output"]
    return f"{base}/{hazard_code}_adm{gadm_level}.csv"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _write_scored_zarr(scored: xr.Dataset, out_path: str) -> None:
    """
    Write scored dataset to zarr using native append/region writes:

    - Store absent                        → fresh mode="w"
    - Scale already exists in store       → region write (only that slice)
    - Scale new, sorts after all existing → append_dim (only new slices)
    - Scale new, sorts before existing    → fallback full rewrite (warn)

    Avoids rewriting unchanged scales — a ~3× saving when adding one scale
    to an existing two-scale store.
    """
    import numpy as np

    def _fix_and_cast(ds: xr.Dataset) -> xr.Dataset:
        """Cast to float16 and fix string coords for zarr compat."""
        ds = ds.astype(np.float16)
        for coord_name in list(ds.coords):
            if ds.coords[coord_name].dtype.kind in ("U", "O", "S"):
                vals = np.array(ds.coords[coord_name].values.tolist())
                if coord_name == "scoring":
                    vals = vals.astype("<U3")  # '100' is 3 chars; fix dtype on first write
                ds = ds.assign_coords({coord_name: vals})
        return ds

    new_scales = [str(s) for s in scored.scoring.values]

    # --- No existing store: fresh write ---
    try:
        existing = xr.open_zarr(out_path)
        existing_scales = [str(s) for s in existing.scoring.values]
    except Exception:
        logger.info("  No existing store — creating fresh")
        # Pre-clean any orphaned GCS objects (zarr mode='w' rmdir can fail on GCS
        # with OSError 404 when objects exist but .zmetadata is missing).
        import gcsfs as _gcsfs
        try:
            _gcsfs.GCSFileSystem().rm(out_path.removeprefix("gs://"), recursive=True)
        except Exception as _e:
            logger.debug(f"  Pre-clean skipped (nothing to delete): {_e}")
        # Use mode='a' to create fresh without internal rmdir call.
        _fix_and_cast(scored).chunk({"scoring": 1}).to_zarr(out_path, mode="a")
        return

    to_replace = [s for s in new_scales if s in existing_scales]
    to_append = [s for s in new_scales if s not in existing_scales]

    logger.info(
        f"  Existing scales: {existing_scales} | replace: {to_replace} | append: {to_append}"
    )

    # Append is safe only when every new scale sorts after every existing scale
    if to_append and existing_scales:
        max_existing = max(int(s) for s in existing_scales)
        append_safe = all(int(s) > max_existing for s in to_append)
    else:
        append_safe = True

    if to_append and not append_safe:
        # New scale would sort before an existing one — must rebuild in sorted order
        logger.warning(
            f"  Unsafe append: {to_append} would interleave with {existing_scales}. "
            "Falling back to full rewrite."
        )
        keep_scales = [s for s in existing_scales if s not in new_scales]
        if keep_scales:
            kept = existing.sel(scoring=keep_scales).compute()  # materialise before overwriting store
            merged = xr.concat([kept, scored], dim="scoring")
        else:
            merged = scored
        order = sorted(range(merged.sizes["scoring"]), key=lambda i: int(merged.scoring.values[i]))
        merged = merged.isel(scoring=order)
        _fix_and_cast(merged).chunk({"scoring": 1}).to_zarr(out_path, mode="w")
        return

    # --- Region write for replaced scales (no shape change) ---
    for scale in to_replace:
        idx = existing_scales.index(scale)
        scale_ds = _fix_and_cast(scored.sel(scoring=[scale]))
        # Region write requires the coordinate to be dropped (already in store)
        # Drop all vars/coords that don't share the 'scoring' dim (already in store)
        vars_to_drop = [k for k, v in scale_ds.variables.items() if "scoring" not in v.dims]
        scale_ds = scale_ds.drop_vars(vars_to_drop)
        scale_ds.to_zarr(out_path, region={"scoring": slice(idx, idx + 1)})
        logger.info(f"  Region-replaced scale '{scale}' at index {idx}")

    # --- Append new scales (extends the scoring dim) ---
    if to_append:
        append_ds = _fix_and_cast(scored.sel(scoring=to_append)).chunk({"scoring": 1})
        append_ds.to_zarr(out_path, mode="a", append_dim="scoring")
        logger.info(f"  Appended scales {to_append}")


def score_hazard(hazard_code: str, scales: List[str] = None) -> xr.Dataset:
    """
    Score a single hazard for one or more scales. Writes output to GCS Zarr.

    Returns the scored Dataset.
    """
    if scales is None:
        scales = ["5", "10"]

    cfg = _pipeline_cfg()
    hazard_cfg = cfg["hazards"][hazard_code]
    hazard_type = hazard_cfg["type"]
    input_path = hazard_cfg["input"]

    logger.info(f"[{hazard_code}] Opening Zarr: {input_path}")
    ds = _open_zarr(input_path)

    threshold_scales = [s for s in scales if s != "100"]
    include_minmax = "100" in scales

    ref_scenario = hazard_cfg.get("minmax_ref_scenario")
    ref_time = hazard_cfg.get("minmax_ref_time")

    # --- special_ls / special_wf: handle all scales including "100" internally ---
    if hazard_type == "special_ls":
        scored = score_ls(ds, scales=scales, ref_scenario=ref_scenario, ref_time=ref_time)

    elif hazard_type == "special_wf":
        scored = score_wf(ds, scales=scales, ref_scenario=ref_scenario, ref_time=ref_time)

    elif hazard_type == "special_rf":
        scored = score_rf(
            ds,
            scales=scales,
            inundation_var=hazard_cfg.get("variable", "return_period_0_5_m"),
            protection_var=hazard_cfg.get("protection_variable", "flood_protection"),
            ref_scenario=ref_scenario,
            ref_time=ref_time,
        )

    elif hazard_type == "special_cf":
        scored = score_cf(
            ds,
            scales=scales,
            inundation_var=hazard_cfg.get("variable", "return_period_0_5_m"),
            protection_var=hazard_cfg.get("protection_variable", "flood_protection"),
            ref_scenario=ref_scenario,
            ref_time=ref_time,
        )

    # --- sub (multiply raw by 2 for 10-point threshold scoring) ---
    elif hazard_type == "sub":
        variable = hazard_cfg["variable"]
        scoring_config = ScoringConfig()
        results = []
        for scale in threshold_scales:
            logger.info(f"[{hazard_code}] Scoring scale={scale} (SUB: ×2 applied for 10-pt)")
            data = ds[variable] * 2 if scale == "10" else ds[variable]
            thresholds, scores = scoring_config.get_thresholds(hazard_code, scale)
            s = score_zarr(data, thresholds, scores, metric=variable)
            s = s.expand_dims("scoring").assign_coords(scoring=[scale])
            results.append(s)
            logger.info(f"[{hazard_code}] Scoring scale={scale} done")
        if include_minmax:
            logger.info(f"[{hazard_code}] Scoring scale=100 (min-max normalization)")
            # Normalise raw data (×2 is a scoring artifact, not a physical transform)
            minmax = score_zarr_minmax(
                ds[variable],
                ref_scenario=hazard_cfg.get("minmax_ref_scenario"),
                ref_time=hazard_cfg.get("minmax_ref_time"),
            )
            minmax = minmax.expand_dims("scoring").assign_coords(scoring=["100"])
            results.append(minmax)
            logger.info(f"[{hazard_code}] Scoring scale=100 done")
        scored = xr.concat(results, dim="scoring")

    # --- standard ---
    else:
        variable = hazard_cfg["variable"]
        scoring_config = ScoringConfig()
        metric_name = list(
            scoring_config.config["hazards"][hazard_code]["metrics"].keys()
        )[0]
        data = ds[variable]
        if "metric_select" in hazard_cfg and "metric" in data.dims:
            data = data.sel(metric=hazard_cfg["metric_select"], drop=True)
        if "scale_factor" in hazard_cfg:
            data = data * hazard_cfg["scale_factor"]
        for dim in list(data.dims):
            if dim not in ("lat", "lon", "x", "y") and data.sizes[dim] == 1:
                data = data.squeeze(dim, drop=True)

        results = []
        if threshold_scales:
            logger.info(f"[{hazard_code}] Scoring threshold scales={threshold_scales}")
            thresholds_per_scale = {
                scale: scoring_config.get_thresholds(hazard_code, scale)
                for scale in threshold_scales
            }
            scored_thresh = score_zarr_multi(
                data,
                scales=threshold_scales,
                thresholds_per_scale=thresholds_per_scale,
                metric=metric_name,
            )
            results.append(scored_thresh)
            logger.info(f"[{hazard_code}] Threshold scoring done")
        if include_minmax:
            logger.info(f"[{hazard_code}] Scoring scale=100 (min-max normalization)")
            minmax = score_zarr_minmax(
                data,
                ref_scenario=hazard_cfg.get("minmax_ref_scenario"),
                ref_time=hazard_cfg.get("minmax_ref_time"),
            )
            minmax = minmax.expand_dims("scoring").assign_coords(scoring=["100"])
            results.append(minmax)
            logger.info(f"[{hazard_code}] Scoring scale=100 done")
        scored = xr.concat(results, dim="scoring") if len(results) > 1 else results[0]

    out_path = _output_path(hazard_code)
    logger.info(f"[{hazard_code}] Writing scored Zarr to {out_path}")
    _write_scored_zarr(scored, out_path)

    if hazard_type == "special_ls":
        logger.info(f"[{hazard_code}] Cleaning up LS tmp zarrs")
        _cleanup_ls(scales)
    elif hazard_type == "special_wf":
        logger.info(f"[{hazard_code}] Cleaning up WF tmp zarrs")
        _cleanup_wf()
    elif hazard_type == "special_rf":
        logger.info(f"[{hazard_code}] Cleaning up RF tmp zarrs")
        _cleanup_rf()
    elif hazard_type == "special_cf":
        logger.info(f"[{hazard_code}] Cleaning up CF tmp zarrs")
        _cleanup_cf()

    logger.info(f"[{hazard_code}] Scoring complete")
    return scored


def score_all_hazards(scales: List[str] = None) -> None:
    """Score all supported hazards."""
    if scales is None:
        scales = ["5", "10"]
    for code in ALL_HAZARD_CODES:
        try:
            score_hazard(code, scales=scales)
        except Exception as exc:
            logger.error(f"[{code}] Scoring failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_hazard(hazard_code: str, gadm_levels: List[int] = None) -> None:
    """
    Aggregate scored Zarr to GADM boundaries and write CSV to GCS.
    Passes the GCS path string to aggregate_gadm so each Coiled worker
    opens its own Zarr connection rather than receiving a pickled dask graph.

    Args:
        gadm_levels: one or more of [1, 2]. Defaults to [1].
    """
    if gadm_levels is None:
        gadm_levels = [1]
    if isinstance(gadm_levels, int):
        gadm_levels = [gadm_levels]

    scored_path = _output_path(hazard_code)
    logger.info(f"[{hazard_code}] Aggregating from {scored_path}")

    for gadm_level in gadm_levels:
        df = aggregate_gadm(scored_path, hazard_code=hazard_code, gadm_level=gadm_level)
        csv_path = _csv_output_path(hazard_code, gadm_level)
        write_csv(df, csv_path)
        logger.info(f"[{hazard_code}] adm{gadm_level} aggregation complete → {csv_path}")


def aggregate_all_hazards(gadm_levels: List[int] = None) -> None:
    """Aggregate all hazards to GADM boundaries."""
    for code in ALL_HAZARD_CODES:
        try:
            aggregate_hazard(code, gadm_levels=gadm_levels)
        except Exception as exc:
            logger.error(f"[{code}] Aggregation failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Combine — merge per-hazard CSVs into final wide-format outputs
# ---------------------------------------------------------------------------

def combine_all(
    gadm_levels: List[int] = None,
    hazard_codes: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    scales: Optional[List[str]] = None,
) -> None:
    """
    Read all per-hazard aggregation CSVs and write combined CSVs, one subfolder
    per scoring scale: combined/five/ and combined/ten/.
    """
    combine_scores(
        gadm_levels=gadm_levels,
        hazard_codes=hazard_codes,
        scenarios=scenarios,
        scales=scales,
    )


# ---------------------------------------------------------------------------
# Full pipeline — Option A: Dask delayed (no extra dependencies)
# ---------------------------------------------------------------------------

def run_pipeline(scales: List[str] = None, gadm_levels: List[int] = None) -> None:
    """
    Run the full pipeline: score all hazards, then aggregate all.

    Uses Dask delayed so all scoring jobs run in parallel on the Coiled cluster,
    and all aggregation jobs run in parallel once scoring is done.

    Expects a Dask client to already be connected (e.g. via compute.get_or_create_cluster).
    Falls back to local threading if no cluster is connected.

    Args:
        scales: scoring scales to compute, e.g. ["5", "10"]. Defaults to both.
        gadm_levels: GADM levels to aggregate, e.g. [1], [2], or [1, 2]. Defaults to [1].
    """
    import dask

    if scales is None:
        scales = ["5", "10"]
    if gadm_levels is None:
        gadm_levels = [1]

    logger.info("=== Pipeline start: scoring ===")
    score_tasks = [
        dask.delayed(score_hazard)(code, scales) for code in ALL_HAZARD_CODES
    ]
    dask.compute(*score_tasks)

    logger.info("=== Pipeline start: aggregation ===")
    agg_tasks = [
        dask.delayed(aggregate_hazard)(code, gadm_levels) for code in ALL_HAZARD_CODES
    ]
    dask.compute(*agg_tasks)

    logger.info("=== Pipeline complete ===")
