"""
app/domain/scoring.py
Data-independent scoring functions that take a DataArray and return scored Datasets.
"""

import logging

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def score_zarr(
    data: xr.DataArray,
    thresholds: List[float],
    scores: List[int],
    metric: Optional[str] = None
) -> xr.Dataset:
    """
    Score a DataArray using provided thresholds and scores.
    
    Args:
        data: Input xarray DataArray to score
        thresholds: List of threshold boundaries
        scores: List of scores corresponding to bins
        metric: New name for the metric coordinate in the output dataset
               (e.g., 'ari_1_to_5', 'landslide_proxy_final')
    
    Returns:
        xarray Dataset containing only the scored variable (named 'score') 
        with renamed metric coordinate
    
    Example:
        scored = score_array(
            ds['ari'],
            [0,5,10,15,20,inf],
            [1,2,3,4,5,6],
            metric='ari_1_to_5'  # This renames the metric coordinate
        )
        # Returns Dataset with:
        # - Data variable: 'score'
        # - Coordinate: 'metric' with value 'ari_1_to_5'
    """
    logger.debug(f"Scoring array with shape {data.shape}, thresholds={thresholds}, scores={scores}")
    
    # Score the data
    scored_data = _score_array(data, thresholds, scores)
    
    # Drop metric dim/coord entirely — stored in attrs instead to avoid
    # "dimension already exists as scalar variable" in to_dataset()
    coords_to_drop = [
        k for k in scored_data.coords
        if k == 'metric' or 'metric' in scored_data.coords[k].dims
    ]
    if coords_to_drop:
        scored_data = scored_data.drop_vars(coords_to_drop)
    if 'metric' in scored_data.dims and scored_data.sizes['metric'] == 1:
        scored_data = scored_data.squeeze('metric', drop=True)
    
    # Always name the data variable 'score'
    scored_data = scored_data.rename('score')
    
    # Add metadata
    scored_data.attrs.update({
        'scoring_method': 'threshold_based',
        'thresholds': str(thresholds),
        'scores': str(scores),
        'source_variable': data.name,
        'source_dims': str(list(data.dims))
    })
    
    # Return as Dataset
    return scored_data.to_dataset()


def _score_chunk(x: np.ndarray, thresholds: list, scores: list) -> np.ndarray:
    """Score a numpy chunk using np.digitize — called once per chunk, not per element."""
    t = np.array(thresholds, dtype=float)
    s = np.array(scores, dtype=float)
    if t[0] > t[-1]:          # descending: reverse both arrays
        t = t[::-1]
        s = s[::-1]
    idx = np.clip(np.digitize(x, t) - 1, 0, len(s) - 1)
    result = s[idx].astype(np.float16)
    result[np.isnan(x)] = np.nan
    result[np.isposinf(x)] = s[-1]
    result[np.isneginf(x)] = s[0]
    return result


def _score_array(
    data: xr.DataArray,
    thresholds: List[float],
    scores: List[int]
) -> xr.DataArray:
    """Internal function to score an array — chunk-level via np.digitize."""
    return xr.apply_ufunc(
        _score_chunk,
        data,
        kwargs={"thresholds": thresholds, "scores": scores},
        dask="parallelized",
        output_dtypes=[np.float16],
    )


def score_zarr_multi(
    data: xr.DataArray,
    scales: List[str],
    thresholds_per_scale: Dict[str, Tuple[List[float], List[int]]],
    metric: Optional[str] = None,
) -> xr.Dataset:
    """
    Score a DataArray for multiple scales, returning a Dataset with a 'scoring' dimension.

    Args:
        data: Input xarray DataArray to score
        scales: List of scale identifiers, e.g. ['5', '10']
        thresholds_per_scale: Dict mapping scale -> (thresholds, scores)
        metric: Metric coordinate value to attach

    Returns:
        xr.Dataset with variable 'score' and coordinate dim 'scoring'
    """
    results = []
    for scale in scales:
        thresholds, scores = thresholds_per_scale[scale]
        scored = score_zarr(data, thresholds, scores, metric=metric)
        scored = scored.expand_dims("scoring").assign_coords(scoring=[scale])
        results.append(scored)
    return xr.concat(results, dim="scoring")


def _select_reference_slice(
    data: xr.DataArray,
    ref_scenario: Optional[str] = None,
    ref_time: Optional[str] = None,
) -> xr.DataArray:
    """
    Select the reference slice of a DataArray for min-max calibration.

    If ref_scenario / ref_time are provided they are used directly (with
    fallback to last available if not found).  Otherwise defaults to
    RCP85 / Lt, falling back to RCP45, then last available.
    """
    result = data
    if "scenario" in result.dims:
        scenarios_lower = {str(s).lower(): s for s in result.scenario.values}
        if ref_scenario is not None:
            key = ref_scenario.lower()
            if key in scenarios_lower:
                result = result.sel(scenario=scenarios_lower[key])
            else:
                result = result.isel(scenario=-1)
        elif "rcp85" in scenarios_lower:
            result = result.sel(scenario=scenarios_lower["rcp85"])
        elif "rcp45" in scenarios_lower:
            result = result.sel(scenario=scenarios_lower["rcp45"])
        else:
            result = result.isel(scenario=-1)
    if "time" in result.dims:
        times = [str(t) for t in result.time.values]
        if ref_time is not None:
            if ref_time in times:
                result = result.sel(time=ref_time)
            else:
                result = result.isel(time=-1)
        elif "Lt" in times:
            result = result.sel(time="Lt")
        else:
            result = result.isel(time=-1)
    return result


def score_zarr_minmax(
    data: xr.DataArray,
    ref_scenario: Optional[str] = None,
    ref_time: Optional[str] = None,
) -> xr.Dataset:
    """
    Normalize data to 0–100 using global min/max from the reference slice.
    Output is float16, clipped to [0, 100].

    ref_scenario / ref_time override the default RCP85/Lt selection — use for
    hazards whose data only covers a subset of scenarios or time periods
    (e.g. TC/TS: RCP85/St).

    Min/max are computed via a single eager pass over the reference slice only —
    the full array is then normalized lazily.
    """
    # min from RCP45/Cc (mildest conditions); max from RCP85/Lt (most severe).
    # ref_scenario/ref_time override both anchors for hazards with limited
    # scenario/time coverage (e.g. TC/TS: RCP85/St).
    min_ref = _select_reference_slice(
        data,
        ref_scenario=ref_scenario or "rcp45",
        ref_time=ref_time or "Cc",
    )
    max_ref = _select_reference_slice(
        data,
        ref_scenario=ref_scenario or "rcp85",
        ref_time=ref_time or "Lt",
    )
    min_val = float(min_ref.min().compute())
    if np.isnan(min_val):
        # RCP45/Cc is empty for hazards where current climate lives under
        # HISTORICAL (e.g. RF, CF). Fall back to HISTORICAL/Cc.
        hist_ref = _select_reference_slice(data, ref_scenario="historical", ref_time=ref_time or "Cc")
        min_val = float(hist_ref.min().compute())

    max_val = float(max_ref.max().compute())

    if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
        normalized = xr.zeros_like(data, dtype=np.float16)
    else:
        normalized = (
            ((data - min_val) / (max_val - min_val) * 100)
            .clip(0, 100)
            .astype(np.float16)
        )

    min_label = f"{ref_scenario or 'rcp45/historical'}_{ref_time or 'Cc'} (or nearest available)"
    max_label = f"{ref_scenario or 'rcp85'}_{ref_time or 'Lt'} (or nearest available)"
    normalized = normalized.rename("score")
    normalized.attrs.update({
        "scoring_method": "minmax_normalization",
        "min_reference": min_label,
        "max_reference": max_label,
        "min_val": min_val,
        "max_val": max_val,
    })
    return normalized.to_dataset()


def score_value(
    value: float,
    thresholds: List[float],
    scores: List[int]
) -> int:
    """Score a single value using thresholds."""
    if pd.isna(value) or np.isnan(value):
        return np.nan
    
    is_descending = thresholds[0] > thresholds[-1]
    
    if is_descending:
        thresholds_asc = list(reversed(thresholds))
        scores_asc = list(reversed(scores))
        
        if np.isinf(value):
            return scores_asc[-1] if value > 0 else scores_asc[0]
        
        bin_idx = np.digitize(value, thresholds_asc) - 1
        bin_idx = max(0, min(bin_idx, len(scores_asc) - 1))
        return scores_asc[bin_idx]
    else:
        if np.isinf(value):
            return scores[-1] if value > 0 else scores[0]
        
        bin_idx = np.digitize(value, thresholds) - 1
        bin_idx = max(0, min(bin_idx, len(scores) - 1))
        return scores[bin_idx]