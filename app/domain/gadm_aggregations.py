"""
app/domain/gadm_aggregations.py
GADM zonal statistics using xvec + exactextract engine.

Scaling pattern:
  - aggregate_gadm() dispatches one dask.delayed task per country
  - Each task receives only a GCS path string (not the full dask graph)
  - Each worker opens its own Zarr connection + loads its own GADM slice
  - No worker ever holds the full global dataset or full GADM in memory
"""

import itertools
import logging
from pathlib import Path
from typing import Optional

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def _pipeline_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_gadm(level: int, iso3: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load GADM GeoDataFrame from GeoParquet.

    Args:
        level: 0, 1, or 2
        iso3: When provided for level 1 or 2, loads only that country's
              partition — each worker loads only its few MB slice.
              When omitted for level 0/1, loads the full file (used by
              aggregate_gadm() on the driver to obtain the country list).
    """
    cfg = _pipeline_cfg()["gadm"]["parquet"]

    if level == 0:
        gdf = gpd.read_parquet(cfg["adm0"])
        if iso3 is not None:
            return gdf[gdf[GID_0] == iso3]
        return gdf
    elif level == 1:
        if iso3 is not None:
            # Per-country partition: only this country's provinces (~5-50 rows)
            return gpd.read_parquet(f"{cfg['adm1_base']}/{iso3}.parquet")
        # Full file: used only to get the country list on the driver
        return gpd.read_parquet(cfg["adm1"])
    elif level == 2:
        if iso3 is None:
            raise ValueError("iso3 must be provided for adm2 loading")
        return gpd.read_parquet(f"{cfg['adm2_base']}/{iso3}.parquet")
    else:
        raise ValueError(f"Unsupported GADM level: {level}")


# GADM column name constants (confirmed from parquet schemas)
GID_0 = "GID_0"   # country code  — present in both adm0 and adm1
GID_1 = "GID_1"   # province ID   — present in adm1 only


def _apply_rp_weights(score_da: xr.DataArray, rp_weights: dict) -> xr.DataArray:
    """
    Map each integer score value to its representative RP weight.

    rp_weights keys are ints (1–5 for 5-point, 1–10 for 10-point).
    Returns a float DataArray of the same shape; NaN pixels remain NaN.
    """
    weighted = xr.full_like(score_da, fill_value=float("nan"), dtype=float)
    for score_val, weight in rp_weights.items():
        weighted = xr.where(score_da == int(score_val), weight, weighted)
    return weighted


def _scoring_scales(ds: xr.Dataset) -> list:
    """Return list of scoring scale strings present in ds, or [None] if no scoring dim."""
    if "scoring" in ds["score"].dims:
        return ds["score"].coords["scoring"].values.tolist()
    return [None]


def _split_hundred_scale(ds: xr.Dataset):
    """
    Split a scored Dataset into (threshold_ds, hundred_ds).

    CF/RF custom aggregation (RP-weighting) only applies to discrete integer
    scales (5, 10). The 0–100 min-max scale uses plain zonal stats instead.

    Returns (threshold_ds, hundred_ds) — either may be None if absent.
    """
    if "scoring" not in ds.dims:
        return ds, None
    scales = [str(s) for s in ds.scoring.values]
    threshold_scales = [s for s in scales if s != "100"]
    has_hundred = "100" in scales
    threshold_ds = ds.sel(scoring=threshold_scales) if threshold_scales else None
    hundred_ds = ds.sel(scoring=["100"]) if has_hundred else None
    return threshold_ds, hundred_ds


def _rescore_by_quantiles(
    series: pd.Series, q20: float, q40: float, q60: float, q80: float
) -> pd.Series:
    """
    Bin float weighted_risk values into 1–5 using four quantile thresholds.

    Values <= 0 or NaN → 0 (no risk / no data).
    """
    conditions = [
        series <= 0,
        series <= q20,
        series <= q40,
        series <= q60,
        series <= q80,
    ]
    choices = [0, 1, 2, 3, 4]
    result = np.select(conditions, choices, default=5)
    result = pd.Series(result, index=series.index, dtype=int)
    result[series.isna()] = 0
    return result


def _zonal_stats_dataset(
    gadm: gpd.GeoDataFrame,
    ds: xr.Dataset,
    stats: list,
    lat_dim: str,
    lon_dim: str,
) -> xr.Dataset:
    """Run xvec zonal_stats on a Dataset and compute."""
    import rioxarray  # noqa: F401
    import xvec  # noqa: F401

    rename_map = {lat_dim: "y", lon_dim: "x"}
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.dims})
    ds = ds.rio.write_crs("EPSG:4326")
    result = ds.xvec.zonal_stats(
        gadm.geometry,
        x_coords="x",
        y_coords="y",
        stats=stats,
        all_touched=True,
    ).compute()
    # Normalize dimension name across xvec versions ("zonal_statistics" → "stats")
    if "zonal_statistics" in result.dims:
        result = result.rename({"zonal_statistics": "stats"})
    return result


def _load_coastline_ds(ds: xr.Dataset) -> xr.DataArray:
    """Load and reindex the coastline raster to match ds spatial grid."""
    import rioxarray  # noqa: F401

    cfg = _pipeline_cfg()["gadm"]
    coastline = xr.open_dataset(cfg["coastline"], engine="rasterio").squeeze()
    lat_dim = "lat" if "lat" in ds.coords else "y"
    lon_dim = "lon" if "lon" in ds.coords else "x"
    return coastline.rename({"x": lon_dim, "y": lat_dim}).reindex_like(ds, method="nearest")


def _apply_coastline_mask(ds: xr.Dataset, hazard_code: str) -> xr.Dataset:
    """Apply coastline mask for CF and RF — sets ocean pixels to NaN."""
    if hazard_code not in ("CF", "RF"):
        return ds
    coastline_reindexed = _load_coastline_ds(ds)
    return ds.where(coastline_reindexed > 0)


def _custom_stats_for_scale(
    gadm: gpd.GeoDataFrame,
    score_da: xr.DataArray,
    rp_weights: dict,
    bins: dict,
    gadm_id_col: str,
    lat_dim: str,
    lon_dim: str,
    custom_col: str,
    denominator_da: xr.DataArray = None,
) -> pd.DataFrame:
    """
    Core RP-weighted aggregation for one scoring scale.

    denominator_da: if provided, use its sum as denominator (CF coastal pixels);
                    otherwise use total valid-pixel count (RF).
    bins must be a dict with q20/q40/q60/q80 — all non-None.
    """
    weighted_da = _apply_rp_weights(score_da, rp_weights)

    if denominator_da is None:
        denom_da = score_da.notnull().astype(float)
    else:
        denom_da = denominator_da

    # Single combined xvec pass — one exactextract call for all needed stats
    combined = xr.Dataset({"w": weighted_da, "n": denom_da, "score": score_da})
    result = _zonal_stats_dataset(gadm, combined, ["sum", "mean", "max"], lat_dim, lon_dim)

    w_sum = np.asarray(result["w"].sel(stats="sum").values, dtype=np.float64)
    n_sum = np.asarray(result["n"].sel(stats="sum").values, dtype=np.float64)
    # Double np.where avoids ZeroDivisionError: denominator is replaced with 1.0
    # where n==0 so division never sees a zero; outer np.where maps those to nan.
    weighted_risk = pd.Series(np.where(n_sum > 0, w_sum / np.where(n_sum > 0, n_sum, 1.0), np.nan))

    # Max-rule: cap weighted_risk at the polygon's max-score RP weight
    max_scores = result["score"].sel(stats="max").values.astype(np.float32)
    max_weights = np.array(
        [rp_weights.get(int(v), np.nan) if not np.isnan(v) else np.nan for v in max_scores]
    )
    weighted_risk = pd.Series(np.minimum(weighted_risk.values, max_weights))

    final_score = _rescore_by_quantiles(weighted_risk, **bins)

    # Build wide df from score mean/max (drop the sum stat)
    result = result.assign_coords(
        gid0=("geometry", gadm[GID_0].values),
        gadm_id=("geometry", gadm[gadm_id_col].values),
    )
    df = result[["score"]].to_dataframe().reset_index().drop(columns=["geometry"], errors="ignore")
    df = df[df["stats"].isin(["mean", "max"])]
    df_wide = df.pivot_table(index="gadm_id", columns="stats", values="score").reset_index()
    df_wide.columns.name = None

    custom_df = pd.DataFrame({
        "gadm_id": gadm[gadm_id_col].values,
        "gid0": gadm[GID_0].values,
        custom_col: final_score.values,
    })
    return custom_df.merge(df_wide, on="gadm_id", how="left")


def _run_rf_custom_stats(
    gadm: gpd.GeoDataFrame,
    ds: xr.Dataset,
    gadm_id_col: str,
    gadm_level: int,
) -> pd.DataFrame:
    """
    RF custom aggregation: RP-weighted pixel count / total pixels → re-scored.
    Loops over all scoring scales present in ds (e.g. '5' and '10').
    """
    cfg = _pipeline_cfg()["custom_aggregation"]
    all_rp_weights = cfg["rp_weights"]
    level_key = f"adm{gadm_level}"

    lat_dim = "lat" if "lat" in ds.coords else "y"
    lon_dim = "lon" if "lon" in ds.coords else "x"

    per_scale = []
    for scale in _scoring_scales(ds):
        scale_key = str(scale)
        score_da = ds["score"].sel(scoring=scale) if scale is not None else ds["score"]
        rp_weights = {int(k): v for k, v in all_rp_weights[scale_key].items()}
        bins = cfg["RF"][level_key].get(scale_key, {})
        if any(bins.get(k) is None for k in ("q20", "q40", "q60", "q80")):
            logger.warning(f"  [RF] Skipping scale={scale_key}: quantile bins not calibrated")
            continue

        extra_dims = [d for d in score_da.dims if d not in (lat_dim, lon_dim)]
        dim_values = [score_da[d].values.tolist() for d in extra_dims]
        for combo_vals in (itertools.product(*dim_values) if extra_dims else [()]):
            sel_dict = dict(zip(extra_dims, combo_vals))
            slice_da = score_da.sel(sel_dict) if sel_dict else score_da
            df = _custom_stats_for_scale(
                gadm, slice_da, rp_weights, bins,
                gadm_id_col, lat_dim, lon_dim,
                custom_col="score_rf_custom",
            )
            for d, v in sel_dict.items():
                df[d] = str(v)
            df["scoring"] = scale_key
            per_scale.append(df)

    if not per_scale:
        logger.warning("  [RF] No calibrated scales found — returning empty DataFrame")
        return pd.DataFrame()
    result = pd.concat(per_scale, ignore_index=True)
    result["hazard"] = "RF"
    result["gadm_level"] = gadm_level
    return result


def _run_cf_custom_stats(
    gadm: gpd.GeoDataFrame,
    ds: xr.Dataset,
    coastline_da: xr.DataArray,
    gadm_id_col: str,
    gadm_level: int,
) -> pd.DataFrame:
    """
    CF custom aggregation: RP-weighted pixel count / coastal pixel count.
    Landlocked polygons (coast_sum == 0) receive score 0.
    Loops over all scoring scales present in ds.
    """
    cfg = _pipeline_cfg()["custom_aggregation"]
    all_rp_weights = cfg["rp_weights"]
    level_key = f"adm{gadm_level}"

    lat_dim = "lat" if "lat" in ds.coords else "y"
    lon_dim = "lon" if "lon" in ds.coords else "x"

    # Coastal-pixel denominator (scale-independent — same mask for both scales)
    if isinstance(coastline_da, xr.Dataset):
        coast_arr = coastline_da[list(coastline_da.data_vars)[0]]
    else:
        coast_arr = coastline_da
    coastal_da = (coast_arr > 0).astype(float)

    per_scale = []
    for scale in _scoring_scales(ds):
        scale_key = str(scale)
        score_da = ds["score"].sel(scoring=scale) if scale is not None else ds["score"]
        rp_weights = {int(k): v for k, v in all_rp_weights[scale_key].items()}
        bins = cfg["CF"][level_key].get(scale_key, {})
        if any(bins.get(k) is None for k in ("q20", "q40", "q60", "q80")):
            logger.warning(f"  [CF] Skipping scale={scale_key}: quantile bins not calibrated")
            continue

        extra_dims = [d for d in score_da.dims if d not in (lat_dim, lon_dim)]
        dim_values = [score_da[d].values.tolist() for d in extra_dims]
        for combo_vals in (itertools.product(*dim_values) if extra_dims else [()]):
            sel_dict = dict(zip(extra_dims, combo_vals))
            slice_da = score_da.sel(sel_dict) if sel_dict else score_da
            df = _custom_stats_for_scale(
                gadm, slice_da, rp_weights, bins,
                gadm_id_col, lat_dim, lon_dim,
                custom_col="score_cf_custom",
                denominator_da=coastal_da,
            )
            for d, v in sel_dict.items():
                df[d] = str(v)
            df["scoring"] = scale_key
            per_scale.append(df)

    if not per_scale:
        logger.warning("  [CF] No calibrated scales found — returning empty DataFrame")
        return pd.DataFrame()
    result = pd.concat(per_scale, ignore_index=True)
    result["hazard"] = "CF"
    result["gadm_level"] = gadm_level
    return result


def _run_zonal_stats(
    gadm: gpd.GeoDataFrame,
    ds: xr.Dataset,
    hazard_code: str,
    gadm_level: int,
) -> pd.DataFrame:
    """
    Run xvec zonal stats for one country slice and return a tidy DataFrame.

    xvec computes mean/max for every combination of non-spatial dimensions
    (e.g. scoring, scenario, time) present in the scored Zarr — each
    combination becomes one row in the output.

    Output is wide: 'score_mean' and 'score_max' as separate columns rather
    than a long 'stats' dimension, so the row count is:
      n_polygons × n_scenarios × n_scoring_scales × n_time_periods × ...
    """
    import xvec  # noqa: F401

    lat_dim = "lat" if "lat" in ds.coords else "y"
    lon_dim = "lon" if "lon" in ds.coords else "x"
    gadm_id_col = GID_1 if gadm_level == 1 else GID_0

    import rioxarray  # noqa: F401

    # rioxarray requires dims named 'x'/'y'; rename if needed
    rename_map = {lat_dim: "y", lon_dim: "x"}
    ds_rio = ds.rename({k: v for k, v in rename_map.items() if k in ds.dims})
    ds_rio = ds_rio.rio.write_crs("EPSG:4326")

    try:
        result = ds_rio.xvec.zonal_stats(
            gadm.geometry,
            x_coords="x",
            y_coords="y",
            stats=["mean", "max", "stdev"],
            all_touched=True,
        ).compute()
    except ValueError:
        # exactextract reshape error: no raster pixels intersect any polygon
        return pd.DataFrame()

    result = result.assign_coords(
        gid0=("geometry", gadm[GID_0].values),
        gadm_id=("geometry", gadm[gadm_id_col].values),
    )

    # to_dataframe produces long format with a 'stats' dim column.
    # Pivot mean/max/stdev into separate columns for a cleaner wide output.
    df = result.to_dataframe().reset_index().drop(columns=["geometry"], errors="ignore")

    # xvec names the stats dimension 'stats' or 'zonal_statistics' depending on version
    stats_col = next((c for c in ("stats", "zonal_statistics") if c in df.columns), None)
    if stats_col:
        id_cols = [c for c in df.columns if c not in (stats_col, "score")]
        df = df.pivot_table(index=id_cols, columns=stats_col, values="score").reset_index()
        df.columns.name = None
        df = df.rename(columns={"mean": "score_mean", "max": "score_max", "stdev": "score_stdev"})

    df["hazard"] = hazard_code
    df["gadm_level"] = gadm_level

    return df


def _aggregate_partition(
    gid0: str,
    scored_zarr_path: str,
    hazard_code: str,
    gadm_level: int,
) -> pd.DataFrame:
    """
    Aggregate scored data for a single country.

    Opens the Zarr store on the worker (avoids pickling the full dask task
    graph across the network). Each worker only reads the Zarr chunks that
    overlap with its country's bounding box.

    Args:
        gid0: GADM country code (GID_0), e.g. 'NGA', 'AFG'
        scored_zarr_path: GCS path to the scored Zarr store
        hazard_code: e.g. 'HS', 'LS'
        gadm_level: 0 or 1
    """
    ds = xr.open_zarr(scored_zarr_path)

    # Deduplicate spatial coords — float precision in source zarrs can produce
    # near-duplicate lat/lon values that cause InvalidIndexError in exactextract
    lat_dim = "lat" if "lat" in ds.coords else "y"
    lon_dim = "lon" if "lon" in ds.coords else "x"
    for dim in (lat_dim, lon_dim):
        if dim in ds.coords and ds[dim].size > 0:
            _, idx = np.unique(ds[dim].values, return_index=True)
            if len(idx) < ds[dim].size:
                ds = ds.isel({dim: np.sort(idx)})

    # Keep only the 'mean' statistic slice — combine.py uses only mean anyway,
    # so aggregating max/min/median/std through xvec is wasted compute.
    # Falls back to the first available value if 'mean' is not present.
    if "statistic" in ds.dims:
        stat_vals = [str(v) for v in ds.statistic.values]
        preferred = next((v for v in ds.statistic.values if str(v).lower() == "mean"), ds.statistic.values[0])
        ds = ds.sel(statistic=[preferred])

    # Squeeze size-1 auxiliary dims (e.g. model='ENSEMBLE') — but preserve semantic
    # dims (scenario, time, scoring) even when size=1 so they appear as columns in
    # the output DataFrame (e.g. SUB has scenario=['RCP85'] — size 1 but meaningful).
    _KEEP_DIMS = {lat_dim, lon_dim, "scenario", "time", "scoring"}
    for dim in list(ds.dims):
        if dim not in _KEEP_DIMS and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=False)

    # Load GADM for this country
    gadm = load_gadm(gadm_level, iso3=gid0)
    if gadm.empty:
        return pd.DataFrame()

    gadm_id_col = GID_1 if gadm_level == 1 else GID_0

    # Sort zarr coords once (lazy — no data loaded yet)
    ds = ds.sortby([lat_dim, lon_dim])

    # Open coastline TIF once per country (not once per province) for CF
    coastline_opened = None
    if hazard_code == "CF":
        import rioxarray  # noqa: F401
        cfg = _pipeline_cfg()["gadm"]
        coastline_opened = xr.open_dataset(cfg["coastline"], engine="rasterio").squeeze()

    # Process each province individually to bound peak memory per worker.
    # Large countries (RUS, CAN, USA) would OOM if loaded as a single bbox;
    # loading one province at a time keeps memory proportional to the largest
    # single province rather than the full country extent.
    results = []
    for i in range(len(gadm)):
        province = gadm.iloc[[i]].reset_index(drop=True)
        minx, miny, maxx, maxy = province.total_bounds
        pad = 0.5  # degrees buffer to avoid clipping edge pixels
        ds_slice = ds.sel(
            {lat_dim: slice(miny - pad, maxy + pad),
             lon_dim: slice(minx - pad, maxx + pad)}
        )

        if ds_slice.sizes.get(lat_dim, 0) == 0 or ds_slice.sizes.get(lon_dim, 0) == 0:
            continue

        with dask.config.set(scheduler="threads"):
            ds_loaded = ds_slice.compute()

        # Cast to float16 — scores are integers 1–10, so float16 is lossless
        # and cuts peak memory 4× vs float64 (the typical stored dtype)
        ds_loaded = ds_loaded.astype(np.float16)

        # Decode byte-string coords (from |S zarr encoding) to plain strings
        decode = {
            c: ds_loaded[c].values.astype(str)
            for c in ds_loaded.coords
            if ds_loaded[c].dtype.kind == "S"
        }
        if decode:
            ds_loaded = ds_loaded.assign_coords(decode)

        if hazard_code == "RF":
            # RP-weighting only applies to discrete scales (5, 10).
            # Scale "100" uses plain zonal stats.
            thresh_ds, hundred_ds = _split_hundred_scale(ds_loaded)
            part_dfs = []
            if thresh_ds is not None:
                part_dfs.append(_run_rf_custom_stats(province, thresh_ds, gadm_id_col, gadm_level))
            if hundred_ds is not None:
                part_dfs.append(_run_zonal_stats(province, hundred_ds, hazard_code, gadm_level))
            df = pd.concat(part_dfs, ignore_index=True) if part_dfs else pd.DataFrame()
        elif hazard_code == "CF":
            lat_dim_loaded = "lat" if "lat" in ds_loaded.coords else "y"
            lon_dim_loaded = "lon" if "lon" in ds_loaded.coords else "x"
            coastline_aligned = coastline_opened.rename({"x": lon_dim_loaded, "y": lat_dim_loaded})
            coastline_da = coastline_aligned.reindex_like(ds_loaded, method="nearest")
            thresh_ds, hundred_ds = _split_hundred_scale(ds_loaded)
            part_dfs = []
            if thresh_ds is not None:
                part_dfs.append(_run_cf_custom_stats(province, thresh_ds, coastline_da, gadm_id_col, gadm_level))
            if hundred_ds is not None:
                part_dfs.append(_run_zonal_stats(province, hundred_ds, hazard_code, gadm_level))
            df = pd.concat(part_dfs, ignore_index=True) if part_dfs else pd.DataFrame()
        else:
            ds_loaded = _apply_coastline_mask(ds_loaded, hazard_code)
            df = _run_zonal_stats(province, ds_loaded, hazard_code, gadm_level)

        if df is not None and not df.empty:
            results.append(df)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def aggregate_gadm( # change to aggregate province
    scored_zarr_path: str,
    hazard_code: str,
    gadm_level: int = 1,
) -> pd.DataFrame:
    """
    Aggregate a scored Zarr to province boundaries via per-country Dask tasks.

    Args:
        scored_zarr_path: GCS path to the scored Zarr (passed as a string so
                          workers open their own connection rather than
                          receiving a pickled dask graph)
        hazard_code: e.g. 'HS'
        gadm_level: 0 or 1

    Returns:
        Concatenated DataFrame of all countries
    """
    SKIP_COUNTRIES = {"ATA"}  # Antarctica — no provinces of interest, huge bbox

    if gadm_level == 0:
        # For adm0, read only the GID_0 column to get the country list —
        # avoids loading geometries (several 100 MB) just for the index.
        cfg = _pipeline_cfg()
        adm0_path = cfg["gadm"]["parquet"]["adm0"]
        adm0_ids = pd.read_parquet(adm0_path, columns=[GID_0])
        countries = [g for g in adm0_ids[GID_0].unique() if g not in SKIP_COUNTRIES]
    else:
        # For adm1+, derive country list from per-country partition files
        cfg = _pipeline_cfg()
        adm1_base = cfg["gadm"]["parquet"]["adm1_base"]
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        files = fs.ls(adm1_base)
        countries = [
            f.split("/")[-1].replace(".parquet", "")
            for f in files if f.endswith(".parquet")
            and f.split("/")[-1].replace(".parquet", "") not in SKIP_COUNTRIES
        ]

    logger.info(
        f"[{hazard_code}] Dispatching zonal stats across "
        f"{len(countries)} countries (adm{gadm_level})"
    )

    delayed_tasks = [
        dask.delayed(_aggregate_partition)(gid0, scored_zarr_path, hazard_code, gadm_level)
        for gid0 in countries
    ]
    results = list(dask.compute(*delayed_tasks))
    non_empty = [df for df in results if df is not None and not df.empty]

    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def write_csv(df: pd.DataFrame, output_path: str):
    """Write aggregated DataFrame to GCS CSV via fsspec."""
    import fsspec

    logger.info(f"Writing CSV to {output_path} ({len(df)} rows)")
    with fsspec.open(output_path, "w") as f:
        df.to_csv(f, index=False)
    logger.info("CSV write complete")
