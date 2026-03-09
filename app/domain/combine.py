"""
app/domain/combine.py
Merge per-hazard aggregation CSVs into final wide-format output CSVs.

Each per-hazard CSV (e.g. HS_adm1.csv) contains rows for every combination of
scoring scale, scenario, time period (and statistic if present in the Zarr).
This module reads all 14 CSVs and produces one folder per scoring scale:

    combined/
      five/                                    # 5-point scoring
        hazard_score_province_mean_45.csv
        hazard_score_province_max_45.csv
        hazard_score_province_std_45.csv
        hazard_score_province_mean_85.csv
        hazard_score_province_max_85.csv
        hazard_score_province_std_85.csv
        hazard_score_country_mean_45.csv       (if adm0 requested)
        ...
      ten/                                     # 10-point scoring
        hazard_score_province_mean_45.csv
        ...

Output columns:
  Province (adm1): GID_0, COUNTRY, GID_1, NAME_1, ENGTYPE_1, VARNAME_1,
                   {HAZARD}_{TIME}, ...
  Country  (adm0): GID_0, COUNTRY, {HAZARD}_{TIME}, ...

Missing values are filled with -9999.
Column ordering: alphabetical by hazard code, then Cc/St/Mt/Lt.
"""

import logging
from pathlib import Path
from typing import List, Optional

import fsspec
import geopandas as gpd
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline.yaml"

# Hazard codes in alphabetical order (matches sample files)
ALL_HAZARD_CODES = [
    "CER", "CF", "CS", "DR", "ER", "HP", "HS",
    "LS", "RF", "SUB", "TC", "TS", "WF", "WS",
]

# Time period labels in the standard order shown in samples
TIME_PERIOD_ORDER = ["Cc", "St", "Mt", "Lt"]

# GADM metadata columns
ADM1_META_COLS = ["GID_0", "COUNTRY", "GID_1", "NAME_1", "ENGTYPE_1", "VARNAME_1"]
ADM0_META_COLS = ["GID_0", "COUNTRY"]

# Human-readable folder names for scoring scales
SCALE_FOLDER = {"5": "five", "10": "ten", "100": "hundred"}


def _pipeline_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _csv_input_path(hazard_code: str, gadm_level: int) -> str:
    cfg = _pipeline_cfg()
    base = cfg["gcs"]["csv_output"]
    return f"{base}/{hazard_code}_adm{gadm_level}.csv"


def _combined_output_path(scale: str, stat: str, scenario: str, gadm_level: int) -> str:
    """
    e.g. scale='5', stat='mean', scenario='RCP45', gadm_level=1
    → gs://.../combined/five/hazard_score_province_mean_45.csv
    """
    cfg = _pipeline_cfg()
    base = cfg["gcs"].get("combined_output", cfg["gcs"]["csv_output"] + "/combined")
    folder = SCALE_FOLDER.get(scale, f"scale{scale}")
    level_label = "province" if gadm_level == 1 else "country"
    scenario_suffix = scenario.upper().replace("RCP", "")
    return f"{base}/{folder}/hazard_score_{level_label}_{stat}_{scenario_suffix}.csv"


def _load_gadm_meta(gadm_level: int) -> pd.DataFrame:
    """Load GADM metadata (non-geometry columns) for joining."""
    cfg = _pipeline_cfg()["gadm"]["parquet"]
    if gadm_level == 1:
        gdf = gpd.read_parquet(cfg["adm1"])
        cols = [c for c in ADM1_META_COLS if c in gdf.columns]
        return pd.DataFrame(gdf[cols])
    else:
        gdf = gpd.read_parquet(cfg["adm0"])
        cols = [c for c in ADM0_META_COLS if c in gdf.columns]
        return pd.DataFrame(gdf[cols])


def _load_hazard_csv(hazard_code: str, gadm_level: int) -> Optional[pd.DataFrame]:
    """Load one per-hazard aggregation CSV. Returns None if not found."""
    path = _csv_input_path(hazard_code, gadm_level)
    try:
        with fsspec.open(path, "r") as f:
            df = pd.read_csv(f)
        logger.info(f"[{hazard_code}] Loaded {len(df)} rows from {path}")
        return df
    except FileNotFoundError:
        logger.warning(f"[{hazard_code}] CSV not found: {path} — skipping")
        return None
    except Exception as exc:
        logger.error(f"[{hazard_code}] Failed to load CSV: {exc}")
        return None


def _extract_wide(
    df: pd.DataFrame,
    hazard_code: str,
    scenario: str,
    stat_col: str,
    scoring_scale: str,
) -> Optional[pd.DataFrame]:
    """
    Filter a per-hazard DataFrame to one scale/scenario/stat and pivot time
    into wide columns: {HAZARD}_{time_period}.

    Strict scale filter — returns None if the requested scale is absent.
    """
    gadm_id_col = "gadm_id"
    if gadm_id_col not in df.columns:
        logger.warning(f"[{hazard_code}] 'gadm_id' column missing — skipping")
        return None

    if stat_col not in df.columns:
        logger.warning(f"[{hazard_code}] '{stat_col}' column missing — skipping")
        return None

    # Strict scale filter — cast to str to handle int-typed scoring column from CSV
    if "scoring" in df.columns:
        scoring_str = df["scoring"].astype(str)
        if scoring_scale not in scoring_str.unique():
            logger.debug(
                f"[{hazard_code}] scale '{scoring_scale}' not in CSV "
                f"(available: {scoring_str.unique().tolist()}) — skipping for this folder"
            )
            return None
        mask = scoring_str == scoring_scale
    else:
        mask = pd.Series(True, index=df.index)

    if "scenario" in df.columns:
        mask &= df["scenario"].str.upper() == scenario.upper()

    sub = df[mask].copy()
    if sub.empty:
        logger.warning(f"[{hazard_code}] No rows for scale={scoring_scale}, scenario={scenario}")
        return None

    # If a 'statistic' dimension column is present (Zarr ensemble dimension),
    # use 'mean' or the first available value
    if "statistic" in sub.columns:
        stat_values = sub["statistic"].unique()
        preferred = next(
            (v for v in ("mean", b"mean") if v in stat_values), stat_values[0]
        )
        sub = sub[sub["statistic"] == preferred]

    if "time" not in sub.columns:
        logger.warning(f"[{hazard_code}] No 'time' column — cannot pivot")
        return None

    pivot = sub.pivot_table(
        index=gadm_id_col,
        columns="time",
        values=stat_col,
        aggfunc="first",
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={t: f"{hazard_code}_{t}" for t in pivot.columns})

    return pivot.reset_index()


def _build_combined(
    gadm_level: int,
    scenario: str,
    stat_col: str,
    scoring_scale: str,
    gadm_meta: pd.DataFrame,
    hazard_dfs: dict,
    round_to_int: bool = True,
) -> pd.DataFrame:
    """
    Build one combined wide-format DataFrame for a given scale/scenario/stat/level.

    Always reads all 14 hazard CSVs — hazards whose CSV is missing or doesn't
    contain the requested scale are filled with -9999.
    """
    id_col = "GID_1" if gadm_level == 1 else "GID_0"
    meta_cols = ADM1_META_COLS if gadm_level == 1 else ADM0_META_COLS

    parts: List[pd.DataFrame] = []
    for code in ALL_HAZARD_CODES:
        df = hazard_dfs.get(code)
        if df is None:
            continue
        wide = _extract_wide(df, code, scenario, stat_col, scoring_scale)
        if wide is not None:
            parts.append(wide)

    if not parts:
        logger.warning(
            f"No data for scale={scoring_scale}, scenario={scenario}, "
            f"stat={stat_col}, adm{gadm_level}"
        )
        return pd.DataFrame()

    # Outer-join all hazard wide frames on gadm_id
    merged = parts[0].set_index("gadm_id")
    for part in parts[1:]:
        merged = merged.join(part.set_index("gadm_id"), how="outer")
    merged = merged.reset_index().rename(columns={"gadm_id": id_col})

    # Join GADM metadata (left so every province/country row is present)
    result = gadm_meta.merge(merged, on=id_col, how="left")

    # Fill missing hazard scores with 0 (no exposure / not applicable)
    hazard_cols = [c for c in result.columns if c not in meta_cols]
    result[hazard_cols] = result[hazard_cols].fillna(0)

    # Round scores: integers for 5-pt/10-pt mean/max; floats (2 dp) for 100-pt or std
    if round_to_int and stat_col in ("score_mean", "score_max"):
        for col in hazard_cols:
            if result[col].dtype.kind == "f":
                result[col] = result[col].round().astype(int)
    else:
        dp = 2 if stat_col in ("score_mean", "score_max") else 4
        for col in hazard_cols:
            if result[col].dtype.kind == "f":
                result[col] = result[col].round(dp)

    # Fixed schema: always 14 hazards × 4 time periods, missing → 0 (no exposure)
    ordered_hazard_cols = [
        f"{code}_{tp}"
        for code in ALL_HAZARD_CODES
        for tp in TIME_PERIOD_ORDER
    ]
    for col in ordered_hazard_cols:
        if col not in result.columns:
            result[col] = 0

    present_meta = [c for c in meta_cols if c in result.columns]
    return result[present_meta + ordered_hazard_cols]


def _compute_hundred_bounds(
    rcp45_df: pd.DataFrame,
    rcp85_df: pd.DataFrame,
) -> dict:
    """
    Compute per-hazard normalization bounds for the 0–100 scale.

    lo = min province score in {HAZARD}_Cc of RCP45  (mildest conditions)
    hi = max province score in {HAZARD}_Lt of RCP85  (most severe conditions)

    Returns {hazard_code: (lo, hi)}.  Hazards where either bound cannot be
    determined (missing column or all -9999) are omitted.
    """
    bounds = {}
    for code in ALL_HAZARD_CODES:
        cc_col = f"{code}_Cc"
        lt_col = f"{code}_Lt"
        lo = hi = None
        if not rcp45_df.empty and cc_col in rcp45_df.columns:
            valid = rcp45_df.loc[rcp45_df[cc_col] != 0, cc_col]
            if not valid.empty:
                lo = float(valid.min())
        if not rcp85_df.empty and lt_col in rcp85_df.columns:
            valid = rcp85_df.loc[rcp85_df[lt_col] != 0, lt_col]
            if not valid.empty:
                hi = float(valid.max())
        if lo is not None and hi is not None and hi != lo:
            bounds[code] = (lo, hi)
    return bounds


def _normalize_hundred_cols(df: pd.DataFrame, hazard_bounds: dict) -> pd.DataFrame:
    """
    Normalize hazard×time columns using shared per-hazard bounds.

    For each {HAZARD}_{TIME} column, apply the (lo, hi) from hazard_bounds so
    that scores are comparable across scenarios and time periods:
      lo (RCP45 Cc min province) → 0
      hi (RCP85 Lt max province) → 100

    Values are clamped to [0, 100] in case intermediate columns exceed the
    anchor. -9999 (missing) values are left untouched.
    """
    for col in df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) != 2 or parts[0] not in hazard_bounds:
            continue
        lo, hi = hazard_bounds[parts[0]]
        mask = df[col] != 0
        valid = df.loc[mask, col]
        if valid.empty:
            continue
        df.loc[mask, col] = (
            ((valid - lo) / (hi - lo) * 100).clip(0, 100).round(2)
        )
    return df


def write_combined_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write combined DataFrame to GCS CSV."""
    logger.info(f"Writing {output_path} ({len(df)} rows, {len(df.columns)} cols)")
    with fsspec.open(output_path, "w") as f:
        df.to_csv(f, index=False)
    logger.info("Done.")


def combine_scores(
    gadm_levels: List[int] = None,
    hazard_codes: List[str] = None,
    scenarios: List[str] = None,
    scales: List[str] = None,
) -> None:
    """
    Load all per-hazard aggregation CSVs and write combined outputs organised
    by scoring scale:

        combined/five/  — one file per stat × scenario × gadm_level
        combined/ten/   — same

    Always reads all 14 hazard CSVs regardless of `hazard_codes`. Missing CSVs
    and scales absent from a CSV are skipped with a warning (-9999 filled).

    Args:
        gadm_levels: GADM levels to combine, default [1]. Pass [0, 1] for
                     both province and country files.
        hazard_codes: ignored — all 14 hazards are always combined. Kept for
                      API symmetry with score/aggregate functions.
        scenarios: list of scenarios, default ['rcp45', 'rcp85'].
        scales: scoring scales to produce folders for, default ['5', '10'].
    """
    if gadm_levels is None:
        gadm_levels = [1]
    if scenarios is None:
        scenarios = ["RCP45", "RCP85"]
    if scales is None:
        scales = ["5", "10"]

    stat_map = {
        "mean": "score_mean",
        "max":  "score_max",
        "std":  "score_stdev",
    }

    for gadm_level in gadm_levels:
        logger.info(f"Loading GADM adm{gadm_level} metadata...")
        gadm_meta = _load_gadm_meta(gadm_level)
        logger.info(f"Loading all hazard CSVs for adm{gadm_level}...")
        hazard_dfs = {code: _load_hazard_csv(code, gadm_level) for code in ALL_HAZARD_CODES}

        for scale in scales:
            folder = SCALE_FOLDER.get(scale, f"scale{scale}")
            print(f"  scale='{scale}' → {folder}/")

            if scale == "100":
                # For the hundred scale, normalization bounds must be shared
                # across scenarios so RCP45 and RCP85 are on the same scale.
                # lo = min province score in {H}_Cc of RCP45 (mildest)
                # hi = max province score in {H}_Lt of RCP85 (most severe)
                # Two-pass: build all DFs first, then compute bounds, then write.
                rcp45_key = next((s for s in scenarios if "45" in s), None)
                rcp85_key = next((s for s in scenarios if "85" in s), None)
                for stat_label, stat_col in stat_map.items():
                    # Pass 1: build all scenario DFs
                    scenario_dfs = {
                        scenario: _build_combined(
                            gadm_level=gadm_level,
                            scenario=scenario,
                            stat_col=stat_col,
                            scoring_scale=scale,
                            gadm_meta=gadm_meta,
                            hazard_dfs=hazard_dfs,
                            round_to_int=False,
                        )
                        for scenario in scenarios
                    }
                    # Pass 2: compute shared bounds
                    rcp45_df = scenario_dfs.get(rcp45_key, pd.DataFrame())
                    rcp85_df = scenario_dfs.get(rcp85_key, pd.DataFrame())
                    hazard_bounds = _compute_hundred_bounds(rcp45_df, rcp85_df)
                    # Pass 3: normalize and write
                    for scenario, df in scenario_dfs.items():
                        if df.empty:
                            print(
                                f"  SKIP adm{gadm_level}/{scenario}/{stat_label} "
                                f"— no data for scale='{scale}'"
                            )
                            continue
                        df = _normalize_hundred_cols(df, hazard_bounds)
                        out_path = _combined_output_path(scale, stat_label, scenario, gadm_level)
                        print(f"  → {out_path} ({len(df)} rows, {len(df.columns)} cols)")
                        write_combined_csv(df, out_path)
            else:
                for scenario in scenarios:
                    for stat_label, stat_col in stat_map.items():
                        df = _build_combined(
                            gadm_level=gadm_level,
                            scenario=scenario,
                            stat_col=stat_col,
                            scoring_scale=scale,
                            gadm_meta=gadm_meta,
                            hazard_dfs=hazard_dfs,
                            round_to_int=True,
                        )
                        if df.empty:
                            print(
                                f"  SKIP adm{gadm_level}/{scenario}/{stat_label} "
                                f"— no data for scale='{scale}'"
                            )
                            continue
                        out_path = _combined_output_path(scale, stat_label, scenario, gadm_level)
                        print(f"  → {out_path} ({len(df)} rows, {len(df.columns)} cols)")
                        write_combined_csv(df, out_path)
