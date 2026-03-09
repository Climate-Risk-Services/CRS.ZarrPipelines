"""
app/utils/pipeline_config.py
Canonical pipeline config loader with env-var override support.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def load_pipeline_config() -> dict:
    """Load pipeline.yaml and apply GCS env var overrides from .env / environment."""
    load_dotenv()
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: dict) -> None:
    gcs = cfg["gcs"]
    for env_var, key in [
        ("GCS_OUTPUT_ZARR_BASE", "output_zarr_base"),
        ("GCS_CSV_OUTPUT",       "csv_output"),
        ("GCS_COMBINED_OUTPUT",  "combined_output"),
        ("GCS_GADM_PARQUET_BASE","gadm_parquet_base"),
        ("GCS_TMP_BASE",         "tmp_base"),
    ]:
        if val := os.environ.get(env_var):
            gcs[key] = val

    gadm = cfg["gadm"]
    for env_var, key in [
        ("GCS_GADM_SOURCE_L0", "source_l0"),
        ("GCS_GADM_SOURCE_L1", "source_l1"),
        ("GCS_GADM_COASTLINE", "coastline"),
    ]:
        if val := os.environ.get(env_var):
            gadm[key] = val

    parquet = gadm["parquet"]
    for env_var, key in [
        ("GCS_GADM_ADM0",      "adm0"),
        ("GCS_GADM_ADM1",      "adm1"),
        ("GCS_GADM_ADM1_BASE", "adm1_base"),
        ("GCS_GADM_ADM2_BASE", "adm2_base"),
    ]:
        if val := os.environ.get(env_var):
            parquet[key] = val

    if input_base := os.environ.get("GCS_INPUT_BASE"):
        for hazard_cfg in cfg["hazards"].values():
            if "input" in hazard_cfg:
                original = hazard_cfg["input"]
                # Replace everything up to and including the default base prefix
                _DEFAULT_INPUT_BASE = (
                    "gs://crs_climate_data_public/PHYSICAL_HAZARDS_WORKING_FOLDER"
                )
                if original.startswith(_DEFAULT_INPUT_BASE):
                    hazard_cfg["input"] = input_base + original[len(_DEFAULT_INPUT_BASE):]
