"""
app/scripts/convert_gadm.py
One-time offline job: convert GADM GeoPackages (on GCS) to GeoParquet.

The source GeoPackages live in europe-west8 (same region as all other CRS data).
Run this on a cloud VM in the same region to avoid egress and get ~10 GB/s
GCS throughput. The adm2 file is the heaviest: ~400k rows with global polygon
geometries; it needs roughly 20-30 GB of RAM to load in full.

Output layout:
  gs://.../GADM/parquet/adm0.parquet          (~200 rows, load in full)
  gs://.../GADM/parquet/adm1.parquet          (~4k rows,  load in full)
  gs://.../GADM/parquet/adm2/{ISO3}.parquet   (~400k rows total, one file per country)

--- Running options ---

Option A — coiled.function from a local shell or notebook (recommended):
  Prerequisite (one-time):
    coiled env create --name crs-gadm --pip app/scripts/requirements-gadm.txt

  Then:
    from app.scripts.convert_gadm import run_on_coiled
    run_on_coiled()

  This uses a pre-built remote environment so there is no local Python version
  or package-sync dependency. All data stays in GCS; only a thin job submission
  goes over the network.

Option B — coiled run CLI with a pre-built env:
  Prerequisite (one-time, same as above):
    coiled env create --name crs-gadm --pip app/scripts/requirements-gadm.txt

  Upload the script to GCS, then run it:
    gsutil cp app/scripts/convert_gadm.py gs://crs_climate_data_public/scripts/
    coiled run \\
      --n_workers 2 \\
      --vm-type e2-highmem-8 \\
      --region europe-west4 \\
      --software crs-gadm \\
      -- bash -c "gsutil cp gs://crs_climate_data_public/scripts/convert_gadm.py . && python convert_gadm.py"

  NOTE: avoid --sync if your local Python version differs from the Coiled default
  (Coiled uses 3.11+; this project is pinned to 3.9).

Option C — Local (not recommended for adm2):
  uv run python app/scripts/convert_gadm.py
"""

import logging
import sys

import fsspec
import geopandas as gpd

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

SRC_L0 = "gs://crs_climate_data_public/GADM/GADM/gadm410/gadm_410levels__adm_0.gpkg"
SRC_L1 = "gs://crs_climate_data_public/GADM/GADM/gadm410/gadm_410levels__adm_1.gpkg"
SRC_L2 = "gs://crs_climate_data_public/GADM/GADM/gadm410/gadm_410levels__adm_2.gpkg"

OUT_ADM0 = "gs://crs_climate_data_public/GADM/parquet/adm0.parquet"
OUT_ADM1 = "gs://crs_climate_data_public/GADM/parquet/adm1.parquet"
OUT_ADM1_BASE = "gs://crs_climate_data_public/GADM/parquet/adm1"
OUT_ADM2_BASE = "gs://crs_climate_data_public/GADM/parquet/adm2"


def _read_gpkg(gcs_path: str) -> gpd.GeoDataFrame:
    logger.info(f"Reading {gcs_path}")
    with fsspec.open(gcs_path, "rb") as f:
        return gpd.read_file(f)


def convert_adm0():
    gdf = _read_gpkg(SRC_L0)
    logger.info(f"adm0: {len(gdf)} rows — writing to {OUT_ADM0}")
    gdf.to_parquet(OUT_ADM0)
    logger.info("adm0 done")


def convert_adm1():
    gdf = _read_gpkg(SRC_L1)
    logger.info(f"adm1: {len(gdf)} rows")

    # Full file — used only for getting the country list in aggregate_gadm()
    logger.info(f"Writing full adm1 to {OUT_ADM1}")
    gdf.to_parquet(OUT_ADM1)

    # Per-country partitions — loaded by workers during aggregation
    countries = gdf["GID_0"].unique()
    logger.info(f"Partitioning adm1 into {len(countries)} country files under {OUT_ADM1_BASE}/")

    fs = fsspec.filesystem("gs")
    for gid0 in countries:
        subset = gdf[gdf["GID_0"] == gid0].copy()
        out_path = f"{OUT_ADM1_BASE}/{gid0}.parquet"
        with fs.open(out_path, "wb") as f:
            subset.to_parquet(f)

    logger.info("adm1 done")


def convert_adm2():
    # adm2 is the heavy one: load once, partition by country, write per-country files.
    # Needs ~20-30 GB RAM. Use a highmem VM.
    gdf = _read_gpkg(SRC_L2)
    logger.info(f"adm2: {len(gdf)} rows total")

    countries = gdf["GID_0"].unique()
    logger.info(f"Partitioning adm2 into {len(countries)} country files")

    fs = fsspec.filesystem("gs")
    for gid0 in countries:
        subset = gdf[gdf["GID_0"] == gid0].copy()
        out_path = f"{OUT_ADM2_BASE}/{gid0}.parquet"
        with fs.open(out_path, "wb") as f:
            subset.to_parquet(f)

    logger.info("adm2 done")


def convert_all():
    convert_adm0()
    convert_adm1()
    #convert_adm2()
    logger.info("All GADM conversions complete")


def run_on_coiled(software_env: str = "crs-gadm"):
    """
    Run the full conversion on a single high-memory Coiled VM using a
    pre-built software environment (avoids local Python version conflicts).

    Prerequisite — create the environment once:
        coiled env create --name crs-gadm --pip app/scripts/requirements-gadm.txt

    Then call from a local shell or notebook:
        from app.scripts.convert_gadm import run_on_coiled
        run_on_coiled()
    """
    import coiled

    # Capture paths as locals so they are pickled into the closure,
    # not resolved via an app import on the remote machine.
    _SRC_L0 = SRC_L0
    _SRC_L1 = SRC_L1
    _SRC_L2 = SRC_L2
    _OUT_ADM0 = OUT_ADM0
    _OUT_ADM1 = OUT_ADM1
    _OUT_ADM1_BASE = OUT_ADM1_BASE
    _OUT_ADM2_BASE = OUT_ADM2_BASE

    @coiled.function(
        n_workers=2,
        vm_type="e2-highmem-8",   # 64 GB RAM, 8 vCPU — worker needs the RAM
        region="europe-west4",    # collocated with GCS bucket
        software=software_env,    # pre-built env; no local package sync needed
        keepalive="10 minutes",
    )
    def _remote():
        import logging
        import sys
        import fsspec
        import geopandas as gpd

        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        log = logging.getLogger(__name__)

        def _read(path):
            log.info(f"Reading {path}")
            with fsspec.open(path, "rb") as f:
                return gpd.read_file(f)

        # adm0 — full file only (~200 rows)
        gdf = _read(_SRC_L0)
        log.info(f"adm0: {len(gdf)} rows")
        gdf.to_parquet(_OUT_ADM0)

        # adm1 — full file + per-country partitions
        gdf = _read(_SRC_L1)
        log.info(f"adm1: {len(gdf)} rows — writing full file")
        gdf.to_parquet(_OUT_ADM1)

        countries = gdf["GID_0"].unique()
        log.info(f"adm1: partitioning into {len(countries)} country files")
        fs = fsspec.filesystem("gs")
        for gid0 in countries:
            subset = gdf[gdf["GID_0"] == gid0].copy()
            with fs.open(f"{_OUT_ADM1_BASE}/{gid0}.parquet", "wb") as f:
                subset.to_parquet(f)

        # adm2 — uncomment when ready; needs ~30 GB RAM
        # gdf = _read(_SRC_L2)
        # fs = fsspec.filesystem("gs")
        # for gid0 in gdf["GID_0"].unique():
        #     subset = gdf[gdf["GID_0"] == gid0].copy()
        #     with fs.open(f"{_OUT_ADM2_BASE}/{gid0}.parquet", "wb") as f:
        #         subset.to_parquet(f)

        log.info("GADM conversion complete")

    _remote()


if __name__ == "__main__":
    convert_all()
