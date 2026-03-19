"""
app/flows/pipeline_flow.py
Prefect flow for the two-step scoring + aggregation pipeline.

Install:
    uv add prefect

Run locally:
    uv run python app/flows/pipeline_flow.py

Deploy to Prefect Cloud / schedule:
    uv run prefect deploy app/flows/pipeline_flow.py:full_pipeline
"""

import logging

from prefect import flow, task

from app.domain.pipeline import ALL_HAZARD_CODES, aggregate_hazard, combine_all, score_hazard

logger = logging.getLogger(__name__)


@task(name="score-hazard", retries=2, retry_delay_seconds=30, log_prints=True)
def score_hazard_task(hazard_code: str, scales: list) -> str:
    score_hazard(hazard_code, scales=scales)
    return hazard_code


@task(name="aggregate-hazard", retries=2, retry_delay_seconds=30, log_prints=True)
def aggregate_hazard_task(hazard_code: str, gadm_levels: list) -> str:
    aggregate_hazard(hazard_code, gadm_levels=gadm_levels)
    return hazard_code


@task(name="combine-scores", retries=1, retry_delay_seconds=30, log_prints=True)
def combine_task(gadm_levels: list, scales: list) -> None:
    logger.info(f"[combine] starting — gadm_levels={gadm_levels}, scales={scales}")
    combine_all(gadm_levels=gadm_levels, scales=scales)
    logger.info("[combine] done")


def _start_cluster(cluster_type: str) -> "dask.distributed.Client":
    """Spin up (or reuse) a Coiled cluster and return a connected Client."""
    from app.utils.compute import get_or_create_cluster
    from dask.distributed import Client

    cluster = get_or_create_cluster(cluster_type=cluster_type)
    client = Client(cluster)
    logger.info(f"[{cluster_type}] cluster ready — dashboard: {client.dashboard_link}")
    return client


def _shutdown_cluster(client: "dask.distributed.Client", cluster_type: str) -> None:
    """Close the client connection and fully shut down the cluster."""
    try:
        client.cluster.close()
        logger.info(f"[{cluster_type}] cluster shut down")
    except Exception as e:
        logger.info(f"[{cluster_type}] cluster shutdown warning: {e}")
    finally:
        client.close()


_HIGHMEM_AGG_HAZARDS = {"CF", "RF"}


def _run_aggregations(hazard_codes: list, gadm_levels: list) -> None:
    """Run aggregations, using the highmem cluster for CF/RF and standard agg for the rest."""
    standard = [c for c in hazard_codes if c not in _HIGHMEM_AGG_HAZARDS]
    highmem = [c for c in hazard_codes if c in _HIGHMEM_AGG_HAZARDS]

    if standard:
        agg_client = _start_cluster("agg")
        try:
            for code in standard:
                logger.info(f"[{code}] aggregating...")
                aggregate_hazard_task.submit(code, gadm_levels).result()
                logger.info(f"[{code}] aggregation done")
        finally:
            _shutdown_cluster(agg_client, "agg")

    if highmem:
        highmem_client = _start_cluster("agg_highmem")
        try:
            for code in highmem:
                logger.info(f"[{code}] aggregating (highmem)...")
                aggregate_hazard_task.submit(code, gadm_levels).result()
                logger.info(f"[{code}] aggregation done")
        finally:
            _shutdown_cluster(highmem_client, "agg_highmem")


@flow(name="crs-full-pipeline", log_prints=True)
def full_pipeline(
    scales: list = None,
    gadm_levels: list = None,
    hazard_codes: list = None,
):
    """
    Score then aggregate all hazards sequentially (score→agg per hazard).
    Uses three Coiled clusters:
      crs-score       (e2-standard-8)  — scoring
      crs-agg         (e2-highmem-16)  — standard hazard aggregation
      crs-agg-highmem (n2-highmem-32)  — CF/RF only (coastline + RP-weighted zonal stats)

    Args:
        scales: e.g. ['5', '10']. Defaults to ['5', '10'].
        gadm_levels: GADM levels to aggregate, e.g. [1], [2], or [1, 2]. Defaults to [1].
        hazard_codes: subset of hazards to run. Defaults to all 14.
    """
    if scales is None:
        scales = ["5", "10"]
    if gadm_levels is None:
        gadm_levels = [1]
    if hazard_codes is None:
        hazard_codes = ALL_HAZARD_CODES

    logger.info(f"Running pipeline for {len(hazard_codes)} hazards, scales={scales}, gadm_levels={gadm_levels}")

    score_client = _start_cluster("score")
    try:
        for code in hazard_codes:
            logger.info(f"[{code}] scoring...")
            score_hazard_task.submit(code, scales).result()
            logger.info(f"[{code}] scoring done")
    finally:
        _shutdown_cluster(score_client, "score")

    _run_aggregations(hazard_codes, gadm_levels)

    logger.info("Combining all hazard CSVs into final outputs...")
    combine_task.submit(gadm_levels, scales).result()
    logger.info("Pipeline complete")


@flow(name="crs-score-only", log_prints=True)
def score_only(scales: list = None, hazard_codes: list = None):
    """Score all hazards, no aggregation."""
    if scales is None:
        scales = ["5", "10"]
    if hazard_codes is None:
        hazard_codes = ALL_HAZARD_CODES

    client = _start_cluster("score")
    try:
        for code in hazard_codes:
            logger.info(f"[{code}] scoring...")
            score_hazard_task.submit(code, scales).result()
            logger.info(f"[{code}] scoring done")
    finally:
        _shutdown_cluster(client, "score")


@flow(name="crs-aggregate-only", log_prints=True)
def aggregate_only(gadm_levels: list = None, hazard_codes: list = None):
    """Aggregate all hazards (scored Zarrs must already exist on GCS)."""
    if gadm_levels is None:
        gadm_levels = [1]
    if hazard_codes is None:
        hazard_codes = ALL_HAZARD_CODES

    _run_aggregations(hazard_codes, gadm_levels)


@flow(name="crs-combine-only", log_prints=True)
def combine_only(gadm_levels: list = None, scales: list = None):
    """Combine all per-hazard aggregation CSVs into final wide-format outputs."""
    if gadm_levels is None:
        gadm_levels = [1]
    if scales is None:
        scales = ["5", "10"]

    logger.info(f"Combining scores for adm_levels={gadm_levels}")
    combine_task.submit(gadm_levels, scales).result()
    logger.info("Combine complete")


if __name__ == "__main__":
    full_pipeline()
