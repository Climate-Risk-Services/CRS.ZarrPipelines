"""
app/utils/compute.py
Coiled cluster lifecycle management.
"""

import logging
from typing import Optional

from app.utils.pipeline_config import load_pipeline_config

logger = logging.getLogger(__name__)

_clusters: dict = {}  # name -> coiled.Cluster


def _load_coiled_config(cluster_type: str = "score") -> dict:
    return load_pipeline_config()["coiled"][cluster_type]


def get_or_create_cluster(cluster_type: str = "score", backend: str = "coiled"):
    """
    Return a running cluster, creating one if it doesn't exist.

    Args:
        cluster_type: 'score' or 'agg' — selects the coiled config block in pipeline.yaml.
                      'score' uses e2-standard-8 (CPU/network bound, zarr chunk processing).
                      'agg'        uses e2-highmem-16 (memory bound, province zarr slices).
                      'agg_highmem' uses n2-highmem-32 (CF/RF only — coastline + RP zonal stats).
        backend: 'coiled' (default) or 'local'.
    """
    from dask.distributed import Client

    if backend == "local":
        from dask.distributed import LocalCluster
        logger.info("Starting LocalCluster")
        cluster = LocalCluster()
        client = Client(cluster)
        print(f"LocalCluster ready — dashboard: {client.dashboard_link}")
        return cluster

    import coiled

    cfg = _load_coiled_config(cluster_type)
    cluster_name = cfg["name"]

    if cluster_name in _clusters:
        cluster = _clusters[cluster_name]
        try:
            _ = cluster.status
            logger.info(f"Reusing existing cluster: {cluster_name}")
            return cluster
        except Exception:
            logger.info(f"Cluster {cluster_name} unreachable, creating new one")

    logger.info(f"Spinning up Coiled cluster: {cluster_name} ({cluster_type})")
    cluster = coiled.Cluster(
        scheduler_vm_types=cfg["scheduler_vm_types"],
        worker_vm_types=cfg["worker_vm_types"],
        region=cfg["region"],
        name=cluster_name,
        n_workers=cfg.get("min_workers", 10),
        shutdown_on_close=cfg.get("shutdown_on_close", False),
        idle_timeout=cfg.get("idle_timeout", "2 hours"),
        package_sync_use_uv_installer=True,
        package_sync_ignore=["tornado", "msgpack", "cloudpickle"],
    )
    cluster.adapt(
        minimum=cfg.get("min_workers", 10),
        maximum=cfg.get("max_workers", 60),
    )
    _clusters[cluster_name] = cluster
    Client(cluster)
    logger.info(f"Cluster ready: {cluster_name}")
    return cluster


def shutdown_cluster(name: Optional[str] = None):
    """Shut down a named Coiled cluster."""
    cfg = _load_coiled_config()
    cluster_name = name or cfg["name"]
    cluster = _clusters.pop(cluster_name, None)
    if cluster is not None:
        logger.info(f"Shutting down cluster: {cluster_name}")
        cluster.close()
