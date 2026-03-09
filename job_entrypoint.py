"""
Cloud Run Job entrypoint.

Reads pipeline parameters from environment variables and runs the specified flow.

Environment variables:
    FLOW          : full | score | agg | combine  (default: full)
    HAZARD_CODES  : comma-separated list, e.g. "HS,CS" (default: all 14)
    SCALES        : comma-separated list, e.g. "5,10" (default: "5,10")
    GADM_LEVELS   : comma-separated integers, e.g. "1" (default: "1")
"""

import os

from app.flows.pipeline_flow import aggregate_only, combine_only, full_pipeline, score_only

flow = os.environ.get("FLOW", "full")
scales = os.environ.get("SCALES", "5,10").split(",")
gadm_levels = [int(x) for x in os.environ.get("GADM_LEVELS", "1").split(",")]
hazard_codes_env = os.environ.get("HAZARD_CODES", "")
hazard_codes = hazard_codes_env.split(",") if hazard_codes_env else None

if flow == "full":
    full_pipeline(scales=scales, gadm_levels=gadm_levels, hazard_codes=hazard_codes)
elif flow == "score":
    score_only(scales=scales, hazard_codes=hazard_codes)
elif flow == "agg":
    aggregate_only(gadm_levels=gadm_levels, hazard_codes=hazard_codes)
elif flow == "combine":
    combine_only(gadm_levels=gadm_levels)
else:
    raise ValueError(f"Unknown FLOW value: {flow!r}. Must be one of: full, score, agg, combine")
