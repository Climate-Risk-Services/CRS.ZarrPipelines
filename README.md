# CRS ZarrPipelines

Climate hazard risk scoring and GADM aggregation pipeline. Reads geospatial raster data (Zarr / xarray / Dask on GCS), applies configurable threshold-based scoring, and aggregates results to GADM administrative boundaries using zonal statistics.

---

## Architecture

### Infrastructure

```
┌─────────────────────────────────────────────────────────┐
│  You / external service                                  │
│                                                          │
│  POST /pipeline/run   {"flow": "full", ...}             │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
                           ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI  (Cloud Run Service — scales to zero)           │
│  Receives request, calls Cloud Run Jobs API, returns     │
│  execution_id immediately.                               │
└──────────────────────────┬───────────────────────────────┘
                           │ Cloud Run Jobs API
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Cloud Run Job  (serverless, up to 24h, billed per sec)  │
│  Runs job_entrypoint.py → calls full_pipeline()          │
│  Uses Prefect @flow/@task for retries and logging        │
└──────────┬───────────────────────────┬───────────────────┘
           │ Coiled API                │ Coiled API
           ▼                           ▼
┌─────────────────┐         ┌──────────────────────┐
│  crs-score      │         │  crs-agg             │
│  e2-standard-8  │         │  e2-highmem-16       │
│  25–60 workers  │         │  10–30 workers       │
│  Zarr scoring   │         │  GADM zonal stats    │
└────────┬────────┘         └──────────┬───────────┘
         │                             │
         └──────────────┬──────────────┘
                        ▼
              GCS  crs_climate_data_public/production_test/
              ├── hazard_scores/{CODE}.zarr
              └── aggregations/
                  ├── {CODE}_adm0.csv          (per-hazard, tidy long format)
                  ├── {CODE}_adm1.csv
                  └── combined/
                      ├── five/                (5-point, integer scores 1–5)
                      │   ├── hazard_score_province_mean_45.csv
                      │   ├── hazard_score_province_max_45.csv
                      │   ├── hazard_score_province_std_45.csv
                      │   ├── hazard_score_province_mean_85.csv
                      │   ├── hazard_score_province_max_85.csv
                      │   └── hazard_score_province_std_85.csv
                      ├── ten/                 (10-point, same files)
                      └── hundred/             (0–100 min-max, float values)
```

**Two Coiled clusters** are used sequentially within each pipeline run:
- `crs-score` — CPU/network bound (Zarr chunk processing), e2-standard-8. **Fully shut down** after all scoring tasks complete.
- `crs-agg` — memory bound (province zarr slices), e2-highmem-16. Stays warm between runs (idle timeout 2h).

### Pipeline flow

```
crs-score cluster open
  score ER → score CER → score CF → ... → score WS   (14 zarrs → GCS)
crs-score cluster close

crs-agg cluster open
  agg ER → agg CER → agg CF → ... → agg WS           (14 per-hazard CSVs → GCS)
crs-agg cluster close

combine                                                (final wide CSVs → GCS)
  aggregations/combined/five/
    hazard_score_province_mean_45.csv
    hazard_score_province_max_45.csv
    hazard_score_province_std_45.csv
    hazard_score_province_mean_85.csv
    hazard_score_province_max_85.csv
    hazard_score_province_std_85.csv
  aggregations/combined/ten/                           (same files)
  aggregations/combined/hundred/                       (float values, 2 dp)
```

Prefect `@task(retries=2)` wraps each hazard — a single failure retries without restarting the whole pipeline.

### Code layout

```
app/
├── api/
│   ├── main.py                  # FastAPI app
│   └── routers/
│       └── pipeline.py          # POST /pipeline/run, GET /pipeline/status/{id}
├── config/
│   ├── scoring.yaml             # Hazard thresholds (all 14 hazards, 5-pt + 10-pt)
│   └── pipeline.yaml            # GCS paths, Coiled cluster config, per-hazard inputs
├── domain/
│   ├── scoring.py               # score_zarr(), score_zarr_multi(), score_zarr_minmax()
│   ├── special/
│   │   ├── ls.py                # Landslide composite: score(ARI) × score(susceptibility)
│   │   ├── wf.py                # Wildfire composite: burnability × FWI
│   │   ├── rf.py                # River Flood: ensemble mean → score → protection subtraction
│   │   └── cf.py                # Coastal Flood: same as RF + coastline mask denominator
│   ├── gadm_aggregations.py     # xvec zonal stats, per-country Dask dispatch
│   ├── combine.py               # combine_scores() — merge per-hazard CSVs → final outputs
│   └── pipeline.py              # score_hazard(), aggregate_hazard(), combine_all()
├── flows/
│   └── pipeline_flow.py         # Prefect flow: full_pipeline, score_only, aggregate_only,
│                                #               combine_only
└── utils/
    ├── scoring_config.py        # ScoringConfig — YAML loader + threshold lookup
    ├── compute.py               # Coiled cluster lifecycle (get_or_create_cluster)
    └── job_store.py             # In-memory job tracking (local dev only)

job_entrypoint.py                # Cloud Run Job entrypoint — reads env vars, calls flow
main.py                          # FastAPI entrypoint — uvicorn on :8000
Dockerfile                       # Single image for both Service and Job
```

### Supported hazards

| Code | Name | Type | Input variable | Notes |
|------|------|------|----------------|-------|
| ER  | Erosion | standard | `erosion` | |
| CER | Cropland Erosion | standard | `cropland_erosion` | |
| CF  | Coastal Flooding | special_cf | `return_period_0_5_m` | ensemble mean → `statistic='mean'`; RP-weighted + coastal pixel denominator |
| RF  | River Flooding | special_rf | `return_period_0_5_m` | 6-model ensemble mean → `statistic='mean'`; RP-weighted aggregation |
| CS  | Cold Stress | standard | `utci` (metric `p5`) | |
| HS  | Heat Stress | standard | `utci` | |
| DR  | Drought | standard | `drought_events_per_decade` | |
| HP  | Heavy Precipitation | standard | `extremes` (metric `p95`, ×100) | raw is 0–1 fraction; thresholds expect 0–100 |
| LS  | Landslide | special_ls | `ari`, `susceptibility` | score(ARI) × score(susceptibility) → score(final); staged zarr writes |
| SUB | Subsidence | sub | `subsidence` | raw ×2 before 10-point scoring |
| TC  | Tropical Cyclone | standard | `TC_windspeed_50_m_s` | |
| TS  | Tropical Storm | standard | `TS_windspeed_18_m_s` | |
| WF  | Wildfire | special_wf | `burnability`, `fwi` | burnability × FWI composite → score |
| WS  | Water Stress | standard | `availability_vs_demand` | |

### Input dimension conventions

The pipeline handles any combination of the following dimensions in the input zarr:

| Dimension | Required | Handling |
|-----------|----------|----------|
| `lat` / `lon` | ✅ | Spatial axes for zonal statistics |
| `scenario` | Recommended | Passes through scoring unchanged → becomes a CSV column; combine filters by `RCP45` / `RCP85` |
| `time` | Recommended | Passes through scoring unchanged → CSV column; combine pivots on `Cc / St / Mt / Lt` |
| `statistic` | Optional | `_aggregate_partition` selects `mean` if present; size-1 is auto-squeezed |
| `model` | Not supported directly | Must be collapsed before scoring — RF/CF do this automatically (ensemble mean → `statistic='mean'`) |
| `metric` | Optional | Size-1 auto-squeezed; multi-value passes through (e.g. HS has `metric=['p95']`) |
| `scoring` | Added by pipeline | Scoring step adds this dim with values like `['5','10','100']`; aggregation and combine use it |

The canonical scored zarr shape is `(scoring, statistic, scenario, time, lat, lon)` with `statistic='mean'` being the value used downstream.

### Scoring scales

| Scale | Output | Description |
|-------|--------|-------------|
| `"5"` | integer 1–5 | Threshold-based bins |
| `"10"` | integer 1–10 | Threshold-based bins (finer granularity) |
| `"100"` | float16 0–100 | Min-max normalisation using **RCP85/Lt** as reference (widest range, prevents clipping of high-risk future values) |

For **special hazards**, the 0–100 scale normalises the intermediate composite:
- **LS** — composite = `score_ari × score_susc` (before final threshold scoring)
- **WF** — composite = `burnability × fwi`
- **RF / CF** — raw inundation depth after ensemble mean (before protection subtraction)
- **SUB** — raw variable (without ×2 scaling artifact)
- **Standard** — prepared variable (after `metric_select` / `scale_factor`)

**Aggregation behaviour per scale:**
- Scales `"5"` and `"10"` — standard xvec zonal stats (mean/max/stdev) for all hazards; CF and RF additionally use the RP-weighted custom path.
- Scale `"100"` — xvec zonal stats (mean/max/stdev) for all hazards including CF and RF (RP-weighting not applied to continuous 0–100 values).

### Float16 throughout

All data is cast to `float16` at every write boundary:

| Stage | Where |
|-------|-------|
| Special hazard stage-1 tmp zarr | `.astype("float16")` before write (LS/WF/RF/CF) |
| Threshold scoring | `_score_chunk` returns `float16` |
| Min-max scoring | `score_zarr_minmax` output is `float16` |
| Output scored zarr | `_fix_and_cast` in `_write_scored_zarr` |
| Aggregation worker | `.astype(np.float16)` after `.compute()` |

This reduces RF's 80 GB tmp zarr to ~40 GB and keeps peak worker memory proportional to the largest province, not the full global grid.

### Smart merge for scored zarrs

`score_hazard()` uses native zarr region-write and append so only affected slices are written — unchanged scales are never touched:

```python
score_hazard("HS", scales=["5", "10"])   # creates HS.zarr  → scoring: [5, 10]
score_hazard("HS", scales=["100"])       # append_dim        → scoring: [5, 10, 100]
score_hazard("HS", scales=["5"])         # region write      → scoring: [5, 10, 100]
```

| Existing | New | Action | Data written |
|---|---|---|---|
| none | any | fresh `mode="w"` | all scales |
| `["5","10"]` | `["100"]` | `append_dim` | new scale only |
| `["5","10"]` | `["5"]` | region write at index 0 | replaced scale only |
| `["5","10"]` | `["5","100"]` | region write + append | 2 scales only |
| `["10","100"]` | `["5"]` | fallback full rewrite (warn) | all scales |

---

## Setup

Requires Python 3.9.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd CRS.ZarrPipelines
uv sync
```

### Google Cloud authentication

```bash
gcloud auth application-default login
```

---

## Configuration

### `app/config/scoring.yaml`

Scoring thresholds for all hazards, both scales. Edit to adjust risk bins.

### `app/config/pipeline.yaml`

GCS input/output paths, Coiled cluster settings, GADM parquet paths, and custom aggregation config.

```yaml
gcs:
  output_zarr_base: gs://crs_climate_data_public/production_test/hazard_scores
  csv_output:       gs://crs_climate_data_public/production_test/aggregations
  combined_output:  gs://crs_climate_data_public/production_test/aggregations/combined

coiled:
  score:
    name: crs-score
    worker_vm_types: e2-standard-8    # CPU/network bound
    min_workers: 25
    max_workers: 60
  agg:
    name: crs-agg
    worker_vm_types: e2-highmem-16    # memory bound
    min_workers: 10
    max_workers: 30

hazards:
  HS: { input: gs://...HS.zarr/, variable: utci, type: standard }
  RF: { input: gs://...RF_inundation_50cm.zarr/, variable: return_period_0_5_m,
        protection_variable: flood_protection, type: special_rf }
  LS: { input: gs://...LS.zarr/, type: special_ls }
  # ... one entry per hazard
```

---

## One-time setup: GADM GeoParquet conversion

GADM boundaries are stored as GeoPackages on GCS. Convert once — the pipeline reads only parquet at runtime.

### GCS output layout

```
gs://crs_climate_data_public/GADM/parquet/
├── adm0.parquet              # 263 rows  — country boundaries
├── adm1.parquet              # ~3600 rows — full provinces (country list lookup)
└── adm1/
    ├── AFG.parquet           # Afghanistan provinces only (~20 rows)
    ├── NGA.parquet
    ...                       # one file per GID_0 country code
```

### Run conversion

```python
from app.scripts.convert_gadm import run_on_coiled
run_on_coiled()   # e2-highmem-8, europe-west4
```

---

## Testing

```bash
uv run pytest tests/
```

Tests cover threshold scoring logic (`_score_chunk`, `score_zarr_multi`), config lookup (`ScoringConfig.get_thresholds`, `score_value`), and the fixed-schema combine step (`_build_combined`) — no network access required.

---

## Local development

You don't need Cloud Run, Coiled, or a running API to develop and test.

### Level 0 — Scoring logic only (no GCS, no Dask)

```python
import numpy as np, xarray as xr
from app.domain.scoring import score_zarr_multi
from app.utils.scoring_config import ScoringConfig

cfg = ScoringConfig()
data = xr.DataArray(np.linspace(0, 50, 100).reshape(10, 10), dims=["lat", "lon"])
thresholds_per_scale = {"5": cfg.get_thresholds("HS", "5"), "10": cfg.get_thresholds("HS", "10")}
scored = score_zarr_multi(data, scales=["5", "10"], thresholds_per_scale=thresholds_per_scale)
print(scored)
```

### Level 1 — Score a real GCS Zarr locally (no Coiled)

```python
from app.domain.pipeline import score_hazard

score_hazard("HS", scales=["5", "10", "100"])  # creates/merges HS.zarr on GCS

# Add "100" to an existing 5/10 zarr without rewriting them:
score_hazard("HS", scales=["100"])
```

### Level 2 — Aggregate one country (no Coiled)

```python
from app.domain.gadm_aggregations import _aggregate_partition
df = _aggregate_partition("NGA", "gs://.../hazard_scores/HS.zarr", "HS", gadm_level=1)
print(df.head())
```

### Level 3 — Combine per-hazard CSVs into final outputs (no Coiled)

Runs locally — only reads GCS CSVs and GADM parquet, no Dask cluster needed.

```python
from app.domain.combine import combine_scores
combine_scores(gadm_levels=[1], scales=["5", "10", "100"])
# → gs://.../aggregations/combined/five/  ten/  hundred/
```

### Level 4 — Full pipeline via Prefect (uses Coiled)

```python
from app.flows.pipeline_flow import full_pipeline
full_pipeline(
    scales=["5", "10", "100"],
    gadm_levels=[0, 1],
    hazard_codes=["HS", "HP"],   # omit hazard_codes to run all 14
)
```

---

## Running the pipeline

### Direct Python (recommended for one-off runs)

```python
from app.flows.pipeline_flow import full_pipeline, score_only, aggregate_only, combine_only

# Full run — all 14 hazards (score → aggregate → combine)
full_pipeline(scales=["5", "10", "100"], gadm_levels=[0, 1])

# Subset
full_pipeline(scales=["5", "10"], gadm_levels=[1], hazard_codes=["HS", "CS"])

# Individual phases
score_only(hazard_codes=["HS"])
aggregate_only(gadm_levels=[1])
combine_only(gadm_levels=[1])
combine_only(gadm_levels=[1], scales=["100"])  # only rebuild hundred/ folder
```

### Via HTTP (FastAPI → Cloud Run Job)

```bash
# Trigger a full pipeline run
curl -X POST https://YOUR_API_URL/pipeline/run \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"flow": "full", "scales": ["5", "10"], "gadm_levels": [1]}'
# → {"execution_id": "crs-pipeline-job-abc123", "status": "launched", "flow": "full"}

# Check status
curl https://YOUR_API_URL/pipeline/status/crs-pipeline-job-abc123 \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)"
```

---

## GCP deployment

### Prerequisites

```bash
export PROJECT=your-gcp-project
export REGION=europe-west4
export REPO=crs
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/zarrpipelines

# Create Artifact Registry repo (once)
gcloud artifacts repositories create $REPO \
  --repository-format=docker --location=$REGION
```

### Build and push image

```bash
gcloud builds submit --tag $IMAGE
```

### Deploy Cloud Run Job (pipeline)

```bash
gcloud run jobs create crs-pipeline-job \
  --image $IMAGE \
  --command "python" \
  --args "job_entrypoint.py" \
  --region $REGION \
  --service-account YOUR_SA@$PROJECT.iam.gserviceaccount.com \
  --task-timeout 86400 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars FLOW=full
```

### Deploy FastAPI as Cloud Run Service (HTTP trigger)

```bash
gcloud run deploy crs-api \
  --image $IMAGE \
  --region $REGION \
  --service-account YOUR_SA@$PROJECT.iam.gserviceaccount.com \
  --no-allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT,CLOUD_RUN_REGION=$REGION,PIPELINE_JOB_NAME=crs-pipeline-job
```

### IAM: allow the API service account to launch jobs

```bash
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:YOUR_SA@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/run.developer"
```

---

## API reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/pipeline/run` | `{"flow": "full\|score\|agg\|combine", "hazard_codes": [...], "scales": [...], "gadm_levels": [...]}` | Launch Cloud Run Job |
| `GET`  | `/pipeline/status/{execution_id}` | — | Query execution status |
| `GET`  | `/health` | — | Health check |

---

## Status

### Working and tested locally

| Component | Status |
|-----------|--------|
| Scoring logic — `score_zarr_multi`, `ScoringConfig` | ✅ unit-tested (`uv run pytest tests/`) |
| `score_hazard()` — all standard hazards | ✅ runs locally against real GCS Zarrs |
| `score_hazard()` — special hazards (LS, WF, RF, CF) | ✅ implemented; LS and WF confirmed end-to-end |
| Smart zarr merge (region-write / append / rewrite) | ✅ tested for all merge cases |
| `_aggregate_partition()` — single country, adm1 | ✅ tested (NGA and others) |
| `aggregate_gadm()` — full adm1 via Coiled | ✅ aggregation runs completing |
| `combine_scores()` — per-hazard CSVs → combined wide outputs | ✅ runs locally; no cluster needed |
| `pipeline_config.py` — env var overrides via `.env` | ✅ smoke-tested |
| GADM adm1 per-country parquet files on GCS | ✅ present and confirmed |

### Implemented but not yet deployed / end-to-end verified

| Component | What's missing |
|-----------|----------------|
| **Prefect flow** (`pipeline_flow.py`) | No Prefect worker deployed; flows run directly in-process only |
| **FastAPI** (`app/api/`) | Not deployed to Cloud Run; local `uv run python main.py` works but job launch calls the Cloud Run Jobs API which requires a deployed Job |
| **Cloud Run Job** (`job_entrypoint.py`, `Dockerfile`) | Image not built or pushed; Job not created in GCP |
| **Full end-to-end pipeline** (score → agg → combine, all 14 hazards) | TC and TS blocked on data regeneration (see below); remaining 12 not run together |
| **adm0 aggregation** | Implemented but large countries (RUS, USA) exceed worker RAM — see workaround options below |
| **adm2 aggregation** | Parquet files not yet converted; code path exists but commented out |

---

## Known limitations / next steps

- **adm0 aggregations** — large countries (RUS, USA, CAN) exceed worker RAM at LS resolution even with float16 (USA ≈ 38 GB slice). Two approaches under consideration: (1) spatial tiling using adm1 bboxes with additive sum/count stats, (2) e2-highmem-32 workers for an adm0-specific cluster. CF/RF adm0 tiling is more complex (coastline denominator must also be tiled).
- **RF/CF 10-point quantile bins** — `q20/q40/q60/q80` for scale `"10"` are `null` in `pipeline.yaml` and must be calibrated from the full global zarr before 10-point custom aggregation works.
- **adm2** — parquet conversion not yet run; uncomment in `run_on_coiled()` when adm0/adm1 verified.
