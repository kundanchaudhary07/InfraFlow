# DeployGuard AI — Phase 4: Deployment Simulation & Change Ingestion

> **Project:** DeployGuard AI  
> **Current phase:** Phase 4 — Deployment Simulation & Change Ingestion  
> **Phase goal:** Build deterministic change-ingestion, git diff parsing, and risk feature extraction layer that merges code change characteristics with runtime telemetry into unified feature vectors.

---

## Project Purpose

DeployGuard AI is a pre-deployment risk analysis and safe-release decision system for DevOps teams.

It will analyse proposed code changes, infrastructure changes, service dependencies, and historical deployment telemetry to predict deployment failure risk and recommend one of:

- `NORMAL` deployment
- `CANARY` deployment
- `LIMITED` rollout
- `BLOCK` deployment

**This phase builds the risk feature extraction layer.** It ingests simulated deployment changes and unified Git diffs, classifies touched paths against critical risk rules (blast radius, configs, core apps, tests), and merges them with runtime telemetry into a machine-readable feature vector for future ML model training and release evaluation.

---

## Architecture Overview

```
deployguard-demo/
│
├── app/
│   ├── main.py                 # FastAPI application factory; registers routers & middleware
│   ├── config.py               # Environment-based configuration (pydantic-settings)
│   ├── models.py               # Pydantic request/response schemas for Order domain
│   ├── data.py                 # In-memory order store and lookup helpers
│   ├── telemetry.py            # Structured JSON logging, RequestTelemetryMiddleware, latency timing
│   ├── change_analysis.py      # Git diff parser, path classification, feature extraction engine
│   └── routes/
│       ├── health.py           # GET /health
│       ├── orders.py           # GET /api/v1/orders, GET /api/v1/orders/{order_id}
│       └── change_analysis.py  # POST /api/v1/analyze-change, GET /api/v1/analyze-change/samples
│
├── tests/
│   ├── conftest.py             # pytest configuration and shared fixtures
│   ├── test_health.py          # Health endpoint tests
│   ├── test_orders.py          # Orders endpoint tests
│   ├── test_telemetry.py       # Telemetry, correlation, latency & security tests
│   └── test_change_analysis.py # Change ingestion, diff parsing, and feature extraction tests
│
├── .dockerignore               # Docker build context exclusion rules
├── .env.example                # Template for local environment variables
├── .gitignore
├── Dockerfile                  # Production-grade container specification
├── pyproject.toml              # pytest settings
├── requirements.txt            # Local development convenience entrypoint
├── requirements-runtime.txt    # Lean runtime-only dependencies (installed in Docker)
├── requirements-dev.txt        # Local test/dev dependencies (pytest, httpx)
└── README.md
```

**Request lifecycle:**

```
HTTP Request
    │
    ▼
FastAPI (app/main.py)
    │
    ▼
RequestTelemetryMiddleware (app/telemetry.py)
    │  ├─ Start monotonic clock (time.perf_counter)
    │  ├─ Extract or generate X-Request-ID
    │  └─ Call next handler
    │         │
    │         ├── /health           ──▶ app/routes/health.py
    │         └── /api/v1/orders/*  ──▶ app/routes/orders.py ──▶ app/data.py
    │         │
    │  ┌──────┴──────────────────────────────────────────────────────┐
    │  │ Post-processing:                                            │
    │  │ ├─ Calculate monotonic latency_ms                           │
    │  │ ├─ Inject X-Request-ID into response headers                │
    │  │ └─ Emit structured JSON log event to stdout (no secrets)    │
    │  └─────────────────────────────────────────────────────────────┘
    ▼
HTTP Response (with X-Request-ID header)
```

---

## Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Python | 3.11 | Runtime |
| pip | 23.x | Dependency installation |
| git | any | Version control |

---

## Virtual Environment Setup

```bash
# Create a virtual environment in the project directory
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate it (macOS / Linux)
source .venv/bin/activate
```

---

## Dependency Installation

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Copy `.env.example` to `.env` and adjust values as needed:

```bash
cp .env.example .env   # macOS / Linux
copy .env.example .env  # Windows
```

The application starts successfully without a `.env` file — all settings have safe defaults.

---

## Running the Application (Local Development)

From the `deployguard-demo/` directory (with the virtual environment activated):

```bash
uvicorn app.main:app --reload
```

The service starts on **http://127.0.0.1:8000** by default.

---

## Docker (Phase 2 Containerization)

The application includes a production-appropriate, security-hardened Docker setup.

### 1. Docker Prerequisites
- Docker Engine 20.10+ / Docker Desktop (Linux containers mode)
- Host port 8000 available (or specify another port when mapping, e.g. `-p 8080:8000`)

### 2. Image and Container Naming Conventions
- **Image Name:** `deployguard-demo:phase2`
- **Container Name:** `deployguard-phase2`

### 3. How to Build the Image
Build the image from the repository root:

```bash
docker build -t deployguard-demo:phase2 .
```

*Build details:*
- Uses `python:3.13-slim` base image for a minimal attack surface.
- Installs only runtime dependencies from `requirements-runtime.txt` (excluding pytest/httpx).
- Leverages layer caching: dependencies are copied and installed prior to copying application source code.
- Creates and runs as an unprivileged system user (`appuser`).
- Never copies local `.venv`, `.env`, test files, or version control into the image.

### 4. How to Run the Container
Run the container in detached mode with host port 8000 mapped to container port 8000:

```bash
docker run -d --name deployguard-phase2 -p 8000:8000 deployguard-demo:phase2
```

### 5. How to View Container Logs
Inspect real-time logs from Uvicorn running inside the container:

```bash
docker logs -f deployguard-phase2
```

### 6. How to Test the API Through the Container
With the container running, send requests to `http://127.0.0.1:8000`:

```bash
# Health check
curl http://127.0.0.1:8000/health

# List all orders
curl http://127.0.0.1:8000/api/v1/orders

# Retrieve a specific order
curl http://127.0.0.1:8000/api/v1/orders/ORD-0001

# Non-existent order (returns 404)
curl http://127.0.0.1:8000/api/v1/orders/ORD-9999

# Interactive Swagger UI (open in browser)
# http://127.0.0.1:8000/docs
```

### 7. How to Stop and Remove the Container
```bash
# Stop the running container
docker stop deployguard-phase3

# Remove the container
docker rm deployguard-phase3
```

---

## Structured Telemetry — Phase 3

Phase 3 introduces production-grade, application-level structured telemetry designed for high-resolution observability and machine learning feature extraction in future DeployGuard releases.

### 1. What Telemetry is Collected
Every HTTP request processed by the service triggers an event with the following fields:
- `timestamp`: Standardized ISO-8601 UTC timestamp with microsecond resolution (e.g. `2026-09-14T00:15:30.123456Z`).
- `level`: Log severity (`INFO` for 2xx/3xx, `WARNING` for 4xx, `ERROR` for 5xx/exceptions).
- `service`: Service identifier (`deployguard-demo`), configurable via `SERVICE_NAME`.
- `version`: Application semantic version (`0.1.0`), configurable via `APP_VERSION`.
- `request_id`: Correlation identifier (UUIDv4 or validated client `X-Request-ID`).
- `method`: HTTP method (`GET`, `POST`, etc.).
- `path`: Request URL path (e.g. `/api/v1/orders`).
- `status_code`: HTTP response status code (e.g. `200`, `404`, `500`).
- `latency_ms`: Duration of request execution in milliseconds measured via monotonic clock.
- `client_ip`: Originating client IP address (or `null` if unreachable).

**Privacy & Security:** Sensitive data (passwords, tokens, `Authorization` headers, `Cookie` headers, request payloads) are strictly excluded from all log records.

### 2. Why Request IDs are Required
Distributed systems and microservices execute hundreds of concurrent requests. An `X-Request-ID` ties client actions across frontend gateways, backend workloads, and eventual ML telemetry pipelines into a single traceable transaction. If a client provides a valid `X-Request-ID`, it is preserved; otherwise, a fresh UUIDv4 is generated and injected into response headers.

### 3. Why Latency is Measured with a Monotonic Clock
Request latency is calculated using `time.perf_counter()`, which relies on the operating system's monotonic clock. Unlike `time.time()` (wall-clock time), monotonic clocks cannot jump backward or drift due to NTP server synchronizations, providing precise, deterministic duration metrics.

### 4. Why Structured JSON Logs are Preferable to Text Logs
Traditional plain-text logs require brittle regex parsing that breaks whenever a log line format changes. Single-line JSON objects are machine-readable, schema-validatable, and immediately ingestible by stream processors, log forwarders (Fluentbit, Logstash), and data science pipelines (Pandas, Polars) without parsing ambiguity.

### 5. How This Telemetry Will Eventually Be Used by DeployGuard AI
In future phases, DeployGuard AI will analyze:
1. **Pre-deployment baselines vs. post-deployment canary performance:** Detecting anomalous latency spikes (`latency_ms`) or elevated error rates (`status_code >= 400`).
2. **Failure correlation:** Correlating specific git commit diffs with increased 5xx errors or tail latency degradation.
3. **Automated rollout decisions:** Driving real-time decisions (`CANARY`, `LIMITED`, `BLOCK`) based on live telemetry feeds.

### 6. How to View Logs Locally
Run the server locally or in Docker, then send requests:

```bash
# When running via Uvicorn locally:
uvicorn app.main:app
# (JSON logs stream directly to standard output)

# When running via Docker:
docker logs -f deployguard-phase3
```

### 7. Example Structured Log Entry
```json
{
  "timestamp": "2026-09-14T00:15:30.123456Z",
  "level": "INFO",
  "service": "deployguard-demo",
  "version": "0.1.0",
  "request_id": "6bbb9019-cd67-4ac3-94d6-37036f4c8e29",
  "method": "GET",
  "path": "/api/v1/orders",
  "status_code": 200,
  "latency_ms": 2.48,
  "client_ip": "127.0.0.1"
}
```

> **Note:** Centralized observability (OpenTelemetry, Prometheus, Grafana, Loki) and distributed tracing are out of scope for Phase 3 and will be integrated in subsequent phases.

---

## Deployment Change Analysis — Phase 4

Phase 4 implements the change-ingestion and feature-extraction engine. It analyzes proposed deployments *before* production rollout, extracting deterministic risk indicators from code changes and combining them with runtime telemetry metrics into a unified feature vector.

### Data Flow Architecture

```
Proposed Deployment Change (Structured Files or Unified Git Diff)
    │
    ▼
Change Ingestion & Git Diff Parser (app/change_analysis.py)
    │  ├─ Parses modified file paths
    │  └─ Calculates line additions (+) and deletions (-)
    │
    ▼
Deterministic Risk Feature Extraction
    │  ├─ Blast radius: files_changed, total_lines_changed
    │  ├─ Domain classification: high_risk_files_count (orders, payments, auth, db, docker)
    │  ├─ Architectural classification: config_files, app_files, test_files, doc_files
    │  └─ Risk flag: high_risk_area_touched (boolean)
    │
    ▼
Runtime Telemetry Integration (from Phase 3 Telemetry Windows)
    │  ├─ mean_latency_ms
    │  ├─ p99_latency_ms
    │  ├─ error_rate (HTTP 4xx/5xx failure ratio)
    │  └─ request_volume
    │
    ▼
Unified Machine-Readable Feature Vector (DeploymentRiskFeatures)
    │
    ▼
Future ML Risk Model (Phase 5: XGBoost / scikit-learn Classification)
    └─ Recommends: NORMAL | CANARY | LIMITED | BLOCK
```

### Why These Features Matter to a DevOps / SRE Engineer

| Feature | SRE / DevOps Operational Significance |
|---|---|
| `files_changed` & `total_lines_changed` | Measures the **blast radius** of a release. Small, frequent changes have significantly lower incident probability than monolithic multi-file diffs. |
| `high_risk_files_count` & `high_risk_area_touched` | Identifies whether the change touches business-critical domains (payments, checkout, auth, database schemas). Changes touching payment or auth logic require tighter gating (e.g. Canary) even with few lines modified. |
| `config_files_changed` | Configuration drift (database URLs, connection pools, timeouts, environment variables) is one of the leading root causes of production outages. Flagging config changes enables automated config validation. |
| `test_files_changed` | Compares application code churn against test suite updates. Large application changes with zero test changes indicate unverified releases with elevated risk. |
| `mean_latency_ms` & `p99_latency_ms` | Ingests pre-deployment canary or staging performance. Latency degradation indicates memory leaks, unindexed queries, or thread starvation. |
| `error_rate` | Directly reflects service health. An elevated error rate combined with high code churn strongly correlates with deployment failure. |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/api/v1/orders` | List all sample orders |
| `GET` | `/api/v1/orders/{order_id}` | Retrieve a single order |
| `POST` | `/api/v1/analyze-change` | Ingest deployment change & extract risk feature vector |
| `GET` | `/api/v1/analyze-change/samples` | List benchmark simulated changes (Change A, B, C, D) |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |
| `GET` | `/openapi.json` | Raw OpenAPI schema |

---

## Example API Responses

### `GET /health`

```json
{
  "status": "healthy"
}
```

### `GET /api/v1/orders`

```json
{
  "orders": [
    {
      "order_id": "ORD-0001",
      "customer_id": "CUST-101",
      "customer_name": "Alice Pemberton",
      "status": "delivered",
      "items": [
        {
          "product_id": "SKU-LAPTOP-PRO",
          "product_name": "LaptopPro 15\" (2025)",
          "quantity": 1,
          "unit_price": 1299.99
        },
        {
          "product_id": "SKU-MOUSE-WL",
          "product_name": "Wireless Ergonomic Mouse",
          "quantity": 1,
          "unit_price": 49.99
        }
      ],
      "total_amount": 1349.98,
      "created_at": "2026-08-01T09:15:00Z",
      "updated_at": "2026-08-05T14:30:00Z",
      "notes": null
    }
  ],
  "count": 5
}
```

### `GET /api/v1/orders/ORD-0001`

```json
{
  "order_id": "ORD-0001",
  "customer_id": "CUST-101",
  "customer_name": "Alice Pemberton",
  "status": "delivered",
  "items": [...],
  "total_amount": 1349.98,
  "created_at": "2026-08-01T09:15:00Z",
  "updated_at": "2026-08-05T14:30:00Z",
  "notes": null
}
```

### `GET /api/v1/orders/ORD-9999` (not found)

```json
{
  "detail": "Order 'ORD-9999' not found."
}
```

---

## Running the Tests

```bash
pytest -v
```

Expected output:

```
tests/test_health.py::test_health_returns_200                          PASSED
tests/test_health.py::test_health_returns_expected_json                PASSED
tests/test_orders.py::test_list_orders_returns_200                     PASSED
tests/test_orders.py::test_list_orders_response_structure              PASSED
tests/test_orders.py::test_list_orders_count_matches_data              PASSED
tests/test_orders.py::test_list_orders_items_have_required_fields      PASSED
tests/test_change_analysis.py::test_high_risk_path_detection PASSED
tests/test_change_analysis.py::test_config_file_detection PASSED
tests/test_change_analysis.py::test_test_file_detection PASSED
tests/test_change_analysis.py::test_doc_file_detection PASSED
tests/test_change_analysis.py::test_application_file_detection PASSED
tests/test_change_analysis.py::test_parse_git_diff_empty PASSED
tests/test_change_analysis.py::test_parse_git_diff_single_file PASSED
tests/test_change_analysis.py::test_parse_git_diff_multiple_files PASSED
tests/test_change_analysis.py::test_extract_features_empty_change PASSED
tests/test_change_analysis.py::test_extract_features_with_raw_diff PASSED
tests/test_change_analysis.py::test_extract_features_documentation_only_change PASSED
tests/test_change_analysis.py::test_extract_features_high_risk_payment_db PASSED
tests/test_change_analysis.py::test_extract_features_combined_with_telemetry PASSED
tests/test_change_analysis.py::test_api_analyze_change_success PASSED
tests/test_change_analysis.py::test_api_analyze_change_with_raw_diff PASSED
tests/test_change_analysis.py::test_api_analyze_change_invalid_payload PASSED
tests/test_change_analysis.py::test_api_list_sample_changes PASSED
tests/test_health.py::test_health_returns_200                          PASSED
tests/test_health.py::test_health_returns_expected_json                PASSED
tests/test_orders.py::test_list_orders_returns_200                     PASSED
tests/test_orders.py::test_list_orders_response_structure              PASSED
tests/test_orders.py::test_list_orders_count_matches_data              PASSED
tests/test_orders.py::test_list_orders_items_have_required_fields      PASSED
tests/test_orders.py::test_get_existing_order_returns_200              PASSED
tests/test_orders.py::test_get_existing_order_returns_correct_id       PASSED
tests/test_orders.py::test_get_existing_order_contains_items           PASSED
tests/test_orders.py::test_get_nonexistent_order_returns_404           PASSED
tests/test_orders.py::test_get_nonexistent_order_returns_error_detail  PASSED
tests/test_orders.py::test_every_sample_order_is_retrievable[ORD-0001] PASSED
tests/test_orders.py::test_every_sample_order_is_retrievable[ORD-0002] PASSED
tests/test_orders.py::test_every_sample_order_is_retrievable[ORD-0003] PASSED
tests/test_orders.py::test_every_sample_order_is_retrievable[ORD-0004] PASSED
tests/test_orders.py::test_every_sample_order_is_retrievable[ORD-0005] PASSED
tests/test_telemetry.py::test_request_receives_generated_request_id_when_none_supplied PASSED
tests/test_telemetry.py::test_request_preserves_supplied_safe_request_id PASSED
tests/test_telemetry.py::test_request_sanitizes_unsafe_request_id      PASSED
tests/test_telemetry.py::test_helper_get_or_generate_request_id_edge_cases PASSED
tests/test_telemetry.py::test_structured_log_emitted_on_successful_request PASSED
tests/test_telemetry.py::test_structured_log_emitted_on_404_request   PASSED
tests/test_telemetry.py::test_structured_log_contains_correct_correlation_id PASSED
tests/test_telemetry.py::test_no_sensitive_headers_or_secrets_logged   PASSED
tests/test_telemetry.py::test_json_formatter_produces_valid_json_string PASSED
```

---

## Current Limitations

| Area | Status |
|------|--------|
| Persistence | No database — data resets on every restart |
| Authentication | No auth — all endpoints are public |
| Pagination | Not implemented — all orders returned in one response |
| Mutation | No POST / PUT / DELETE endpoints for orders |
| Observability | Structured JSON telemetry & correlation IDs implemented (Phase 3); Prometheus/Grafana not implemented |
| Change Ingestion | Git diff parser & deterministic risk feature extraction implemented (Phase 4) |
| Containerisation | Docker containerization implemented (Phases 2 - 4); Kubernetes not implemented |
| CI/CD | No pipeline |
| ML / risk model | Feature vectors extracted; ML model (XGBoost / scikit-learn) not implemented yet |

---

## What Will Be Added in Future Phases

> **Important:** Nothing below is implemented yet. These are planned phases only.

| Phase | Description |
|-------|-------------|
| 5 | ML model (XGBoost / scikit-learn) for failure-probability prediction |
| 6 | DeployGuard decision engine (NORMAL / CANARY / LIMITED / BLOCK) |
| 7 | CI/CD integration (GitHub Actions or Jenkins) |
| 8 | Container orchestration (Kubernetes) |
| 9 | Observability stack (Prometheus, Grafana) |
| 10 | Dashboard UI |

---

## Technologies Used (Phases 1 - 4)

| Technology | Version | Reason |
|------------|---------|--------|
| Python | 3.13 | Primary language runtime |
| FastAPI | 0.115.x | ASGI web framework with built-in OpenAPI generation |
| Uvicorn | 0.34.x | ASGI server |
| Pydantic v2 | 2.11.x | Data validation, schemas, and feature vector modeling |
| pydantic-settings | 2.9.x | Environment-based configuration |
| Docker | 20.10+ | Container engine (Phases 2 - 4) |
| Python standard logging + json | 3.13 | Structured single-line JSON log streaming (Phase 3) |
| Pure-Python Unified Diff Parser | Custom | Lightweight, zero-dependency git diff extraction (Phase 4) |
| httpx | 0.28.x | HTTP client backing FastAPI's TestClient (local dev only) |
| pytest | 8.3.x | Test runner (local dev only) |

**Not used and not planned for this phase:** Kubernetes, Prometheus, Grafana, PostgreSQL, Redis, Kafka, React, AWS, Terraform, XGBoost, scikit-learn, Jenkins, GitHub Actions, OpenTelemetry, Loki.
