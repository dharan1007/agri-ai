# Agri AI — Agricultural Decision Support Platform

Agri AI is a multi-service agricultural assistant that combines crop planning, plant-disease inference, government-scheme discovery, a web interface, model-training utilities, and local infrastructure in one repository.

This README describes the implementation that is actually present in the repository. Large binary image datasets under `agri-assistant/datasets/` are training/evaluation assets; the application architecture is defined by the source, configuration, database, and model files around them.

## What is implemented

The platform currently has three independently deployable FastAPI services plus a Next.js web application:

1. **Disease Predictor** — accepts plant images and returns ranked disease predictions.
2. **Crop Calendar** — combines crop-specific agronomic windows with live Open-Meteo weather data to estimate sowing and harvest windows.
3. **Schemes Recommender** — filters and ranks government/agricultural schemes from PostgreSQL using farmer, crop, and location inputs.
4. **Web App** — provides browser-facing flows for disease prediction, crop-calendar planning, and scheme discovery through API proxy routes.

The repository also includes disease-model training/evaluation code, scheme-ingestion utilities, raw image datasets, Dockerfiles, PostgreSQL initialization, Redis, MinIO, and Docker Compose orchestration.

## Repository architecture

```text
agri-ai/
└── agri-assistant/
    ├── apps/
    │   └── web/                     # Next.js web interface + API proxy routes
    ├── data/
    │   ├── models/disease/          # disease training/evaluation utilities
    │   └── schemes/                 # government-scheme ingestion/sample data
    ├── datasets/
    │   └── raw/                     # plant/disease image datasets
    ├── infra/
    │   └── compose/
    │       ├── docker-compose.yml   # local multi-service stack
    │       └── initdb.sql           # PostgreSQL schema/seed initialization
    └── services/
        ├── crop-calendar/
        ├── disease-predictor/
        └── schemes-recommender/
```

## 1. Disease Predictor

Location: `agri-assistant/services/disease-predictor/`

The disease service is a FastAPI application with two inference backends.

### Supported inference backends

**ONNX backend**

- Used first when `MODEL_LOCAL_PATH` exists.
- Runs through `onnxruntime` with the CPU execution provider.
- Images are converted to RGB, resized to `224x224`, normalized with ImageNet-style mean/std values, and transformed to NCHW format.

**JSON linear backend**

- Used when an ONNX model is unavailable and `MODEL_JSON_PATH` exists.
- Extracts nine deterministic image features:
  - mean R/G/B
  - standard deviation R/G/B
  - mean grayscale value
  - mean Sobel gradient magnitude
  - grayscale variance
- Applies the stored `W` and `b` parameters in `model.json` and converts logits to probabilities with softmax.

The JSON model makes the service runnable without requiring a full CNN artifact, while ONNX provides the path for a conventional exported vision model.

### Disease API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Reports service health and selected inference backend |
| `GET` | `/labels` | Returns the configured disease/class labels |
| `POST` | `/predict` | Accepts a JPEG/PNG upload and returns top-1 plus top-k predictions |

Important environment variables:

```text
MODEL_LOCAL_PATH   # default: /app/model.onnx
MODEL_JSON_PATH    # default: /app/model.json
LABELS_PATH        # default: /app/labels.json
TOPK               # default: 3
```

The service rejects non-JPEG/PNG uploads and exposes the backend/model version in its prediction response.

## 2. Crop Calendar

Location: `agri-assistant/services/crop-calendar/`

The crop-calendar service is a FastAPI application built around live weather signals rather than a static calendar alone.

Currently supported crops:

- wheat
- rice
- maize

For each crop, the code defines a baseline sowing window and maturity duration. It then requests daily minimum temperature, maximum temperature, and precipitation from the free Open-Meteo API using the supplied latitude and longitude.

The planner evaluates crop-specific temperature/rainfall conditions and selects the earliest favorable run in the weather window. When favorable conditions are not found, it falls back to the crop's baseline calendar for the current year.

### Crop Calendar API

```http
GET /calendar?lat=<latitude>&lon=<longitude>&crop=wheat
```

The response includes:

- selected crop
- sowing window
- estimated harvest window
- weather summary
- human-readable narrative

Current logic is explicitly heuristic. It should be treated as a planning aid rather than agronomic or regulatory advice.

## 3. Schemes Recommender

Location: `agri-assistant/services/schemes-recommender/`

This FastAPI service reads active schemes from PostgreSQL and ranks them against a farmer profile.

Supported filters/profile inputs include:

- country
- state
- district
- crop
- land size in hectares
- annual income
- farmer category
- irrigation type

The database row includes geography, implementing agency, benefit type/details, structured eligibility data, keywords, URL, and active state. A runtime score is calculated using factors such as category match, income-cap eligibility, irrigation keywords, land-holding constraints, and high-impact benefit types such as subsidy/cash/insurance.

### Scheme API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Basic service health |
| `GET` | `/schemes` | Filter and rank active schemes |

`DATABASE_URL` defaults to the local Docker Compose PostgreSQL instance when no override is supplied.

## Web application

Location: `agri-assistant/apps/web/`

The Next.js app contains pages and API proxy routes for the three domain services:

```text
pages/
├── index.js
├── predict.js
├── calendar.js
├── schemes.js
└── api/
    ├── ping.js
    ├── predict.js
    ├── calendar.js
    └── schemes.js
```

Docker Compose exposes service URLs to the web application through:

```text
NEXT_PUBLIC_PREDICT_URL
NEXT_PUBLIC_SCHEMES_URL
NEXT_PUBLIC_CALENDAR_URL
```

## Local infrastructure

The checked-in Docker Compose stack provisions:

| Service | Local port | Role |
|---|---:|---|
| PostgreSQL 16 | `5433` | structured application/scheme data |
| MinIO | `9000` / console `9001` | S3-compatible object storage |
| Redis 7 | `6379` | cache/coordination infrastructure |
| Disease Predictor | `8001` | image inference API |
| Schemes Recommender | `8002` | scheme matching API |
| Crop Calendar | `8003` | weather/crop planning API |
| Web | `3000` | browser application |

### Start the full stack

```bash
git clone https://github.com/dharan1007/agri-ai.git
cd agri-ai/agri-assistant/infra/compose
docker compose up --build
```

Then open:

```text
http://localhost:3000
```

The Compose file currently uses development credentials for PostgreSQL and MinIO. Replace them before any non-local deployment.

## Model/data workflow

The repository separates runtime inference from model development:

```text
data/models/disease/
├── train.py
├── eval_json.py
├── eval_onnx.py
└── requirements.txt
```

The raw dataset is stored below `datasets/raw/`. Do not infer production model quality from the presence or size of the dataset alone; validate trained artifacts against held-out data and the evaluation utilities before release.

### Dataset provenance and reuse

The repository contains raw plant-disease images under `agri-assistant/datasets/raw/`, but it does not document their upstream source, attribution, or applicable terms. Treat permission to redistribute these files or use them for third-party model training as unresolved until the source and terms are documented. Dataset rights may differ from the code license.

The `data/schemes/sample_schemes.csv` file is sample input, not verified live government guidance. Do not use its rows for real recommendations until each scheme's current eligibility, benefits, and source are checked with the issuing agency.

Scheme data has its own ingestion path under `data/schemes/` and is persisted into the PostgreSQL schema initialized by `infra/compose/initdb.sql`.

## Development notes

- The FastAPI services are intentionally independent and can be run/deployed separately.
- The crop-calendar weather integration requires outbound access to Open-Meteo.
- Disease inference requires `labels.json` and at least one valid model artifact (`model.onnx` or `model.json`).
- The scheme service requires a reachable PostgreSQL database containing the `gov_schemes` table.
- Current CORS configuration in the Python services is oriented around a local web client on `http://localhost:3000`; production origins should be configured explicitly.
- The checked-in Compose credentials are development defaults and are not suitable as production secrets.

## Current implementation status

This repository contains working service code, infrastructure definitions, model tooling, datasets, and a web integration surface. However, production readiness depends on the quality and freshness of scheme data, validated disease-model accuracy, production secret management, deployment-specific CORS/network configuration, monitoring, and domain validation of agronomic recommendations.

## License

No root license file was found during the repository audit. Add an explicit license before distributing or accepting external contributions.