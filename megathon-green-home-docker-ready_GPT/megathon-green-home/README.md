# MEGATHON X: SINGULARITY — Green Home Score & Retrofit Engine

Docker-ready Flask API + interactive web dashboard for Lloyds Problem Statement 2.

## What is implemented

- Weighted Green Home Score (fixed weights from the brief), transparent component scores and classification.
- Energy-intensity calculation based on annualised metered electricity + gas per square metre.
- Property-level train/validation/test split respected by `split` and `property_id`.
- Feature engineering from property and monthly-energy tables.
- Trainable baseline regressors for improvement potential, per-fuel savings and indicative value impact; evaluation on the supplied held-out test split.
- Rule-based retrofit feasibility/ranking, with insulation-first sequencing and per-fuel scenario accounting.
- Carbon calculation from separately documented electricity and gas factors.
- Scenario (“what-if”) endpoint reruns deterministic calculations before any optional LLM explanation. No LLM is included by default; the app remains usable without an API key.
- Browser UI with property search, score gauge, retrofit cards, per-fuel savings, carbon, value range and scenario controls.
- Upload endpoint for compatible, unseen synthetic property/energy CSVs. Required columns are validated; optional features receive documented defaults. Uploads are not automatically treated as training truth.

## Important data caveat

The supplied sample has 1,500 properties, 18,000 monthly-energy rows and 1,500 reference-label rows. The supplied labels are reference estimates; they are not proof of real-world causal retrofit impacts. The app reports model outputs as estimates and the evaluation report records this limitation. Before final submission, replace the illustrative emissions factors in `.env.example` with a suitable current source, record its URL/date/units, and regenerate the report.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train.py
python run.py
```

Open `http://localhost:8000`.

## Docker: build → image → container

```bash
docker build -t megathon-green-home:1.0 .
docker run --name green-home -p 8000:8000 megathon-green-home:1.0
```

Then open `http://localhost:8000`. Health check: `http://localhost:8000/api/health`.

To persist trained models/reports outside the container, mount directories:

```bash
docker run --name green-home -p 8000:8000 \
  -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" \
  megathon-green-home:1.0
```

The container uses a non-root user and Gunicorn. Do not expose Flask debug mode in production.

## API

- `GET /api/health`
- `GET /api/summary`
- `GET /api/properties?limit=100`
- `GET /api/property/<property_id>`
- `POST /api/what-if` JSON: `{ "property_id": "P00001", "retrofits": ["loft_insulation", "solar_panels"] }`
- `POST /api/upload-dataset` multipart CSVs: `properties` (required), `energy_monthly` (optional). Validates and loads an exploratory dataset into memory for preview; does not retrain automatically.
- `POST /api/retrain` retrains baseline models from the configured dataset folder; use only trusted files.

## Dataset schema

Required `properties.csv` columns: `property_id`, `floor_area`, `energy_rating`, `heating`, `insulation`, `glazing`, `annual_electricity_kwh`, `annual_gas_kwh`.

Optional: `property_type`, `age`, `bedrooms`, `solar`, `smart_controls`, `region`, `value_current_gbp`, `has_loft`, `cavity_walls`, `wall_insulated`, `draught_proofed`, `heat_pump_space`, `solar_roof_suitable`, `space_heat_kwh`, `hot_water_kwh`, `base_electricity_kwh`, `other_gas_kwh`, `pv_offset_kwh`, `split`.

`energy_monthly.csv` columns: `property_id`, `year`, `month`, `electricity_kwh`, `gas_kwh` (optional `split`). If monthly data exists, it is summed consistently per property; no missing month is silently treated as zero.

## Reproducibility / validation

```bash
python train.py
cat reports/evaluation.json
pytest -q
```

The training pipeline trains only on rows marked `train`, selects validation metrics for comparison, and reports held-out test metrics separately. Splits are property-level. For unseen datasets without reference labels, the app can score/recommend, but supervised model quality cannot be measured until valid reference labels are supplied.

## Push to GitHub

Create an empty repository, then:

```bash
git init
git add .
git commit -m "Build Docker-ready Green Home retrofit engine"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
git push -u origin main
```

Do not commit secrets, private/customer data, or real production credentials.
