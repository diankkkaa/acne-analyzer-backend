# Acne Analyzer API

FastAPI backend for acne image classification (PyTorch prototypical network), personalized Ukrainian recommendations (local LLM + RAG), Microsoft SQL Server storage, and AWS S3 image uploads.

## Setup

1. **Python 3.10+**, **ODBC Driver 17 for SQL Server** ([download](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)).

2. Copy environment file and edit values:

```bash
copy .env.example .env
```

3. **ML support set** (required for inference): create a folder (default `support_set/`, see `SUPPORT_SET_PATH` in `.env`) with this layout:

```
support_set/
├── acne0/   # 5 reference images (jpg/png/…)
├── acne1/
├── acne2/
└── acne3/
```

4. Place **`best_model.pt`** (or your checkpoint path set in `MODEL_PATH`) at the configured path.

5. Place the **knowledge base PDF** at `RAG_PDF_PATH` (e.g. `knowledge_base.pdf`).

6. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(PyTorch: for GPU or a specific platform, follow [pytorch.org](https://pytorch.org) install instructions.)

## Run

From the `acne-analyzer-backend` directory (repository root of this package):

```bash
uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

On startup the app loads the ML model + prototypes and the RAG PDF chunks. Check readiness with **GET /health** (`model_loaded`, `kb_loaded`).

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness and ML/RAG init flags: `status`, `model_loaded`, `kb_loaded` |
| POST | `/auth/register` | Register user (JSON: `email`, `password`); returns user profile |
| POST | `/auth/login` | OAuth2 form: `username` (email), `password`; returns JWT `access_token` |
| GET | `/user/me` | Current user profile (**JWT**) |
| POST | `/questionnaire` | Create questionnaire (**JWT**); returns saved record |
| PUT | `/questionnaire` | Update latest questionnaire for user (**JWT**) |
| GET | `/questionnaire` | Get latest questionnaire (**JWT**) |
| POST | `/upload-image` | Upload image to S3 + DB (**JWT**); returns `image_id`, `file_path` |
| POST | `/analyze` | Multipart: image file, optional `questionnaire_id` form field (**JWT**); full ML + RAG pipeline; returns class, confidence, recommendation, `prediction_id` |
| DELETE | `/analysis/{prediction_id}` | Delete prediction (+ optional `delete_s3` query); owner only (**JWT**) |
| GET | `/history` | List predictions for user with image URL, class, confidence, date, 100-char recommendation preview (**JWT**) |
| GET | `/analysis/{prediction_id}` | Full analysis detail: image, class, confidence, questionnaire, full recommendation (**JWT**) |

Protected routes expect header: `Authorization: Bearer <access_token>`.

## `.env.example`

```
DATABASE_URL=mssql+aioodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server
SECRET_KEY=change_me
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_BUCKET_NAME=acne-analyzer-photos
AWS_REGION=eu-central-1
LLM_API_URL=http://localhost:1234/v1/chat/completions
LLM_MODEL=local-model
RAG_PDF_PATH=knowledge_base.pdf
MODEL_PATH=best_model.pt
SUPPORT_SET_PATH=support_set/
```

## Project layout

- `app/main.py` — app, CORS, lifespan (ML + RAG init), global error handler, health
- `app/config.py` — settings (`pydantic-settings`)
- `app/database.py` — async SQLAlchemy session
- `app/models/` — ORM models (MSSQL)
- `app/schemas/` — Pydantic DTOs
- `app/routers/` — HTTP routes
- `app/services/` — auth, S3, ML, RAG, DB helpers

## CORS

Development middleware allows **all origins** (`*`). Restrict `allow_origins` in production.
