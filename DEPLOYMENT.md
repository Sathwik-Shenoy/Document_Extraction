# Deployment Guide

## Local Docker Validation

```bash
docker compose up --build
curl -sS http://127.0.0.1:8000/api/v1/health
curl -sS http://127.0.0.1:8000/health/dependencies
pytest -q
```

## Render / Railway Steps

1. Push the repository to GitHub and connect that repo to a new Render/Railway web service.
2. Set the start command exactly to:
   - `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Configure environment variables:
   - `ENVIRONMENT=production`
   - `REQUIRE_API_KEY=true`
   - `API_KEY=<strong-random-key>`
   - `RATE_LIMIT_PER_MINUTE=10`
   - `USE_HEAVY_MODELS=false` for a faster, safer deploy
4. Add managed Postgres and Redis only if your deployment setup requires them, then point `DATABASE_URL` and `REDIS_URL` at the managed services.
5. After deploy, verify the service in this order:
   - `GET /api/v1/health`
   - `GET /health/dependencies`
   - `GET /api/v1/metrics`
   - `POST /api/v1/documents/analyze` with `Authorization: Bearer <API_KEY>`
6. If the upload request fails, check that the runtime has the PDF/DOCX/OCR dependencies installed in the same environment as Uvicorn.

## Public Submission Checklist

- Public URL is reachable and returns `200` on `/api/v1/health`.
- API key is shared with evaluators.
- GitHub repository URL is shared.
- `/health/dependencies` reports all extractors ready.
- `/api/v1/metrics` responds successfully after at least one request.
- Run `pytest -q` before final submission.
