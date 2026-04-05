# AI-Powered Document Analysis API

FastAPI service for robust document analysis with:

- OCR preprocessing for noisy images (blur, skew, contrast correction)
- Hierarchical summarization with truncation-safe fallback chain
- Fuzzy, normalized entity extraction with alias handling
- Ensemble sentiment scoring with explanations
- Per-component confidence scoring + overall confidence
- API key authentication (configurable)
- Rate limiting (configurable, sliding window)
- Retrieval endpoint for processed document results
- Metrics endpoint for processing stats
- Dockerized deployment with PostgreSQL + Redis

## Project Structure

- `app/services/extraction.py` - OCR + file-type text extraction
- `app/services/nlp.py` - summarization, NER, fuzzy normalization, sentiment ensemble
- `app/services/confidence.py` - confidence scoring logic
- `app/services/security.py` - API key validation + rate limiting
- `app/services/metrics.py` - processing metrics collection
- `app/routes/document.py` - analysis endpoint + response composition
- `tests/` - API and NLP edge-case tests

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Analyze:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/analyze" \
  -F "file=@sample.txt"
```

Authenticated analyze (when `REQUIRE_API_KEY=true`):

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/analyze" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@sample.txt"
```

Dependency preflight:

```bash
curl http://127.0.0.1:8000/health/dependencies
```

Metrics:

```bash
curl http://127.0.0.1:8000/api/v1/metrics
```

Get result by ID:

```bash
curl http://127.0.0.1:8000/api/v1/documents/results/<document_id>
```

## Docker

```bash
docker compose up --build
```

## Notes on Model Strategy

By default, `USE_HEAVY_MODELS=false` to keep startup lightweight. In this mode, lexical + regex + spaCy fallback paths keep the pipeline robust and testable.

Set `USE_HEAVY_MODELS=true` to enable transformer-based primary models for summarization, NER, and ensemble sentiment voting.

## Compliance Notes

- Supported formats: PDF, DOCX, image OCR, TXT/MD.
- Layout preservation improvements:
  - PDF pages and extracted tables are marked in text output.
  - DOCX headings and table rows are preserved with structure markers.
- See `DEPLOYMENT.md` for cloud deployment steps and submission checklist.
