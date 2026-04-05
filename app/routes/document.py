from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile

from app.models.schemas import AnalyzeResponse, ConfidenceScores
from app.services.confidence import (
    entities_confidence,
    extraction_confidence,
    overall_confidence,
    sentiment_confidence,
    summary_confidence,
)
from app.services.extraction import extract_text
from app.services.metrics import record_processing
from app.services.nlp import MODEL_REGISTRY, analyze_sentiment, extract_entities, summarize_hierarchical
from app.services.security import client_key_from_request, enforce_rate_limit, verify_api_key

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
_DOCUMENT_RESULTS: Dict[str, Dict[str, Any]] = {}


@router.post("/analyze", response_model=AnalyzeResponse)
@router.post("/upload", response_model=AnalyzeResponse)
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    _auth: None = Depends(verify_api_key),
) -> AnalyzeResponse:
    started = time.perf_counter()
    client_id = client_key_from_request(request.client.host if request.client else None, authorization)
    enforce_rate_limit(client_id)

    raw_bytes = await file.read()
    if not raw_bytes:
        record_processing(int((time.perf_counter() - started) * 1000), success=False)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    ext = extract_text(raw_bytes, filename=file.filename or "unknown", content_type=file.content_type)
    if not ext.text.strip():
        dependency_hints = {
            "pdfplumber_not_available": "Install into the runtime interpreter: python -m pip install pdfplumber",
            "python_docx_not_available": "Install into the runtime interpreter: python -m pip install python-docx",
            "pytesseract_not_available": "Install into the runtime interpreter: python -m pip install pytesseract",
            "opencv_not_available": "Install into the runtime interpreter: python -m pip install opencv-python-headless",
        }
        hints = [dependency_hints[w] for w in ext.warnings if w in dependency_hints]
        record_processing(int((time.perf_counter() - started) * 1000), success=False)
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not extract text",
                "warnings": ext.warnings,
                "hints": hints,
                "dependency_health": "/health/dependencies",
            },
        )

    summary, summary_meta = summarize_hierarchical(ext.text, ext.file_type)
    entities, relationships, entity_meta = extract_entities(ext.text)
    sentiment, sent_meta = analyze_sentiment(ext.text)

    extraction_score = extraction_confidence(ext.text, ocr_warnings=len(ext.warnings), unreadable_tokens=ext.unreadable_tokens)
    summary_score = summary_confidence(summary)

    entity_scores = [float(e.get("confidence", 0.6)) for e in entities]
    uniq_ratio = min(1.0, (len(entities) / max(1, sum(e.get("frequency", 1) for e in entities))))
    entities_score = entities_confidence(entity_scores, unique_ratio=uniq_ratio)

    sentiment_score = sentiment_confidence(sentiment["score"], sentiment.get("agreement", 0.5))
    overall = overall_confidence([extraction_score, summary_score, entities_score, sentiment_score])

    elapsed = int((time.perf_counter() - started) * 1000)
    result = AnalyzeResponse(
        document_id=str(uuid.uuid4()),
        file_type=ext.file_type,
        extracted_text=ext.text,
        summary=summary,
        entities=entities,
        relationships=relationships,
        sentiment={
            "label": sentiment["label"],
            "score": sentiment["score"],
            "explanation": sentiment["explanation"],
            "votes": sentiment["votes"],
        },
        confidence_scores=ConfidenceScores(
            extraction=round(extraction_score, 3),
            summary=round(summary_score, 3),
            entities=round(entities_score, 3),
            sentiment=round(sentiment_score, 3),
            overall=round(overall, 3),
        ),
        model_versions={
            "summarization": f"{MODEL_REGISTRY.summarization['primary']} (fallback: {MODEL_REGISTRY.summarization['fallback']})",
            "ner": f"{MODEL_REGISTRY.ner['primary']} (fallback: spaCy + regex)",
            "sentiment": "ensemble [distilbert + bert-mnli + roberta-mnli]",
        },
        metadata={
            "filename": file.filename,
            "content_type": file.content_type,
            "processing_time_ms": elapsed,
            "warnings": ext.warnings,
            "ocr": ext.metadata,
            "strategies": {
                "summary": summary_meta,
                "entities": entity_meta,
                "sentiment": sent_meta,
            },
        },
    )
    _DOCUMENT_RESULTS[result.document_id] = result.model_dump()
    record_processing(elapsed, success=True)
    return result


@router.get("/results/{document_id}")
def get_document_result(
    document_id: str,
    _auth: None = Depends(verify_api_key),
):
    item = _DOCUMENT_RESULTS.get(document_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Document result not found")
    return item
