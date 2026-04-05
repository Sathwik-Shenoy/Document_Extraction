from fastapi import FastAPI

from app.core.config import settings
from app.routes.document import router as document_router
from app.services.extraction import get_extraction_dependency_status
from app.services.metrics import get_metrics

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(document_router)


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "health": "/health",
            "api_health": "/api/v1/health",
            "dependency_health": "/health/dependencies",
            "docs": "/docs",
            "analyze": "/api/v1/documents/analyze",
            "upload": "/api/v1/documents/upload",
            "document_result": "/api/v1/documents/results/{document_id}",
            "metrics": "/api/v1/metrics",
            "auth_status": "/api/v1/auth-status",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.get("/api/v1/health")
def api_health():
    return {"status": "healthy", "service": settings.app_name, "version": settings.app_version}


@app.get("/health/dependencies")
def dependency_health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "extractors": get_extraction_dependency_status(),
    }


@app.get("/api/v1/metrics")
def metrics():
    return get_metrics()


@app.get("/api/v1/auth-status")
def auth_status():
    return {
        "require_api_key": settings.require_api_key,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
    }
