from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import analyze, auth, history, questionnaire, user
from app.services import face_service, ml_service, rag_service

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_loaded = False
    app.state.kb_loaded = False

    try:
        ml_service.load_model()
        ml_service.build_prototypes()
        app.state.model_loaded = True
        logger.info("✅ ML model loaded")
    except Exception:
        logger.exception("ML initialization failed")

    try:
        face_service.load_detector()
        logger.info("✅ Face detector loaded")
    except Exception:
        logger.exception("Face detector initialization failed")

    try:
        rag_service.load_knowledge_base()
        app.state.kb_loaded = True
        logger.info("✅ Knowledge base loaded")
    except Exception:
        logger.exception("RAG knowledge base initialization failed")

    yield


app = FastAPI(
    title="Acne Analyzer API",
    version="1.0.0",
    description="Backend for acne classification and personalized recommendations",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(analyze.router)
app.include_router(questionnaire.router)
app.include_router(history.router)
app.include_router(user.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or "Internal server error"},
    )


@app.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": bool(getattr(request.app.state, "model_loaded", False)),
        "kb_loaded": bool(getattr(request.app.state, "kb_loaded", False)),
    }
