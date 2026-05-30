from __future__ import annotations

from decimal import Decimal
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from PIL import Image
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.db_models import Image as ImageRow
from app.models.db_models import Prediction
from app.models.db_models import QuestionnaireResponse as QuestionnaireRow
from app.models.db_models import Recommendation
from app.models.db_models import User
from app.schemas.schemas import AnalyzePipelineResponse, ImageCreatedResponse
from app.services import face_service, ml_service, rag_service, s3_service

router = APIRouter(tags=["analyze"])

_FACE_REJECT_DETAIL = (
    "На зображенні не виявлено обличчя людини. "
    "Будь ласка, завантажте фото обличчя."
)

_QUESTIONNAIRE_FIELDS: tuple[str, ...] = (
    "has_family_history",
    "has_hormonal_acne",
    "uses_comedogenic_products",
    "hair_products_trigger",
    "stress_trigger",
    "sleep_quality",
    "is_smoking",
    "water_intake",
    "dairy_consumption",
    "high_glycemic_diet",
    "food_triggers",
    "skin_type",
    "oily_skin",
    "jawline_acne",
    "has_demodex",
    "vitamin_d_level",
    "dairy_sensitivity",
)


def _questionnaire_to_dict(row: QuestionnaireRow | None) -> dict | None:
    if row is None:
        return None
    return {k: getattr(row, k) for k in _QUESTIONNAIRE_FIELDS}


async def _read_and_validate_face(file: UploadFile) -> tuple[Image.Image, bytes]:
    raw = await file.read()
    pil = Image.open(BytesIO(raw)).convert("RGB")
    if not face_service.is_human_face(pil):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_FACE_REJECT_DETAIL,
        )
    return pil, raw


def _bytes_to_upload(raw: bytes, filename: str | None) -> UploadFile:
    return UploadFile(file=BytesIO(raw), filename=filename or "image.jpg")


@router.post("/upload-image", response_model=ImageCreatedResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ImageCreatedResponse:
    _, raw = await _read_and_validate_face(file)
    url = s3_service.upload_image(_bytes_to_upload(raw, file.filename))
    row = ImageRow(user_id=current_user.id, file_url=url)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return ImageCreatedResponse(image_id=row.id, file_path=url)


@router.post("/analyze", response_model=AnalyzePipelineResponse)
async def analyze(
    file: UploadFile = File(...),
    questionnaire_id: int | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalyzePipelineResponse:
    q_dict: dict | None = None
    if questionnaire_id is not None:
        qr = await db.execute(
            select(QuestionnaireRow).where(QuestionnaireRow.id == questionnaire_id),
        )
        q_row = qr.scalar_one_or_none()
        if q_row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Questionnaire not found",
            )
        if q_row.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Questionnaire belongs to another user",
            )
        q_dict = _questionnaire_to_dict(q_row)

    pil, raw = await _read_and_validate_face(file)
    upload_clone = _bytes_to_upload(raw, file.filename or "analyze.jpg")
    file_url = s3_service.upload_image(upload_clone)

    img_row = ImageRow(user_id=current_user.id, file_url=file_url)
    db.add(img_row)
    await db.flush()

    try:
        ml_out = ml_service.predict(pil)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e) or "ML model not ready",
        ) from e

    conf = Decimal(f"{float(ml_out['confidence']):.4f}")
    pred_row = Prediction(
        user_id=current_user.id,
        image_id=img_row.id,
        questionnaire_id=questionnaire_id,
        predicted_class=ml_out["predicted_class"],
        confidence=conf,
    )
    db.add(pred_row)
    await db.flush()

    try:
        rec_text, prompt_used = rag_service.generate_recommendation(
            ml_out["predicted_class"],
            float(ml_out["confidence"]),
            q_dict,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e) or "Knowledge base not loaded",
        ) from e

    rec_row = Recommendation(
        prediction_id=pred_row.id,
        user_id=current_user.id,
        prompt_used=prompt_used,
        recommendation_text=rec_text,
        model_used=settings.LLM_MODEL,
    )
    db.add(rec_row)
    await db.commit()
    await db.refresh(pred_row)

    return AnalyzePipelineResponse(
        predicted_class=pred_row.predicted_class,
        confidence=pred_row.confidence,
        recommendation_text=rec_text,
        prediction_id=pred_row.id,
    )


@router.delete("/analysis/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    prediction_id: int,
    delete_s3: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    stmt = (
        select(Prediction)
        .where(Prediction.id == prediction_id)
        .options(selectinload(Prediction.image))
    )
    res = await db.execute(stmt)
    pred = res.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    if pred.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")

    file_url = pred.image.file_url if pred.image else None
    image_id = pred.image_id

    await db.execute(sa_delete(Recommendation).where(Recommendation.prediction_id == prediction_id))
    await db.execute(sa_delete(Prediction).where(Prediction.id == prediction_id))
    await db.commit()

    if delete_s3 and file_url:
        try:
            s3_service.delete_image(file_url)
        except (ValueError, RuntimeError):
            pass
        await db.execute(sa_delete(ImageRow).where(ImageRow.id == image_id))
        await db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
