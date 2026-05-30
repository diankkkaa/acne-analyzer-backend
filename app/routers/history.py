from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import get_current_user
from app.models.db_models import Prediction
from app.models.db_models import User
from app.schemas.schemas import AnalysisDetailResponse, HistoryItem, HistoryResponse, QuestionnaireResponse

router = APIRouter(tags=["history"])

@router.get("/history", response_model=HistoryResponse)
async def list_history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HistoryResponse:
    stmt = (
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .options(
            selectinload(Prediction.image),
            selectinload(Prediction.recommendations),
        )
        .order_by(Prediction.created_at.desc())
    )
    result = await db.execute(stmt)
    preds = result.scalars().unique().all()

    items: list[HistoryItem] = []
    for p in preds:
        recs = sorted(p.recommendations, key=lambda r: r.created_at, reverse=True)
        full_text = recs[0].recommendation_text if recs else None
        preview = full_text[:100] if full_text else None
        file_url = p.image.file_url if p.image else ""
        items.append(
            HistoryItem(
                prediction_id=p.id,
                image_id=p.image_id,
                file_url=file_url,
                predicted_class=p.predicted_class,
                confidence=p.confidence,
                created_at=p.created_at,
                recommendation_preview=preview,
            ),
        )
    return HistoryResponse(items=items)


@router.get("/analysis/{prediction_id}", response_model=AnalysisDetailResponse)
async def get_analysis_detail(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalysisDetailResponse:
    stmt = (
        select(Prediction)
        .where(Prediction.id == prediction_id)
        .options(
            selectinload(Prediction.image),
            selectinload(Prediction.questionnaire),
            selectinload(Prediction.recommendations),
        )
    )
    res = await db.execute(stmt)
    pred = res.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    if pred.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")

    questionnaire_out = None
    if pred.questionnaire is not None:
        questionnaire_out = QuestionnaireResponse.model_validate(pred.questionnaire)

    recs = sorted(pred.recommendations, key=lambda r: r.created_at, reverse=True)
    rec_text = recs[0].recommendation_text if recs else None

    return AnalysisDetailResponse(
        prediction_id=pred.id,
        image_id=pred.image_id,
        image_url=pred.image.file_url if pred.image else "",
        predicted_class=pred.predicted_class,
        confidence=pred.confidence,
        created_at=pred.created_at,
        questionnaire=questionnaire_out,
        recommendation_text=rec_text,
    )
