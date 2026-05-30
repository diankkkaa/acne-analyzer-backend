from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.db_models import QuestionnaireResponse as QuestionnaireRow
from app.models.db_models import User
from app.schemas.schemas import QuestionnaireCreate, QuestionnaireResponse, QuestionnaireUpdate

router = APIRouter(prefix="/questionnaire", tags=["questionnaire"])


async def _latest_questionnaire(
    db: AsyncSession,
    user_id: int,
) -> QuestionnaireRow | None:
    stmt = (
        select(QuestionnaireRow)
        .where(QuestionnaireRow.user_id == user_id)
        .order_by(QuestionnaireRow.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


@router.post("", response_model=QuestionnaireResponse, status_code=status.HTTP_201_CREATED)
async def create_questionnaire(
    body: QuestionnaireCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QuestionnaireRow:
    row = QuestionnaireRow(user_id=current_user.id, **body.model_dump())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.put("", response_model=QuestionnaireResponse)
async def update_latest_questionnaire(
    body: QuestionnaireUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QuestionnaireRow:
    row = await _latest_questionnaire(db, current_user.id)
    payload = body.model_dump(exclude_unset=True)
    if row is None:
        row = QuestionnaireRow(user_id=current_user.id, **payload)
        db.add(row)
    else:
        for key, value in payload.items():
            setattr(row, key, value)
    await db.commit()
    await db.refresh(row)
    return row


@router.get("", response_model=QuestionnaireResponse)
async def get_latest_questionnaire(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QuestionnaireRow:
    row = await _latest_questionnaire(db, current_user.id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No questionnaire found",
        )
    return row
