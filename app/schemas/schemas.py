from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

PredictedClass = Literal["acne0", "acne1", "acne2", "acne3", "clear"]

PREDICTED_CLASS_DESCRIPTION = (
    "Acne severity class: acne0 (grade 1 mild), acne1 (grade 2 moderate), "
    "acne2 (grade 3 moderate-severe), acne3 (grade 4 severe), "
    "clear (no acne)."
)

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    sub: str | None = None


class QuestionnaireBase(BaseModel):
    has_family_history: str | None = None
    has_hormonal_acne: str | None = None
    uses_comedogenic_products: str | None = None
    hair_products_trigger: str | None = None
    stress_trigger: str | None = None
    sleep_quality: str | None = None
    is_smoking: str | None = None
    water_intake: str | None = None
    dairy_consumption: str | None = None
    high_glycemic_diet: str | None = None
    food_triggers: str | None = None
    skin_type: str | None = None
    oily_skin: str | None = None
    jawline_acne: str | None = None
    has_demodex: str | None = None
    vitamin_d_level: str | None = None
    dairy_sensitivity: str | None = None


class QuestionnaireCreate(QuestionnaireBase):
    pass


class QuestionnaireUpdate(QuestionnaireBase):
    pass


class QuestionnaireResponse(QuestionnaireBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    image_id: int
    questionnaire_id: int | None
    predicted_class: PredictedClass = Field(description=PREDICTED_CLASS_DESCRIPTION)
    confidence: Decimal
    created_at: datetime


class AnalyzeRequest(BaseModel):
    image_id: int
    questionnaire_id: int | None = None


class AnalyzeResponse(BaseModel):
    predicted_class: PredictedClass = Field(description=PREDICTED_CLASS_DESCRIPTION)
    confidence: Decimal
    recommendation_text: str
    prediction_id: int | None = None


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prediction_id: int
    image_id: int
    file_url: str
    predicted_class: PredictedClass = Field(description=PREDICTED_CLASS_DESCRIPTION)
    confidence: Decimal
    created_at: datetime
    recommendation_preview: str | None = None


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class ImageUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_url: str
    upload_time: datetime


class ImageCreatedResponse(BaseModel):
    image_id: int
    file_path: str


class AnalyzePipelineResponse(BaseModel):
    predicted_class: PredictedClass = Field(description=PREDICTED_CLASS_DESCRIPTION)
    confidence: Decimal
    recommendation_text: str
    prediction_id: int


class AnalysisDetailResponse(BaseModel):
    prediction_id: int
    image_id: int
    image_url: str
    predicted_class: PredictedClass = Field(description=PREDICTED_CLASS_DESCRIPTION)
    confidence: Decimal
    created_at: datetime
    questionnaire: QuestionnaireResponse | None = None
    recommendation_text: str | None = None
