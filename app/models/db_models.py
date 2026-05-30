from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric
from sqlalchemy.dialects.mssql import NVARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.schema import Identity
from sqlalchemy.sql import text


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "Users"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    email: Mapped[str] = mapped_column(NVARCHAR(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(NVARCHAR(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("GETDATE()"),
        nullable=False,
    )

    images: Mapped[list[Image]] = relationship(back_populates="user")
    questionnaire_responses: Mapped[list[QuestionnaireResponse]] = relationship(
        back_populates="user",
    )
    predictions: Mapped[list[Prediction]] = relationship(back_populates="user")
    recommendations: Mapped[list[Recommendation]] = relationship(back_populates="user")


class Image(Base):
    __tablename__ = "Images"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("Users.id"), nullable=False)
    file_url: Mapped[str] = mapped_column(NVARCHAR(500), nullable=False)
    upload_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("GETDATE()"),
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="images")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="image")


class QuestionnaireResponse(Base):
    __tablename__ = "Questionnaire_Responses"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("Users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("GETDATE()"),
        nullable=False,
    )
    has_family_history: Mapped[str | None] = mapped_column(NVARCHAR(20))
    has_hormonal_acne: Mapped[str | None] = mapped_column(NVARCHAR(20))
    uses_comedogenic_products: Mapped[str | None] = mapped_column(NVARCHAR(20))
    hair_products_trigger: Mapped[str | None] = mapped_column(NVARCHAR(20))
    stress_trigger: Mapped[str | None] = mapped_column(NVARCHAR(20))
    sleep_quality: Mapped[str | None] = mapped_column(NVARCHAR(20))
    is_smoking: Mapped[str | None] = mapped_column(NVARCHAR(20))
    water_intake: Mapped[str | None] = mapped_column(NVARCHAR(20))
    dairy_consumption: Mapped[str | None] = mapped_column(NVARCHAR(20))
    high_glycemic_diet: Mapped[str | None] = mapped_column(NVARCHAR(20))
    food_triggers: Mapped[str | None] = mapped_column(NVARCHAR(20))
    skin_type: Mapped[str | None] = mapped_column(NVARCHAR(20))
    oily_skin: Mapped[str | None] = mapped_column(NVARCHAR(20))
    jawline_acne: Mapped[str | None] = mapped_column(NVARCHAR(20))
    has_demodex: Mapped[str | None] = mapped_column(NVARCHAR(20))
    vitamin_d_level: Mapped[str | None] = mapped_column(NVARCHAR(20))
    dairy_sensitivity: Mapped[str | None] = mapped_column(NVARCHAR(20))

    user: Mapped[User] = relationship(back_populates="questionnaire_responses")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="questionnaire",
    )


class Prediction(Base):
    __tablename__ = "Predictions"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("Users.id"), nullable=False)
    image_id: Mapped[int] = mapped_column(ForeignKey("Images.id"), nullable=False)
    questionnaire_id: Mapped[int | None] = mapped_column(
        ForeignKey("Questionnaire_Responses.id"),
        nullable=True,
    )
    predicted_class: Mapped[str] = mapped_column(NVARCHAR(20), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("GETDATE()"),
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="predictions")
    image: Mapped[Image] = relationship(back_populates="predictions")
    questionnaire: Mapped[QuestionnaireResponse | None] = relationship(
        back_populates="predictions",
    )
    recommendations: Mapped[list[Recommendation]] = relationship(
        back_populates="prediction",
    )


class Recommendation(Base):
    __tablename__ = "Recommendations"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("Predictions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("Users.id"), nullable=False)
    prompt_used: Mapped[str | None] = mapped_column(NVARCHAR(None))
    recommendation_text: Mapped[str] = mapped_column(NVARCHAR(None), nullable=False)
    model_used: Mapped[str | None] = mapped_column(NVARCHAR(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("GETDATE()"),
        nullable=False,
    )

    prediction: Mapped[Prediction] = relationship(back_populates="recommendations")
    user: Mapped[User] = relationship(back_populates="recommendations")
