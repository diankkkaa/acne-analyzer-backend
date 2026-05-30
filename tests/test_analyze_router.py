from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError

from app.database import get_db
from app.dependencies import get_current_user
from app.main import app
from app.models.db_models import Image as ImageRow
from app.models.db_models import Prediction
from app.models.db_models import QuestionnaireResponse as QuestionnaireRow
from app.models.db_models import User
from app.routers import analyze as analyze_router


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


def _execute_result(scalar_value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_value
    return result


def _configure_flush_refresh(mock_db: AsyncMock) -> None:
    async def refresh_side_effect(obj) -> None:
        if isinstance(obj, ImageRow) and getattr(obj, "id", None) is None:
            obj.id = 5
        elif isinstance(obj, Prediction):
            if getattr(obj, "id", None) is None:
                obj.id = 10
            obj.predicted_class = obj.predicted_class or "acne1"
            obj.confidence = obj.confidence or Decimal("0.8500")

    mock_db.flush = AsyncMock()
    mock_db.refresh = AsyncMock(side_effect=refresh_side_effect)


@pytest.fixture
def image_file():
    return ("face.jpg", _jpeg_bytes(), "image/jpeg")


@pytest.fixture
def mock_s3_url():
    return "https://acne-analyzer-photos.s3.eu-central-1.amazonaws.com/test-key.jpg"


class TestUploadImage:
    def test_upload_image_rejects_when_no_face(
        self,
        authenticated_client,
        image_file,
    ):
        with patch("app.routers.analyze.face_service.is_human_face", return_value=False):
            response = authenticated_client.post(
                "/upload-image",
                files={"file": image_file},
            )

        assert response.status_code == 422
        assert "не виявлено обличчя" in response.json()["detail"]

    def test_upload_image_success(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        image_file,
        mock_s3_url,
    ):
        _configure_flush_refresh(mock_db)

        with (
            patch("app.routers.analyze.face_service.is_human_face", return_value=True),
            patch("app.routers.analyze.s3_service.upload_image", return_value=mock_s3_url),
        ):
            response = authenticated_client.post(
                "/upload-image",
                files={"file": image_file},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["image_id"] == 5
        assert data["file_path"] == mock_s3_url
        mock_db.add.assert_called()
        mock_db.commit.assert_awaited()

    def test_upload_image_rejects_invalid_file(
        self,
        mock_db: AsyncMock,
        mock_user: User,
    ):
        async def override_get_db():
            yield mock_db

        async def override_get_current_user():
            return mock_user

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_current_user] = override_get_current_user

        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                with patch(
                    "app.routers.analyze.Image.open",
                    side_effect=UnidentifiedImageError("cannot identify image file"),
                ):
                    response = client.post(
                        "/upload-image",
                        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
                    )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 500
        assert "cannot identify image file" in response.json()["detail"]

    def test_questionnaire_to_dict_none(self):
        assert analyze_router._questionnaire_to_dict(None) is None


class TestAnalyzePipeline:
    def test_analyze_full_success(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        image_file,
        mock_s3_url,
    ):
        _configure_flush_refresh(mock_db)

        with (
            patch("app.routers.analyze.face_service.is_human_face", return_value=True),
            patch("app.routers.analyze.s3_service.upload_image", return_value=mock_s3_url),
            patch(
                "app.routers.analyze.ml_service.predict",
                return_value={"predicted_class": "acne1", "confidence": 0.87},
            ),
            patch(
                "app.routers.analyze.rag_service.generate_recommendation",
                return_value=("Рекомендації для догляду.", "prompt text"),
            ),
        ):
            response = authenticated_client.post(
                "/analyze",
                files={"file": image_file},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "acne1"
        assert data["confidence"] == "0.8700"
        assert data["recommendation_text"] == "Рекомендації для догляду."
        assert data["prediction_id"] == 10
        assert mock_db.add.call_count >= 3
        mock_db.commit.assert_awaited()

    def test_analyze_with_questionnaire_success(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        mock_user,
        image_file,
        mock_s3_url,
    ):
        _configure_flush_refresh(mock_db)
        questionnaire = QuestionnaireRow(
            id=3,
            user_id=mock_user.id,
            created_at=datetime.now(timezone.utc),
            skin_type="oily",
        )
        mock_db.execute.return_value = _execute_result(questionnaire)

        with (
            patch("app.routers.analyze.face_service.is_human_face", return_value=True),
            patch("app.routers.analyze.s3_service.upload_image", return_value=mock_s3_url),
            patch(
                "app.routers.analyze.ml_service.predict",
                return_value={"predicted_class": "acne0", "confidence": 0.75},
            ),
            patch(
                "app.routers.analyze.rag_service.generate_recommendation",
                return_value=("Поради.", "prompt"),
            ) as mock_rag,
        ):
            response = authenticated_client.post(
                "/analyze",
                files={"file": image_file},
                data={"questionnaire_id": "3"},
            )

        assert response.status_code == 200
        mock_rag.assert_called_once()
        assert mock_rag.call_args.args[2]["skin_type"] == "oily"

    def test_analyze_questionnaire_not_found(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        image_file,
    ):
        mock_db.execute.return_value = _execute_result(None)

        response = authenticated_client.post(
            "/analyze",
            files={"file": image_file},
            data={"questionnaire_id": "999"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Questionnaire not found"

    def test_analyze_questionnaire_forbidden_for_other_user(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        image_file,
    ):
        questionnaire = QuestionnaireRow(
            id=4,
            user_id=999,
            created_at=datetime.now(timezone.utc),
            skin_type="dry",
        )
        mock_db.execute.return_value = _execute_result(questionnaire)

        response = authenticated_client.post(
            "/analyze",
            files={"file": image_file},
            data={"questionnaire_id": "4"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "Questionnaire belongs to another user"

    def test_analyze_rejects_when_no_face(
        self,
        authenticated_client,
        image_file,
    ):
        with patch("app.routers.analyze.face_service.is_human_face", return_value=False):
            response = authenticated_client.post(
                "/analyze",
                files={"file": image_file},
            )

        assert response.status_code == 422
        assert "не виявлено обличчя" in response.json()["detail"]

    def test_analyze_ml_not_ready(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        image_file,
        mock_s3_url,
    ):
        _configure_flush_refresh(mock_db)

        with (
            patch("app.routers.analyze.face_service.is_human_face", return_value=True),
            patch("app.routers.analyze.s3_service.upload_image", return_value=mock_s3_url),
            patch(
                "app.routers.analyze.ml_service.predict",
                side_effect=RuntimeError("Model not loaded"),
            ),
        ):
            response = authenticated_client.post(
                "/analyze",
                files={"file": image_file},
            )

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_analyze_rag_not_ready(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        image_file,
        mock_s3_url,
    ):
        _configure_flush_refresh(mock_db)

        with (
            patch("app.routers.analyze.face_service.is_human_face", return_value=True),
            patch("app.routers.analyze.s3_service.upload_image", return_value=mock_s3_url),
            patch(
                "app.routers.analyze.ml_service.predict",
                return_value={"predicted_class": "acne2", "confidence": 0.66},
            ),
            patch(
                "app.routers.analyze.rag_service.generate_recommendation",
                side_effect=RuntimeError("Knowledge base not loaded"),
            ),
        ):
            response = authenticated_client.post(
                "/analyze",
                files={"file": image_file},
            )

        assert response.status_code == 503
        assert "Knowledge base not loaded" in response.json()["detail"]


class TestDeleteAnalysis:
    def _prediction(self, user_id: int = 1, prediction_id: int = 10) -> Prediction:
        image = ImageRow(
            id=5,
            user_id=user_id,
            file_url="https://acne-analyzer-photos.s3.eu-central-1.amazonaws.com/key.jpg",
        )
        return Prediction(
            id=prediction_id,
            user_id=user_id,
            image_id=image.id,
            predicted_class="acne1",
            confidence=Decimal("0.8000"),
            image=image,
        )

    def test_delete_analysis_not_found(self, authenticated_client, mock_db: AsyncMock):
        mock_db.execute.return_value = _execute_result(None)

        response = authenticated_client.delete("/analysis/999")

        assert response.status_code == 404
        assert response.json()["detail"] == "Prediction not found"

    def test_delete_analysis_forbidden_for_other_user(
        self,
        authenticated_client,
        mock_db: AsyncMock,
    ):
        mock_db.execute.return_value = _execute_result(self._prediction(user_id=999))

        response = authenticated_client.delete("/analysis/10")

        assert response.status_code == 403
        assert response.json()["detail"] == "Not allowed"

    def test_delete_analysis_success_without_s3(
        self,
        authenticated_client,
        mock_db: AsyncMock,
    ):
        mock_db.execute.return_value = _execute_result(self._prediction())

        with patch("app.routers.analyze.s3_service.delete_image") as mock_delete:
            response = authenticated_client.delete("/analysis/10")

        assert response.status_code == 204
        mock_delete.assert_not_called()
        assert mock_db.commit.await_count >= 1

    def test_delete_analysis_success_with_s3(
        self,
        authenticated_client,
        mock_db: AsyncMock,
    ):
        prediction = self._prediction()
        mock_db.execute.return_value = _execute_result(prediction)
        file_url = prediction.image.file_url

        with patch("app.routers.analyze.s3_service.delete_image") as mock_delete:
            response = authenticated_client.delete("/analysis/10?delete_s3=true")

        assert response.status_code == 204
        mock_delete.assert_called_once_with(file_url)
        assert mock_db.commit.await_count >= 2

    def test_delete_analysis_invalid_s3_url_does_not_fail(
        self,
        authenticated_client,
        mock_db: AsyncMock,
    ):
        prediction = self._prediction()
        prediction.image.file_url = "https://invalid.example.com/not-s3.jpg"
        mock_db.execute.return_value = _execute_result(prediction)

        with patch(
            "app.routers.analyze.s3_service.delete_image",
            side_effect=ValueError("invalid url"),
        ):
            response = authenticated_client.delete("/analysis/10?delete_s3=true")

        assert response.status_code == 204

    def test_delete_analysis_s3_runtime_error_does_not_fail(
        self,
        authenticated_client,
        mock_db: AsyncMock,
    ):
        prediction = self._prediction()
        mock_db.execute.return_value = _execute_result(prediction)

        with patch(
            "app.routers.analyze.s3_service.delete_image",
            side_effect=RuntimeError("S3 delete failed"),
        ):
            response = authenticated_client.delete("/analysis/10?delete_s3=true")

        assert response.status_code == 204
