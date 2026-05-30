from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.models.db_models import QuestionnaireResponse as QuestionnaireRow


def _questionnaire_result(row: QuestionnaireRow | None) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    return result


class TestCreateQuestionnaire:
    def test_create_questionnaire_success(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        mock_user,
    ):
        async def refresh_side_effect(row: QuestionnaireRow) -> None:
            row.id = 10
            row.created_at = datetime.now(timezone.utc)

        mock_db.refresh.side_effect = refresh_side_effect

        response = authenticated_client.post(
            "/questionnaire",
            json={"skin_type": "oily", "sleep_quality": "good"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == mock_user.id
        assert data["skin_type"] == "oily"
        assert data["sleep_quality"] == "good"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()
        mock_db.refresh.assert_awaited_once()


class TestGetQuestionnaire:
    def test_get_existing_questionnaire(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        mock_questionnaire: QuestionnaireRow,
    ):
        mock_db.execute.return_value = _questionnaire_result(mock_questionnaire)

        response = authenticated_client.get("/questionnaire")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_questionnaire.id
        assert data["skin_type"] == mock_questionnaire.skin_type

    def test_get_nonexistent_questionnaire(
        self,
        authenticated_client,
        mock_db: AsyncMock,
    ):
        mock_db.execute.return_value = _questionnaire_result(None)

        response = authenticated_client.get("/questionnaire")

        assert response.status_code == 404
        assert response.json()["detail"] == "No questionnaire found"


class TestUpdateQuestionnaire:
    def test_update_creates_when_none_exists(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        mock_user,
    ):
        mock_db.execute.return_value = _questionnaire_result(None)

        async def refresh_side_effect(row: QuestionnaireRow) -> None:
            row.id = 11
            row.created_at = datetime.now(timezone.utc)

        mock_db.refresh.side_effect = refresh_side_effect

        response = authenticated_client.put(
            "/questionnaire",
            json={"skin_type": "dry", "water_intake": "low"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["skin_type"] == "dry"
        assert data["water_intake"] == "low"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    def test_update_existing_questionnaire(
        self,
        authenticated_client,
        mock_db: AsyncMock,
        mock_questionnaire: QuestionnaireRow,
    ):
        mock_db.execute.return_value = _questionnaire_result(mock_questionnaire)

        response = authenticated_client.put(
            "/questionnaire",
            json={"skin_type": "combination"},
        )

        assert response.status_code == 200
        assert mock_questionnaire.skin_type == "combination"
        mock_db.add.assert_not_called()
        mock_db.commit.assert_awaited_once()
        mock_db.refresh.assert_awaited_once()
