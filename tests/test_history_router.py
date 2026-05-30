from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.models.db_models import Image, Prediction, Recommendation


class TestListHistory:
    def test_list_history_empty(self, authenticated_client, mock_db: AsyncMock):
        result = MagicMock()
        result.scalars.return_value.unique.return_value.all.return_value = []
        mock_db.execute.return_value = result

        response = authenticated_client.get("/history")

        assert response.status_code == 200
        assert response.json()["items"] == []


class TestGetAnalysisDetail:
    def test_get_analysis_not_found(self, authenticated_client, mock_db: AsyncMock):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        response = authenticated_client.get("/analysis/999")

        assert response.status_code == 404
        assert response.json()["detail"] == "Prediction not found"

    def test_get_analysis_forbidden_for_other_user(
        self,
        authenticated_client,
        mock_db: AsyncMock,
    ):
        prediction = Prediction(
            id=1,
            user_id=999,
            image_id=1,
            predicted_class="acne0",
            confidence=Decimal("0.9500"),
            created_at=datetime.now(timezone.utc),
            image=Image(id=1, user_id=999, file_url="http://example.com/img.jpg"),
            recommendations=[],
        )
        result = MagicMock()
        result.scalar_one_or_none.return_value = prediction
        mock_db.execute.return_value = result

        response = authenticated_client.get("/analysis/1")

        assert response.status_code == 403
        assert response.json()["detail"] == "Not allowed"
