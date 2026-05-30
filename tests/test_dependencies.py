from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from app.models.db_models import User


class TestGetCurrentUserDependency:
    def test_get_current_user_via_auth_headers(
        self,
        client,
        mock_db: AsyncMock,
        mock_user: User,
        auth_headers: dict[str, str],
    ):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        response = client.get("/user/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == mock_user.email
        assert data["id"] == mock_user.id

    def test_get_current_user_unauthorized_without_token(self, client):
        response = client.get("/user/me")

        assert response.status_code == 401
