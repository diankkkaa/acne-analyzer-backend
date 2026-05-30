from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.exc import IntegrityError

from app.models.db_models import User


class TestRegister:
    def test_register_success(self, client, mock_db: AsyncMock):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        async def refresh_side_effect(user: User) -> None:
            user.id = 2
            user.created_at = datetime.now(timezone.utc)

        mock_db.refresh.side_effect = refresh_side_effect

        response = client.post(
            "/auth/register",
            json={"email": "newuser@test.com", "password": "password123"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@test.com"
        assert data["id"] == 2
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()
        mock_db.refresh.assert_awaited_once()

    def test_register_duplicate_email_via_lookup(self, client, mock_db: AsyncMock, mock_user: User):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        response = client.post(
            "/auth/register",
            json={"email": "test@test.com", "password": "password123"},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "Email already registered"
        mock_db.add.assert_not_called()

    def test_register_duplicate_email_integrity_error(self, client, mock_db: AsyncMock):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result
        mock_db.commit.side_effect = IntegrityError("INSERT", {}, Exception("duplicate"))

        response = client.post(
            "/auth/register",
            json={"email": "race@test.com", "password": "password123"},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "Email already registered"
        mock_db.rollback.assert_awaited_once()

    def test_register_normalizes_email(self, client, mock_db: AsyncMock):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        async def refresh_side_effect(user: User) -> None:
            user.id = 3
            user.created_at = datetime.now(timezone.utc)

        mock_db.refresh.side_effect = refresh_side_effect

        response = client.post(
            "/auth/register",
            json={"email": "  MixedCase@Test.COM  ", "password": "password123"},
        )

        assert response.status_code == 201
        assert response.json()["email"] == "mixedcase@test.com"


class TestLogin:
    def test_login_success(self, client, mock_db: AsyncMock, mock_user: User):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        response = client.post(
            "/auth/login",
            data={"username": "test@test.com", "password": "password123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        from tests.conftest import decode_token

        decoded = decode_token(data["access_token"])
        assert decoded["sub"] == mock_user.email

    def test_login_wrong_password(self, client, mock_db: AsyncMock, mock_user: User):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        response = client.post(
            "/auth/login",
            data={"username": "test@test.com", "password": "wrong-password"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect email or password"

    def test_login_nonexistent_user(self, client, mock_db: AsyncMock):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        response = client.post(
            "/auth/login",
            data={"username": "ghost@test.com", "password": "password123"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect email or password"

    def test_login_normalizes_username(self, client, mock_db: AsyncMock, mock_user: User):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        response = client.post(
            "/auth/login",
            data={"username": "  TEST@TEST.COM  ", "password": "password123"},
        )

        assert response.status_code == 200
