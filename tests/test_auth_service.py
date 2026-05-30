from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from jose import jwt

from app.config import settings
from app.models.db_models import User
from app.services import auth_service
from tests.conftest import decode_token


class TestPasswordHashing:
    def test_hash_password_returns_string_not_equal_to_original(self):
        password = "password123"
        hashed = auth_service.hash_password(password)

        assert isinstance(hashed, str)
        assert hashed != password

    def test_verify_password_correct(self, mock_user: User):
        assert auth_service.verify_password("password123", mock_user.password_hash) is True

    def test_verify_password_incorrect(self, mock_user: User):
        assert auth_service.verify_password("wrong-password", mock_user.password_hash) is False

    def test_hash_password_truncates_long_password(self):
        long_password = "a" * 100
        hashed = auth_service.hash_password(long_password)
        assert auth_service.verify_password(long_password, hashed) is True


class TestCreateAccessToken:
    def test_create_access_token_contains_sub(self):
        token = auth_service.create_access_token(data={"sub": "test@test.com"})
        payload = decode_token(token)

        assert payload["sub"] == "test@test.com"
        assert "exp" in payload


class TestGetUserByEmail:
    async def test_get_user_by_email_found(self, mock_db: AsyncMock, mock_user: User):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        user = await auth_service.get_user_by_email(mock_db, mock_user.email)

        assert user is mock_user
        mock_db.execute.assert_awaited_once()

    async def test_get_user_by_email_not_found(self, mock_db: AsyncMock):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        user = await auth_service.get_user_by_email(mock_db, "missing@test.com")

        assert user is None


class TestGetUserById:
    async def test_get_user_by_id_found(self, mock_db: AsyncMock, mock_user: User):
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        user = await auth_service.get_user_by_id(mock_db, mock_user.id)

        assert user is mock_user

    async def test_get_user_by_id_not_found(self, mock_db: AsyncMock):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        user = await auth_service.get_user_by_id(mock_db, 999)

        assert user is None


class TestGetCurrentUser:
    async def test_get_current_user_success(
        self,
        mock_db: AsyncMock,
        mock_user: User,
    ):
        token = auth_service.create_access_token(data={"sub": mock_user.email})
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = result

        user = await auth_service.get_current_user(token, mock_db)

        assert user is mock_user

    async def test_get_current_user_invalid_token(self, mock_db: AsyncMock):
        with pytest.raises(HTTPException) as exc_info:
            await auth_service.get_current_user("not-a-valid-jwt", mock_db)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Could not validate credentials"

    async def test_get_current_user_missing_sub(self, mock_db: AsyncMock):
        token = jwt.encode({"exp": 9999999999}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.get_current_user(token, mock_db)

        assert exc_info.value.status_code == 401

    async def test_get_current_user_sub_not_string(self, mock_db: AsyncMock):
        token = jwt.encode(
            {"sub": 123, "exp": 9999999999},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.get_current_user(token, mock_db)

        assert exc_info.value.status_code == 401

    async def test_get_current_user_user_not_found(self, mock_db: AsyncMock, mock_user: User):
        token = auth_service.create_access_token(data={"sub": mock_user.email})
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.get_current_user(token, mock_db)

        assert exc_info.value.status_code == 401
