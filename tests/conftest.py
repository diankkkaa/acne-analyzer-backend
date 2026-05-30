from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.main import app
from app.models.db_models import QuestionnaireResponse as QuestionnaireRow
from app.models.db_models import User
from app.services import auth_service


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock(spec=AsyncSession)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_user() -> User:
    return User(
        id=1,
        email="test@test.com",
        password_hash=auth_service.hash_password("password123"),
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def auth_headers(mock_user: User) -> dict[str, str]:
    token = auth_service.create_access_token(data={"sub": mock_user.email})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_questionnaire(mock_user: User) -> QuestionnaireRow:
    return QuestionnaireRow(
        id=1,
        user_id=mock_user.id,
        created_at=datetime.now(timezone.utc),
        skin_type="oily",
        sleep_quality="good",
    )


@pytest.fixture
def client(mock_db: AsyncMock) -> TestClient:
    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def authenticated_client(
    client: TestClient,
    mock_user: User,
) -> TestClient:
    async def override_get_current_user():
        return mock_user

    app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    app.dependency_overrides.pop(get_current_user, None)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
