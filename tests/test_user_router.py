from app.models.db_models import User


class TestUserRouter:
    def test_read_me(self, authenticated_client, mock_user: User):
        response = authenticated_client.get("/user/me")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_user.id
        assert data["email"] == mock_user.email
