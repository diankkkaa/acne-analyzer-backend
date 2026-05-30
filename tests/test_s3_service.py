from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from starlette.datastructures import UploadFile

from app.services import s3_service


BUCKET = "acne-analyzer-photos"
REGION = "eu-central-1"


def _upload_file(content: bytes = b"image-bytes", filename: str = "photo.jpg", content_type: str | None = "image/jpeg") -> UploadFile:
    headers = {"content-type": content_type} if content_type else {}
    return UploadFile(filename=filename, file=BytesIO(content), headers=headers)


def _expected_url(key: str) -> str:
    return f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{key}"


@pytest.fixture
def mock_s3_settings():
    with patch.object(s3_service, "settings") as mock_settings:
        mock_settings.AWS_BUCKET_NAME = BUCKET
        mock_settings.AWS_REGION = REGION
        mock_settings.AWS_ACCESS_KEY_ID = "test-key"
        mock_settings.AWS_SECRET_ACCESS_KEY = "test-secret"
        yield mock_settings


@pytest.fixture
def mock_boto_client():
    client = MagicMock()
    with patch("app.services.s3_service.boto3.client", return_value=client) as mock_factory:
        yield client, mock_factory


class TestS3Client:
    def test_s3_client_with_credentials(self, mock_s3_settings, mock_boto_client):
        _, factory = mock_boto_client
        s3_service._s3_client()
        factory.assert_called_once_with(
            "s3",
            region_name=REGION,
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
        )

    def test_s3_client_without_credentials(self, mock_boto_client):
        with patch.object(s3_service, "settings") as mock_settings:
            mock_settings.AWS_REGION = REGION
            mock_settings.AWS_ACCESS_KEY_ID = ""
            mock_settings.AWS_SECRET_ACCESS_KEY = ""
            _, factory = mock_boto_client
            s3_service._s3_client()
            factory.assert_called_once_with("s3", region_name=REGION)


class TestUploadImage:
    def test_upload_image_returns_valid_url(self, mock_s3_settings, mock_boto_client):
        client, _ = mock_boto_client
        fixed_uuid = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        with patch("app.services.s3_service.uuid.uuid4", return_value=fixed_uuid):
            url = s3_service.upload_image(_upload_file())

        assert url == _expected_url("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jpg")
        client.upload_fileobj.assert_called_once()
        args, kwargs = client.upload_fileobj.call_args
        assert args[1] == BUCKET
        assert args[2] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jpg"
        assert kwargs["ExtraArgs"]["ContentType"] == "image/jpeg"

    def test_upload_image_generates_unique_names_for_same_file(
        self,
        mock_s3_settings,
        mock_boto_client,
    ):
        u1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
        u2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
        upload = _upload_file()

        with patch("app.services.s3_service.uuid.uuid4", side_effect=[u1, u2]):
            url1 = s3_service.upload_image(upload)
            url2 = s3_service.upload_image(_upload_file())

        assert url1 != url2
        assert "11111111" in url1
        assert "22222222" in url2

    def test_upload_image_defaults_extension_for_unknown_suffix(
        self,
        mock_s3_settings,
        mock_boto_client,
    ):
        fixed_uuid = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        upload = _upload_file(content=b"x", filename="document.txt", content_type=None)

        with patch("app.services.s3_service.uuid.uuid4", return_value=fixed_uuid):
            url = s3_service.upload_image(upload)

        assert url.endswith(".jpg")
        args, _ = mock_boto_client[0].upload_fileobj.call_args
        assert args[2] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jpg"

    def test_upload_image_without_content_type(self, mock_s3_settings, mock_boto_client):
        client, _ = mock_boto_client
        upload = _upload_file(content=b"x", filename="photo.png", content_type=None)

        with patch("app.services.s3_service.uuid.uuid4", return_value=uuid.uuid4()):
            s3_service.upload_image(upload)

        call_args = client.upload_fileobj.call_args
        assert "ExtraArgs" not in (call_args.kwargs or {})


class TestObjectKeyFromUrl:
    def test_virtual_hosted_style_url(self, mock_s3_settings):
        url = _expected_url("users/1/photo.jpg")
        assert s3_service._object_key_from_url(url) == "users/1/photo.jpg"

    def test_path_style_url(self, mock_s3_settings):
        url = f"https://s3.{REGION}.amazonaws.com/{BUCKET}/folder/image.png"
        assert s3_service._object_key_from_url(url) == "folder/image.png"

    def test_legacy_regex_url(self, mock_s3_settings):
        url = f"https://{BUCKET}.s3-{REGION}.amazonaws.com/legacy/key.webp"
        assert s3_service._object_key_from_url(url) == "legacy/key.webp"

    def test_legacy_regex_url_when_other_parsers_fail(self, mock_s3_settings):
        url = f"https://{BUCKET}.s3.{REGION}.amazonaws.com/regex/key.jpg"
        with (
            patch("app.services.s3_service._key_from_virtual_hosted_url", return_value=None),
            patch("app.services.s3_service._key_from_path_style_url", return_value=None),
        ):
            assert s3_service._object_key_from_url(url) == "regex/key.jpg"

    def test_path_style_url_wrong_bucket_returns_none_via_delete(self, mock_s3_settings, mock_boto_client):
        url = f"https://s3.{REGION}.amazonaws.com/other-bucket/folder/image.png"
        with pytest.raises(ValueError, match="URL does not match"):
            s3_service.delete_image(url)

    def test_virtual_hosted_wrong_bucket_raises(self, mock_s3_settings):
        with pytest.raises(ValueError, match="URL does not match"):
            s3_service._object_key_from_url(
                "https://other-bucket.s3.eu-central-1.amazonaws.com/key.jpg",
            )


class TestDeleteImage:
    def test_delete_image_calls_delete_object_with_correct_key(
        self,
        mock_s3_settings,
        mock_boto_client,
    ):
        client, _ = mock_boto_client
        url = _expected_url("to-delete.jpg")

        s3_service.delete_image(url)

        client.delete_object.assert_called_once_with(Bucket=BUCKET, Key="to-delete.jpg")

    def test_delete_image_invalid_url_raises_value_error(self, mock_s3_settings, mock_boto_client):
        with pytest.raises(ValueError, match="URL does not match"):
            s3_service.delete_image("not-a-valid-s3-url")

        mock_boto_client[0].delete_object.assert_not_called()

    def test_delete_image_client_error_raises_runtime_error(
        self,
        mock_s3_settings,
        mock_boto_client,
    ):
        client, _ = mock_boto_client
        client.delete_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "DeleteObject",
        )

        with pytest.raises(RuntimeError, match="S3 delete failed"):
            s3_service.delete_image(_expected_url("protected.jpg"))
