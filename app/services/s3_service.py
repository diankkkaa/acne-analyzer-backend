from __future__ import annotations

import re
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile

from app.config import settings


def _s3_client():
    kwargs = {"region_name": settings.AWS_REGION}
    if settings.AWS_ACCESS_KEY_ID:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
    if settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    return boto3.client("s3", **kwargs)


def _public_object_url(bucket: str, region: str, key: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def upload_image(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        ext = ".jpg"
    key = f"{uuid.uuid4()}{ext}"

    extra_args: dict = {}
    if file.content_type:
        extra_args["ContentType"] = file.content_type

    client = _s3_client()
    file.file.seek(0)
    if extra_args:
        client.upload_fileobj(
            file.file,
            settings.AWS_BUCKET_NAME,
            key,
            ExtraArgs=extra_args,
        )
    else:
        client.upload_fileobj(file.file, settings.AWS_BUCKET_NAME, key)
    return _public_object_url(settings.AWS_BUCKET_NAME, settings.AWS_REGION, key)


def _key_from_virtual_hosted_url(parsed: urlparse, bucket: str) -> str | None:
    host = (parsed.netloc or "").lower()
    if not host.startswith(f"{bucket.lower()}.s3"):
        return None
    path = unquote(parsed.path.lstrip("/"))
    return path or None


def _key_from_path_style_url(parsed: urlparse, bucket: str, region: str) -> str | None:
    host = (parsed.netloc or "").lower()
    if host != f"s3.{region}.amazonaws.com" and not host.startswith("s3."):
        return None
    parts = [p for p in unquote(parsed.path).split("/") if p]
    if len(parts) >= 2 and parts[0] == bucket:
        return "/".join(parts[1:])
    return None


def _object_key_from_url(file_url: str) -> str:
    parsed = urlparse(file_url.strip())
    bucket = settings.AWS_BUCKET_NAME
    region = settings.AWS_REGION

    key = _key_from_virtual_hosted_url(parsed, bucket)
    if key is not None:
        return key

    key = _key_from_path_style_url(parsed, bucket, region)
    if key is not None:
        return key

    m = re.match(
        rf"^https?://{re.escape(bucket)}\.s3[.-]{re.escape(region)}\.amazonaws\.com/(.+)$",
        file_url.strip(),
        re.IGNORECASE,
    )
    if m:
        return unquote(m.group(1))

    raise ValueError(f"URL does not match configured bucket/region or is not a valid S3 URL: {file_url!r}")


def delete_image(file_url: str) -> None:
    key = _object_key_from_url(file_url)
    client = _s3_client()
    try:
        client.delete_object(Bucket=settings.AWS_BUCKET_NAME, Key=key)
    except ClientError as e:
        raise RuntimeError(f"S3 delete failed for key {key!r}: {e}") from e
