"""
ObjectStore — manages file storage in MinIO (S3-compatible).

Stores source PDFs, images, and any extracted media assets. File references
are stored in the SourceDocument model as object_key values.
"""

from __future__ import annotations

import logging
from io import BytesIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ObjectStore:
    """Thin wrapper around boto3 S3/MinIO for object storage operations."""

    def __init__(self):
        self._s3 = boto3.client(
            "s3",
            endpoint_url=settings.storage_endpoint,
            aws_access_key_id=settings.storage_access_key,
            aws_secret_access_key=settings.storage_secret_key,
            config=Config(signature_version="s3v4"),
            use_ssl=settings.storage_use_ssl,
        )
        self._bucket = settings.storage_bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                self._s3.create_bucket(Bucket=self._bucket)
                logger.info("object_store.bucket_created", extra={"bucket": self._bucket})
            else:
                raise

    def upload(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload bytes to MinIO and return the object key."""
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("object_store.upload", extra={"key": key, "size": len(data)})
        return key

    def download(self, key: str) -> bytes:
        """Download an object by key."""
        response = self._s3.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=key)

    def presign_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a pre-signed URL for temporary public access."""
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )
