from minio import Minio

from utils.settings import settings

BUCKET_NAME = settings.bucket_name
minio_client = Minio(
    endpoint=settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=False
)
