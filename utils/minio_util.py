import os

from dotenv import load_dotenv
from minio import Minio

load_dotenv()
ENDPOINT = os.getenv('ENDPOINT')
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
minio_client = Minio(
    endpoint=ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False
)
