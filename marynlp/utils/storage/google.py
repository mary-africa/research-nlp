import os
from typing import Union, Optional

# stored in `marynlp.utils.storage.google`

from google.cloud.storage import Client as GCStorageClient
from google.cloud.storage.bucket import Bucket as GCBucket
from google.cloud.storage.blob import Blob as GCBlob

# creating the function that we need to download contents from the 
#  create utilities on to that you can use to create helpful information abou the output of the model
def get_google_bucket_storage_client_from_credential_file(credential_file: Union[str, os.PathLike]) -> GCStorageClient:
    return GCStorageClient.from_service_account_json(str(credential_file))

def get_bucket_from_client(bucket_name: str, storage_client: GCStorageClient, *args, **kwargs) -> GCBucket:
    bucket = storage_client.get_bucket(bucket_name, *args, **kwargs)
    return bucket

def get_blob_from_bucket(blob_name: str, bucket: GCBucket, *args, **kwargs) -> GCBlob:
    return bucket.blob(blob_name, *args, **kwargs)

def save_to_file(save_to_path: str, blob: GCBlob):
    # save the contents of the file
    blob.download_to_filename(save_to_path)
