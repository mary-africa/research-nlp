from pathlib import Path

from . import google
from . import local

from marynlp.utils import log
from typing import Optional

def get_bucket(google_credential_path: str, bucket_name: str) -> google.GCBucket:
    gc = google.get_google_bucket_storage_client_from_credential_file(google_credential_path)
    return google.get_bucket_from_client(bucket_name, gc)

def localize_google_cloud_file(blob_name, bucket: google.GCBucket, save_to_path: Optional[str] = None):
    """Localizes the files on the cloud"""
    # convert blob name to store path
    local_path = save_to_path

    if save_to_path is None:
        local_path = local.get_path_from_store(blob_name)

    log.info("Downloading '%s'..." % blob_name)

    blob = google.get_blob_from_bucket(blob_name, bucket)
    google.save_to_file(local_path, blob)

    log.info('Download complete: ' + local_path)
    return local_path


def unzip_file(file_to_zip: str, save_to_path: Optional[str] = None):
    """Unpacks the contents of `file_to_zip` to the location `save_to_path`"""
    from zipfile import ZipFile

    with ZipFile(file_to_zip, mode='r') as zpf:
        zpf.extractall(path=save_to_path)

