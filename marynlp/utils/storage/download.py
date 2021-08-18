from pathlib import Path
from . import localize_google_cloud_file, local, unzip_file, google, download

from ..log import logger as log


def file_from_google_temporary(cloud_file_blob_name: str, bucket: google.GCBucket):
    store_local_path = local.get_temp_path(cloud_file_blob_name)
    return file_from_google(cloud_file_blob_name, bucket, store_local_path)


def file_from_google_to_store(cloud_file_blob_name: str, bucket: google.GCBucket):
    store_local_path = local.get_path_from_store(cloud_file_blob_name)
    return file_from_google(cloud_file_blob_name, bucket, store_local_path)


def file_from_google(cloud_file_blob_name: str, bucket: google.GCBucket, save_to_path: str):
    # create parent folders
    Path(save_to_path).parent.mkdir(exist_ok=True)
    return localize_google_cloud_file(cloud_file_blob_name, bucket=bucket, save_to_path=save_to_path)


def prepare_zipped_model_from_google(cloud_file_blob_name: str, bucket: google.GCBucket, folder_name_for_contents: str = None):
    store_folder_name = folder_name_for_contents
    if folder_name_for_contents is None:
        # get the new name
        store_folder_name = ".".join(cloud_file_blob_name.split(".")[:-1])  
    
    # unzip to differen location
    store_folder_path = local.get_path_from_store(store_folder_name)

    # Check if the model exists in that path
    if Path(store_folder_path).exists():
        # exists
        log.info('File exists:', store_folder_path)
        return store_folder_path

    # model contents didn't exists before
    # download model to temporary location
    log.info('Downloading:', cloud_file_blob_name)
    temp_zipped_path = download.file_from_google_temporary(cloud_file_blob_name, bucket)
    log.info("Download completed to path:", temp_zipped_path)
    
    unzip_file(temp_zipped_path, store_folder_path)    

    log.inf("Unzipped to path:", store_folder_path)
    return str(store_folder_path)
