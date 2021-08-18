from pathlib import Path
from ..utils.storage import download

# Flair relatedel models
FLAIR_TEXT_CLASSIFIERS = {
    'sw-exp-sent_analy-small': (
        'flair/classifier/sw-exp-sent_analy-small.zip', # Blob name as it's save on the GCP bucket
        True # is the blob unzippable or not
    ),
    'early-wpc': ('flair/classifier/exp-wpc-small.zip', True),
    'early-sentiment-hasf': ('flair/classifier/sw-ft100-ffw-bilstm-exp-sent_analy-small-h256-noreproj.zip', True)
}

FLAIR_SEQUENCE_TAGGERS = {
    'early-alpha-tag-ner': ('flair/taggers/sw-ner-gen1f-base.zip', True),
    'early-alpha-tag-pos': ('flair/taggers/sw-pos-early-h256.zip', True)
}

FLAIR_EMBEDDINGS = {
    # TODO: add a reader to include the references after opening the folder
    'sw-early-uncased-forward': ('flair/language-model/sw-early-uncased-forward.zip', True),
}


def get_model_from_google_bucket(src: str, flair_model_path_dict, bucket):
    """Gets the model"""
    if src not in flair_model_path_dict:
        assert Path(src).exists(), "The model path '%s' doesn't exist" % src
        return src
    
    # download the model and provice it
    blob_name, zipped = flair_model_path_dict[src]
    
    if zipped:
        # downloads the model and unzip
        return download.prepare_zipped_model_from_google(blob_name, bucket)
    
    # if not zipped
    return download.file_from_google_to_store(blob_name, bucket)
