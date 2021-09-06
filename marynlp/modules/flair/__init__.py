from ... import funcutils as f
from ..source import get_model_from_google_bucket, FLAIR_TEXT_CLASSIFIERS, FLAIR_SEQUENCE_TAGGERS

# Including this hear alone makes it such that you must have 
# flair to be able to use this model
from flair.models import SequenceTagger
from .text_classifiers import TextClassifier

# Model builders
build_sequence_tagger: SequenceTagger = f.apply(SequenceTagger.load)(f.partial(get_model_from_google_bucket, flair_model_path_dict=FLAIR_SEQUENCE_TAGGERS))
build_text_classifier: TextClassifier = f.apply(TextClassifier.load)(f.partial(get_model_from_google_bucket, flair_model_path_dict=FLAIR_TEXT_CLASSIFIERS))
