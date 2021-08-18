
from ...text import flair
from flair.models import TextClassifier as FlairTextClassifier
from flair.data import Sentence

from typing import Iterable, List, Union
from functools import reduce


class TextClassifier(object):
    def __init__(self, text_clf: FlairTextClassifier):
        self.classifier = text_clf
        self.transform = flair.normalize_and_mask_swahili_text

    @property
    def native_object(self):
        return self.text_clf

    def native_predict(self, *args, **kwargs):
        """Making prediction as it you are working with flair directly"""
        return self.text_clf.predict(*args, **kwargs)

    def classify(self, text: Union[str, Iterable[str]], language_code='sw'):
        if isinstance(text, str): 
            sentence = Sentence(self.transform(text), language_code=language_code)
            self.native_predict(sentence)
            out = sentence.to_dict()
            return dict(label=out["label"])

        # Assumed iterable
        outputs = [Sentence(self.transform(t), language_code=language_code) for t in text] 
        
        # Make predictions for all the outputs
        for sentence in outputs:
            self.classifier.predict(sentence)

        # shape the outputs as dictionaries
        outputs = [out.to_dict() for out in outputs] 

        # return outputs
        return reduce((lambda x, y: x + y), [out['labels'] for out in outputs])

    def classify_proba(self, text: Union[str, List[str]], language_code='sw'):
        if isinstance(text, str):
            sentence = Sentence(self.transform(text), language_code=language_code)
            self.classifier.predict(sentence, multi_class_prob=True)
            return sentence.to_dict()

        outputs = [self.classifier.predict(text).to_dict() for t in text]
        return outputs

    @classmethod
    def load(cls, model_path: str, model_name = 'sw-flair-model'):
        model = FlairTextClassifier(model_path)
        model.name = model_name
        return cls(model)
