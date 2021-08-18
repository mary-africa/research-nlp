import torch
from torchaudio.transforms import MelSpectrogram
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

import pandas as pd

from overrides import overrides

from .text_encoders import CharacterEncoder, GreedyEncoder
from .nn import SpeechRecognitionModel

default_model_params = dict(
    n_cnn_layers=3,
    n_rnn_layers=5,
    rnn_dim=512,
    n_feats=128,
)

SPEECH_RECOGNITION_MODELS = {
    "mnm23-early": "voice/mnm-early-6k-ep100-bc64.zip"
}


class InferenceSpeechRecognition(object):
    """The Speech Recognizer

    Args:
        speech_recogn_model: A Trained Speech Recognition model
        char_list: List of all the characters to consider in the model
    """
    def __init__(self,
                 speech_recogn_model: SpeechRecognitionModel,
                 character_encoder: CharacterEncoder,
                 test_transformer: nn.Module):

        self.speech_model = speech_recogn_model
        self.greedy_encoder = GreedyEncoder(character_encoder)
        self.test_transform = test_transformer

    def recognize(self, spect_tensor: torch.Tensor):
        """
        Takes in an input data
        """
        output = self.test_transform(spect_tensor).unsqueeze(1)
        output = self.speech_model(output)
        output = F.log_softmax(output, dim=2)

        return self.greedy_encoder.decode_test(output)

class MNM23SpeechRecognizer(object):
    """The Speech Recognizer

    Args:
        speech_recogn_model: A Trained Speech Recognition model
        char_list: List of all the characters to consider in the model
    """
    def __init__(self,
                 speech_recogn_model: SpeechRecognitionModel,
                 character_encoder: CharacterEncoder,
                 test_transformer: nn.Module):

        self.speech_model = speech_recogn_model
        self.greedy_encoder = GreedyEncoder(character_encoder)
        self.test_transform = test_transformer

    def recognize(self, spect_tensor: torch.Tensor):
        """
        Takes in an input data
        """
        output = self.test_transform(spect_tensor).unsqueeze(1)
        output = self.speech_model(output)
        output = F.log_softmax(output, dim=2)

        return self.greedy_encoder.decode_test(output)

    @classmethod
    @overrides
    def from_pretrained(cls, src: str, credentials_json_path: str = None, **module_kwargs):
        src = src.lower()
        model_option_name = 'final_model'
        char_summary = 'char_summary.csv'

        if src in cls.pretrained_models:
            # check if the credentials key exists
            assert credentials_json_path, 'If using pre-trained model, you need to include the credentials key file'
            assert Path(credentials_json_path).exists(), "Credentials key file doesn't exist"

            model_dir_path = cls._get_pretrained_model_path(src, credentials_json_path)
        else:
            model_dir_path = src

        # use path model
        model_dir_path = Path(model_dir_path)

        assert model_dir_path.exists(), 'model directory \'{}\' doesn\'t exist'.format(model_dir_path)

        # setting the contents to load the data
        model_full_path = model_dir_path.joinpath(model_option_name)
        character_summary_csv_path = model_dir_path.joinpath(char_summary)
        charset = sorted(pd.read_csv(character_summary_csv_path, index_col=0)['char'].values.tolist())

        char_encoder = CharacterEncoder(data=charset)

        # This MUST BE included
        default_model_params['n_class'] = char_encoder.count + 1

        speech_model = SpeechRecognitionModel(**default_model_params)
        speech_model.load_state_dict(torch.load(str(model_full_path), map_location=torch.device('cpu')))

        test_transform = MelSpectrogram()
        return cls(speech_model, char_encoder, test_transform)
