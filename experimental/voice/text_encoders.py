import torch
import re
from typing import List


class CharacterEncoder(object):
    """Maps characters in a string as sequence of characters
    the space character should be <SPACE>
    Args:
        file_name (str): The path of the csv file that contains
            the characters to map
        data (`List[str]`): The list of characters that are
            to be used for mapping
    """

    # to indicate space
    _SPACE_ = '<SPACE>'

    def __init__(self, data: List[str] = None):
        self.char2ix = dict(zip(data, range(len(data) + 1)))
        self.ix2char = {v: k for k, v in self.char2ix.items()}

    def encode(self, text: str) -> List[int]:
        """
        Use a character map and convert text to an integer sequence.
        Notice that spaces in the text are also encoded with 1.

        args: text - text to be encoded
        """
        text = text.strip()

        characters = [c if not re.match(r'\s', c) else self._SPACE_ for c in list(text)]
        return [self.char2ix[c] for c in characters]

    def decode(self, indices: List[int]) -> str:
        """
        Use a character map and convert integer labels to a text sequence.
        It converts each integer into its corresponding char and joins the chars to form strings.
        Notice that the strings will be separated wherever the integer 1 appears.

        args: labels - integer values to be converted to texts(chars)
        """
        characters = [self.ix2char[ix] for ix in indices]
        return "".join([c if c != self._SPACE_ else ' ' for c in characters])

    @property
    def BLANK_INDEX(self):
        return self.count

    @property
    def count(self):
        """Returns the number of characters"""
        return len(self.char2ix)


class GreedyEncoder(object):
    def __init__(self, text_encoder: CharacterEncoder):
        self.text_encoder = text_encoder

    def decode_target(self,
                      output: torch.Tensor,
                      labels: torch.Tensor,
                      label_lengths,
                      collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        blank_label = self.text_encoder.BLANK_INDEX

        for i, args in enumerate(arg_maxes):
            decoded = []
            decode_text = self.text_encoder.decode(labels[i][:label_lengths[i]].tolist())
            targets.append(decode_text)
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j -1]:
                        continue

                    decoded.append(index.item())

            decodes.append(self.text_encoder.decode(decoded))

        return decodes, targets

    def decode_test(self,
                    output,
                    collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        blank_label = self.text_encoder.BLANK_INDEX

        for i, args in enumerate(arg_maxes):
            decoded = []
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decoded.append(index.item())

            decodes.append(self.text_encoder.decode(decoded))

        return decodes
