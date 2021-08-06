import sys, re
from collections import Counter
from text.utils.morphology import MorphologyAnalyzer

class Document():
    """
    The Document is the main structure of our NLP process. It wraps the document being treated.
    A Document is composed of sentences and is iterable.

    Attributes
    ----------
    raw : str
        The raw text string passed as input to be sentencized.
    sentences : list of Sentence
        The list of sentences after sentencizing.
    """

    def __init__(self, sentence_pieces, document_text=None, morph_path=None, document_path=None):
        """
        Parameters
        ----------
        document_text : str
            Text to be sentencized. Initialization immediately sentencizes the input text based on the input parameters. 
            The sentences are also immediately tokenized.
        """
        if document_path is not None:
            with open(document_path, 'r', encoding='utf-8', errors='ignore') as file:
                document_text=file.read()
        else:
            assert document_text is not None, "please specify text document"
            
        self.raw = document_text

        if morph_path is not None:
            self.morph_analyzer = MorphologyAnalyzer(morph_path)
            
        self.sentences: List[str] = sentencize(self.raw, sentence_pieces) if isinstance(self.raw, str) else [sentencize(text, sentence_pieces) for text in self.raw]
        self._index = 0

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, key):
        if isinstance(self.raw, str):
            return self.get_sentences(self.sentences, self.raw, key)

        return [self.get_sentences(sentences,  self.raw[key]) for sentences in self.sentences[key]]

    def get_sentences(self, sentences, raw_text, key=None):
        sent = sentences[key] if key is not None else sentences

        start_pos = raw_text.find(sent)
        end_pos = start_pos+len(sent)
        
        return Sentence(start_pos, end_pos, raw_text, self.morph_analyzer)

    def __repr__(self):
        return self.raw if isinstance(self.raw, str) else ' '.join(self.raw)

    def __str__(self):
        return self.raw if isinstance(self.raw, str) else ' '.join(self.raw)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.sentences):
            sent = self.sentences[self._index]
            self._index+=1
            start_pos = self.raw.find(sent)
            end_pos = start_pos+len(sent)

            return Sentence(start_pos, end_pos, self.raw, self.morph_analyzer)
        raise StopIteration


class Sentence():
    """
    Sentences are divisions of a Document. They are usually separated by punctuations. 
    Sentences are divided into words. One can iterate over sentence words.

    Attributes
    ----------
    start_pos: int
        The starting position of the sentence in the raw Document.
    end_pos: int
        The ending position of the sentence in the raw Document. Includes punctuation position.
    previous_sentence: Sentence or None
        A pointer to the previous Sentence in a linked list manner. Prepared for future navigation.
    next_sentence: Sentence or None
        A pointer to the next Sentence in a linked list manner. Prepared for future navigation.
    words : list of word
        The list of words after tokenizing.
    Methods
    -------
    get: str
        Returns the string representation of the Sentence.
    """

    def __init__(self, start_position, end_position, raw_document_reference, tokenizer=None):
        """
        Parameters
        ----------
        start_position : int
            The starting position of the sentence in the raw Document.
        end_position : int
            The ending position of the sentence in the raw Document. Includes punctuation position.
        raw_document_reference: string
            The raw document string where the sentence is localized.
        tokenizer : path or str
            Tokenizer for sub word breakdown of text
        """

        self.start_pos = int(start_position)
        self.end_pos = int(end_position)
        self._document_string = raw_document_reference
        self.words = split_words(self._document_string[self.start_pos:self.end_pos])
        self.morph_analyzer = tokenizer
        self.return_morphemes = self.morph_analyzer is not None
        self._index = 0

    def __len__(self):
        return len(self.words)

    def get(self):
        return self._document_string[self.start_pos:self.end_pos]

    def __getitem__(self, key):
        if self.return_morphemes:            
            return {str(self.words[key]):self.morph_analyzer.break_text([str(w) for w in self.words if "" != str(w)])[str(self.words[key])]}

        return str(self.words[key])

    def __repr__(self):
        return self.get()

    def __str__(self):
        return self.get()

    def __eq__(self, other):
        return self.get() == other

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.words):
            if self.return_morphemes:  
                result = {str(self.words[self._index]):self.morph_analyzer.break_text([str(w) for w in self.words if "" != str(w)])[str(self.words[self._index])]}

            else:
                result = str(self.words[self._index])

            self._index+=1
            return result
        raise StopIteration

class Word():
    """
    Words are divisions of a Sentence. They are usually separated by whitespaces.
    Attributes
    ----------
    start_pos: int
        The starting position of the word in the Sentence.

    end_pos: int
        The ending position of the word in the Sentence.

    previous_word: word or None
        A pointer to the previous word in a linked list manner. Prepared for future navigation. 
        First word in a sentence is always SOS - acronym for Start of Sentence.

    next_word: word or None
        A pointer to the next word in a linked list manner. Prepared for future navigation. 
        Last word in a sentence is always EOS - acronym for End of Sentence.

    SOS: boolean
        Is the word the start of a Sentence?

    EOS: boolean
        Is the word the end of a Sentence?

    Methods
    -------
    get: str
        Returns the string representation of the word.
    """

    def __init__(self, start_position, end_position, raw_sentence_reference, SOS = False, EOS = False):
        """
        Parameters
        ----------
        start_position : int
            The starting position of the word in the Sentence.

        end_position : int
            The ending position of the word in the Sentence.

        raw_sentence_reference: str
            The Sentence string where the word is localized. It may be a substring of a raw Document representation.

        SOS: boolean, optional
            Creates a Start of Sentence word. This influence word representation and printing.

        EOS: boolean, optional
            Creates a End of Sentence word. This influence word representation and printing.

        """

        self.start_pos = int(start_position)
        self.end_pos = int(end_position)
        self._sentence_string = raw_sentence_reference
        self.SOS = SOS
        self.EOS = EOS

    def get(self):
        if self.SOS:
            return '<SOS>'
        elif self.EOS:
            return '<EOS>'

        return self._sentence_string[self.start_pos:self.end_pos]

    def __repr__(self):
        return self.get()

    def __str__(self):
        return self.get()

    def __eq__(self, other):
        return self.get() == other

## Static Functions

def sentencize(
    raw_input_document,
    sentence_pieces,
     sentence_boundaries = ['/n'],#['(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)','\.{2,}','\!+','\:+','\?+'],
      delimiter_token='<SPLIT>'):
    """
    Sentencizes a string based on sentence boundaries. Returns a list of Sentences.
    Parameters
    ----------
    raw_input_document: str
        The raw input document in a string format.

    sentence_boundaries: list of str, optional
        A list of regex used to delimit sentence boundaries. Default regex includes correct period splitting, reticences, exclamation mark, question mark and colons.
        The default can be accessed by the global variable DEFAULT_SENTENCE_BOUNDARIES.

    delimiter_token: str, optional
        The word used for document segmentation. Usually a "agnostic" word. Defaults to <SPLIT>.
    """
    working_document = raw_input_document
    punctuation_patterns = sentence_boundaries

    for punct in punctuation_patterns:
        working_document = re.sub(punct, '\g<0>'+delimiter_token, working_document, flags=re.UNICODE)
    
    list_of_string_sentences = [x.replace('<s>','').replace('</s>','').strip() for x in working_document.split(delimiter_token) if x.strip() != ""]
    
    return split_sentences(list_of_string_sentences, sentence_pieces)

def split_sentences(sentences:list, sentence_pieces:list):
    if len(sentences)==1:
        sentences = sentences[0].split('\n')

    len_count = Counter([len(s.strip().split()) for s in sentences])

    modal_sentence_len = [k for k,v in len_count.items() if v==max(len_count.values())][0]
    modal_sentences = [sent.strip() for sent in sentences if len(sent.strip().split())<=modal_sentence_len]

    xl_sentences = [sent.strip() for sent in sentences if len(sent.strip().split())>modal_sentence_len]
    xl_sentences = [' '.join(sent.split()[i:i+modal_sentence_len]) for sent in xl_sentences for i in list(range(0,len(sent.split()),modal_sentence_len))]

    sentences = modal_sentences+xl_sentences

    return [sent for txt in sentences\
                    for sent in [' '.join(txt.strip().split()[:int(len(txt.strip().split())*j)])\
                            if len(' '.join(txt.strip().split()[:int(len(txt.strip().split())*j)]))>3\
                                else txt for j in set([i/sentence_piece for sentence_piece in sentence_pieces for i in range(1, sentence_piece+1)])\
                                    if len(txt.strip().split())>6]]
    
def split_words(
    raw_input_sentence, 
    join_split_text = True, 
    split_text_char = '\-', 
    punctuation_patterns= ['(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)','\.{2,}','\!+','\:+','\?+','\,+', r'\(|\)|\[|\]|\{|\}|\<|\>'], 
    split_characters = r'\s|\t|\n|\r', 
    delimiter_token='<SPLIT>',
    start_end = False    
    ):
    """
    Tokenizes a string based on word boundaries. Returns a list of words.
    Parameters
    ----------
    raw_input_sentence: str
        The raw input Sentence in a string format.

    join_split_text: boolean, optional
        Wheter to try to join multi-line text splits, like in the case of "sen-\ntence". Defaults to True.

    split_text_char: str, optional
        The split char used for checking and joining split strings. Defaults to hyphen.

    punctuation_patterns: list of str, optional
        A list of regex used to turn punctuations into words. Aside from sentence boundaries, also includes commas and parenthesis.
        The default can be accessed by the global variable DEFAULT_PUNCTUATIONS.

    split_characters: str, optional
        A string with regex for split characters. These are used to do tokenization after the sentence is preprocessed.
        Defaults to any whitespace (\s), any tab char (\t), newlines (\n) and carriage returns (\r).

    delimiter_token: str, optional
        The token used for sentence segmentation. Usually a "agnostic" token. Defaults to <SPLIT>.

    """
    
    working_sentence = raw_input_sentence
    #First deal with possible word splits:
    if join_split_text:
        working_sentence = re.sub('[a-z]+('+split_text_char+'[\n])[a-z]+','', working_sentence)
    
    #Escape punctuation
    for punct in punctuation_patterns:
        working_sentence = re.sub(punct, " \g<0> ", working_sentence)
    
    #Split at any split_characters
    working_sentence = re.sub(split_characters, delimiter_token, working_sentence)
    list_of_word_strings = [x.strip() for x in working_sentence.split(delimiter_token) if x.strip() !=""]
    
    previous = Word(0,0,raw_input_sentence, SOS=True)
    list_of_words = [previous]

    def map_words(word):
        
        start_pos = raw_input_sentence.find(word)
        end_pos = start_pos+len(word)

        return Word(start_pos,end_pos,raw_input_sentence)

    if start_end:
        list_of_words += list(map(map_words, list_of_word_strings))

        previous = list_of_words[-1]

        if previous.SOS != True:
            eos = Word(len(raw_input_sentence), len(raw_input_sentence), raw_input_sentence, EOS=True)

            previous.next_word=eos
            eos.previous_word = previous
            
            list_of_words.append(eos)
            
    else:
        list_of_words = list(map(map_words, list_of_word_strings))
        
    return list_of_words