import sys, re, json
import numpy as np

from .heuristic import SEDHeuristic

class MorphologyAnalyzer(SEDHeuristic):
    '''
    Tokenizer for Swahili adapted from the String Edit Distance Heuristic Morphology Analyzer:
    https://www.researchgate.net/publication/228566510_Refining_the_SED_heuristic_for_morpheme_discovery_Another_look_at_Swahili
    '''
    def __init__(self, morph_path=None):
        super().__init__() 
        self.wordlist = None
        self.morphemes = {}            
        self.frequencies = {}
        self.sing_stems = ['f', 'l', 'j']
        self.noun_prefix = ['u', 'm', 'wa', 'mi', 'j', 'ma', 'ki', 'vi', 'ch', 'vy', 'ji', 'me']

        if morph_path is not None:
            self.morphemes.update(self.load_from_disk(morph_path))
        
        if not self.morphemes:
            print('warning:','no morpheme templates found, please load template from path or extract templates from data for optimal tokenization')
        
        self.stems = sorted([k for k in self.morphemes.keys() if k != 'breaks'], key=len, reverse=True)

    def get_sub_words(self, templates, min_word_len):
        '''
        Helper function to get subwords from templates. Merges templates and combines substrings
        from twins and siblings of the templates since these likely contain related parts of various words
        and can thus be broken down further to get basic sub words including stems and morphemes

        Args:
            templates - word templates extracted from data
            min_word_len - minimum length of substring from templates to be considered for pairing and ultimately breaking down

        Returns:
            list of subwords
        '''
        temps = templates[0]

        #combine templates into a single dict
        for i in range(1, len(templates)):
            temps = self.merge_dicts(temps, templates[i])

        #get subwords from twins in templates
        temps = [s for gp in temps.keys() if gp not in ['orphans', 'cost'] for s in temps[gp]]

        subs = list(set([tw for pair in self.get_pairs(temps, min_word_len=min_word_len) for tw in list(self.get_twins(pair).values()) if tw not in 'NULL']))
            
        return subs
    
    def get_frequency(self, subwords):
        '''
        Helper function to get the frequency of subwords obtained from templates.
        Gets the number of times each subword occurs in the word list from which templates were extracted

        Args:
            subwords - character sequences extracted from substrings in the templates

        Returns: 
            dictionary with each subword as key and its corresponding frequency as the value
        '''
        freq_dict = {}
        
        for sw in subwords:
            freq_dict[sw] = sum(sw in word for word in self.wordlist)/len(self.wordlist)
            
        return freq_dict
    
    def get_dom_stems(self, sub_word_frequencies, cut_off_freq):
        '''
        get dominant stems from list of subwords. The dominant stem is likely the longest subword that occurs most frequently 
        in the wordlist. It is therefore longer than a standard morpheme of two chars length and as appears as frequent as
        the unique number of root-words(words without altering morphemes)

        Args:
            sub_word_frequencies - dictionary of subwords and their corresponding frequencies

        Returns:
            list of dominant stems
        '''
        dom_stems = []
        for k,v in sub_word_frequencies.items():
            if k not in self.sing_stems and len(k)>2 and v>=cut_off_freq:
                dom_stems.append(k)

            elif len(k)==1 and k in self.sing_stems:
                dom_stems.append(k)
        
        stem_twins = list(set([tw for pair in self.get_pairs(dom_stems, min_word_len=2) for tw in list(self.get_twins(pair).values()) if tw not in 'NULL']))
        dom_stems = list(set([stem for stem in dom_stems for tw in stem_twins if stem.find(tw)>0 or sub_word_frequencies[stem]>=cut_off_freq or set(stem).symmetric_difference(set(tw))<4]+stem_twins))
        dom_stems = [d for d in self.substringSieve(dom_stems) if len(d)>2]
                
        return dom_stems
    
    def get_associated_morphemes(self, templates, dom_stems, sub_words, store_path=None):
        
        '''
        extract other morphemes associated with the dominant stems. Searches for subwords from siblings and orphans of the templates
        since these likely contain the morphemes making up the words and stores them for further analysis or use in the tokenizer

        Args:
            templates - dictionary containing the templates extracted from data
            dom_stems - list of dominant stems
            sub_words - list of subwords 
            store_path - path on disk to store morphemes
        '''
        morph_dict = {}
        leftovers = []

        sibs = [sib for k in sub_words for temp in templates for sib in temp['siblings'] if k in temp['twins'] and all(k not in sib for k in sub_words) if sib not in 'NULL']\
            + [st for st in sub_words if st not in dom_stems and st != 'NULL']

        orphs = [orp for k in sub_words for temp in templates for orp in temp['orphans'] if k in temp['twins'] and all(k not in orp for k in sub_words) if orp not in 'NULL']\
            + [st for st in sub_words if st not in dom_stems and st != 'NULL']

        def map_stems(dom_stem):
            morphs = {}
            words_with_stem = [w for w in self.wordlist if dom_stem in w]

            morps = [morp for morp in list(set(sibs).union(set(orphs))) if len(morp)>3 or len(morp)==1 and morp in self.vowels or len(morp)==2 and morp[1] in self.vowels or len(morp)==3 and morp[2] in self.vowels]

            morphs['stem'] = dom_stem
            morphs['morphemes'] = [m for m in morps if sum(m in word for word in words_with_stem)/len(words_with_stem)>=0.25 and len(m)<5]
            leftovers.extend([m for m in morps if m not in morphs['morphemes']])

            return {dom_stem:morphs}

        morph_dict.update(list(map(map_stems, dom_stems))[0])
        
        morph_dict['breaks'] = leftovers
        self.morphemes.update(morph_dict)
        
        if store_path is not None:
            self.save_to_disk(morph_dict, store_path)
        
    def save_to_disk(self, items, store_path):
        '''
        store morphemes in file on disk

        Args:
            items      - collection of items
            store_path - location on disk to store morphemes
        '''
        with open(str(store_path), 'w') as file:
            json.dump(items, file)
   
    def load_from_disk(self, store_path):
        '''
        load morphemes from path on disk
        '''
        with open(str(store_path), 'r') as file:
            items = json.load(file)
        
        return items

    def analyse(self, cut_off_freq=0.25, wordlist=None, text_path=None, store_path=None):
        '''
        Run morphology analysis on data and get morphemes. Uses the templates extracted from the data
        to get morphemes that make up the words 

        Args:
            cut_off_freq - minimum occurence of morpheme for it to be considered a stem
            Wordlist  - collection of strings with words from which morphemes are to be extracted
            text_path - path to strings from which morphemes are to be extracted
            store_path - path on disk where morphemes are to be stored
            
        '''
        if text_path is not None:
            with open(text_path) as file_object:
                wordlist = [l for line in file_object for l in line.rstrip('\n').split(' ')]
        
        min_word_len = 4
        
        if isinstance(wordlist, str):
            wordlist = wordlist.split(' ')
        
        self.wordlist = wordlist
        templates = self.get_templates(wordlist)  
        
        subs = self.get_sub_words(templates, min_word_len)
        freq = self.get_frequency(subs)
        
        self.frequencies.update(freq)
        subs = list(set(subs))     
        
        dom_stems = self.get_dom_stems(freq, cut_off_freq)   
        self.get_associated_morphemes(templates, dom_stems, subs, store_path)
        
    def break_word(self, word, stems):
        '''
        find initial break of words by locating the stem, if exists in the collection of stems from analyzer, and
        breaking the word off at these points

        Args:
            word - string to be tokenized
            stems - collection of stems extrated using the morphology analyzer

        Returns:
            list containing the stem as the initial break and a list of substrings containing the rest of the character
            sequences from the word
        '''
        breaks = []
        morphs = None
        
        for stem in stems:
            #partition the word at the stem and include stem as first morpheme
            if stem in word:
                breaks.append(stem)
                init_break = word.partition(stem)
                morphs = [br for br in init_break if stem not in br and br not in '']
                break

        return breaks, morphs
        
    def get_morphemes(self, word, stems):
        '''
        break words down to stems and associated morphemes. searches for stem as initial break of word
        and uses the collection of other morphemes extracted using the morphology analyzer to search and
        find other possible breaks in the word. 

        Args:
            word - string to be tokenized
            stems - collection of stems extrated using the morphology analyzer

        Returns:
            list of morphemes composing the word 
        '''
        
        breaks, morphs = self.break_word(word, stems)

        if self.morphemes:
            brks = self.morphemes['breaks']
        else:
            brks = []
        n_class = [pre+vow for pre in self.noun_prefix for vow in self.vowels if len(pre)>1 and pre[1] not in self.vowels] 

        morphemes = [morp for stem in self.morphemes.keys() if stem != 'breaks' for morp in self.morphemes[stem]['morphemes'] + n_class if (len(morp)==2 and morp[1] in self.vowels)\
            or (len(morp)==3 and all(morp[i] not in self.vowels for i in range(2)) and morp[2] in self.vowels)]
            
                      
        if breaks:
            #get remaining partitions based on morphemes associated with the stem found in the word
            for morp in self.morphemes[breaks[0]]['morphemes']+n_class:
                if (len(morp)==2 and morp[1] in self.vowels) or (len(morp)==3 and all(morp[i] not in self.vowels for i in range(2)) and morp[2] in self.vowels):
                    
                    for i,br in enumerate(morphs):
                        brk = br.partition(morp)
                        interm_break = [ib for ib in brk if ib not in '' and ib not in [mor for mor in self.morphemes[breaks[0]]['morphemes']\
                             if len(mor)>2] and len(ib)<len(''.join(brk)) or ib in self.vowels and (word.find(morp)==0 or word.rfind(morp)==-1)]
                        
                        if interm_break:
                            breaks.append(max(interm_break))
                            morphs[i] = min(interm_break)
                            
            breaks.extend(np.setdiff1d(morphs,breaks).tolist())
            tokens = list(set(breaks))
            
            return tokens

        elif not breaks and len(word)>3:
            subs = sorted(list(set(morphemes+brks)), key=len, reverse=True)
            morphs = [s for s in subs if s in word]
            
            if morphs:
                breaks = [gp for gp in re.sub(r'|'.join(map(re.escape, morphs)), ' ', word).split(' ') if gp not in '' and len(gp)>1 or (len(gp)==1 and gp in self.vowels and word.find(gp)==0\
                     or word.rfind(gp)==len(word)-1)]
                     
            return breaks+morphs
        
        return breaks     
    
    def break_noun(self, word, stems):
        '''
        Uses predefined noun classes to find initial breaks in nouns or noun-like words and proceeds to further break the words
        down to basic morphemes

        Args:
            word  - string to be tokenized
            stems - list of stems obtained from data using the morphology analyzer

        Returns:
            list containing the corresponding noun class prefix and a list of leftover subwords
        '''

        #search for prefix denoting the noun class
        for pre in sorted(self.noun_prefix+[np+vow for np in self.noun_prefix for vow in self.vowels if np[-1] not in self.vowels], key=len, reverse=True):
            if re.search(rf"\A{pre}", word):
                breaks = [pre]
                break
        
        #any subword longer than four characters likely contains several morphemes and is therefore further broken down
        if any(len(part)>4 for part in word.partition(breaks[0])):
            init_break = []

            for part in word.partition(breaks[0]):
                if len(part)>4:
                    morp = self.get_morphemes(part, stems)

                    if morp and len(''.join(morp))>=len(part):
                        init_break.extend(morp)
                        
                    elif morp and len(''.join(morp))<len(part):
                        init_break.extend(part.partition(min(morp)))
                        
                    else:
                        init_break.append(part)

                else:
                    init_break.extend(part)
        else:
            init_break = word.partition(breaks[0])
            
        return init_break, breaks
    
    def get_breaks(self, word, stems):
        '''
        experimental quick fix for nouns. since nouns have less morphemes and forms than verbs, the algorithm is
        less effective at analyzing them.
        '''        
        init_break, breaks = self.break_noun(word, stems)
        morphs = []
        
        def map_morphs(br):
            _morphs = []
            if br.rfind('ni')==0: 
                interm_break = list(br.partition('ni'))
                return [br for br in interm_break if br != '' and len(br)>2 or (len(br)==1 and br in self.vowels and (re.search(rf'\A{br}', word) or re.search(rf'{br}\Z', word)))\
                    or (len(br)==2 and br[0] not in self.vowels)]
            else:
                interm_break = []
                return [br for br in init_break if br.find(breaks[0])!=0 and br != '' and len(br)>2 or (len(br)==1 and br in self.vowels and re.search(rf'\A{br}', word)\
                    or re.search(rf'{br}\Z', word)) or (len(br)==2 and br[0] not in self.vowels)]+interm_break

        return list(set(breaks+[ls for lr in list(map(map_morphs, init_break)) for ls in lr]))
                
    def align_tokens(self, word, tokens):
        '''
        list morphemes according to their order of appearance in the original word. This is essential for sequence based
        models and embedding algorithms based on composition functions such as CNN or LSTMs
        '''
        tok_idx = {c: [m.start() for m in re.finditer(c, word)] for c in tokens}
        idx_tup = sorted([tup for tup in [(k,idx) for k,v in tok_idx.items() for idx in v]], key= lambda x: x[1])
        tokens = [str(tok[0]) for i,tok in enumerate(idx_tup) if len(tok[0])>1 and i==0 or len(tok[0])>1 and tok[1] > idx_tup[i-1][1]+len(idx_tup[i-1][0])-1 or len(tok[0])==1 and i==0 or len(tok[0])==1 and tok[0] not in idx_tup[i-1][0]]                

        return tokens
    
    def read_from_path(self, text_path):
        '''
        read text from path on disk
        '''
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as file_object:
            text = [l for line in file_object for l in line.rstrip('\n').split(' ') if l not in '']
            
        return text

    def break_text(self, text_collection, text=None):

        def map_breaks(word):
            break_dict = {}
            tokens = None

            if not any(re.search(rf"\A{pre}", word) for pre in self.noun_prefix):#ensure the word is not a noun
                tokens = self.get_morphemes(word, self.stems)

                if not tokens:
                    break_dict[word] = word

            else:
                tokens = self.get_breaks(word, self.stems)
                
            if tokens is not None and tokens:
                break_dict[word] = self.align_tokens(word, tokens)
            
            return break_dict

        break_dict = {k:v for c in list(map(map_breaks, text_collection)) for k,v in c.items()}

        if text is not None:
            _txt = " ".join(text)
            _breaks = [k for k in break_dict.keys()]
            _tokens = self.align_tokens(_txt, _breaks)

            return list(zip(_tokens, [break_dict[tk] for tk in _tokens]))

        return break_dict
