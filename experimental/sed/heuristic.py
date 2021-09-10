import re
import itertools
import numpy as np

class SEDHeuristic():
    def __init__(self):        
        self.vowels = ['a', 'e', 'i', 'o', 'u']

    def get_pairs(self, wordlist, threshold = 5, min_word_len = 5, size = 9):
        '''
        Find all possible combinations of words and pair-up non-identical words

        Args:

            wordlist - list of words
            threshold - minimum number of characters needed to appear in both strings for a valid pair
            min_word_len - minimum length of string to be considered valid for pairing
            size - maximum number of non-identical characters in strings for a valid pair

        Returns:
            list of word pairs

        '''
        words = []
        wordlist = list(set(wordlist))
        for word in wordlist:
            if len(word)>min_word_len:
                words.append(word)

        word_pairs = []
        for pair in itertools.combinations(words, 2):
            intersection = set(pair[0]).intersection(set(pair[1]))
            sym_diff = set(pair[0]).symmetric_difference(set(pair[1]))

            if (len(intersection)>threshold or len(sym_diff)<size) and pair[0]!=pair[1]:
                word_pairs.append(pair)

        return word_pairs
    
    def get_twins(self, pair):
        '''
        Find substrings in pairs of strings with identical sequence of characters(twins) using string edit distance
        Adapted from https://www.researchgate.net/publication/228566510_Refining_the_SED_heuristic_for_morpheme_discovery_Another_look_at_Swahili 

        Args:
            pair - pair of strings to be compared and aligned

        Returns:
            Dictionary with twins as values and their indices as keys
        '''
        s1, s2 = pair[0], pair[1]
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        twins = {}
        twin_count = 0
        twins[twin_count] = list()

        c = np.Inf
        distances = range(len(s1) + 1)

        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]

            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])

                    if distances_[-1]<=distances_[-2]:#distance is lower for twin characters
                        if (s1[i1:i1+2]==s2[i2:i2+2]) or (s1[i1-1:i1+1]==s2[i2-1:i2+1]):#check context of twin characters in each string
                            twins[twin_count] += c2
                            twins[twin_count] = ''.join(twins[twin_count])
                            c = i2

                else:
                    if i1==len(s1)-1 and twin_count==0 and not twins[twin_count]:#create NULL production if no twin for any of the first characters
                            twins[twin_count] = 'NULL'
                            c = i2-1

                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))

                    if i1==len(s1)-1 and i2-c==1:#create new key for next twin
                        twin_count = len(twins)
                        twins[twin_count] = list()

            distances = distances_

        for i in range(len(twins)):
            #remove empty lists
            if not twins[i]:
                del twins[i]

        return twins
    
    def get_cost(self, alignment):
        '''
        calculate the cost of having characters that are not aligned in the pairs of words

        Args:
            alignment - dictionary containing the twins, siblings, and orphans extracted from a pair of words

        Returns:
            alignment dictionary with calculated cost
        '''
        if len(alignment['siblings'])==1 and alignment['siblings'][0] != 'NULL':
            sib_cost = 0

        else:
            sib_cost = (len(alignment['siblings']))*1.5

        orph_cost = len(''.join(alignment['orphans']))

        alignment['cost'] = sib_cost+orph_cost
                
    def get_orphans(self, s_rems, twins, siblings):
        '''
        extract characters in one string that have no alignment with any other in separate string from word pair.
        searches through what's left once twins are removed from strings and extracts character sequences in one word
        that don't have any correspondence with those in the other word

        Args:
            s_rems   - the remaining character sequences once the twins are removed from the pairs of words
            twins    - identical character sequences extracted from both strings in word pair
            siblings - non-identical character sequences extracted from word pair

        Returns:
            list of orphans
        '''
        orphs = []
        for s_rem in s_rems:
            #get orphans
            if siblings:
                orp = re.sub(r'|'.join(map(re.escape, sorted(siblings))), '', s_rem)
                orphs.append(orp)

        orphs = [orp for orp in orphs if (orp not in ''.join(siblings) and orp not in self.vowels and orp not in ''.join(twins)) or (orp in ''.join(siblings) and orp in self.vowels)]
        
        return orphs
        
    def get_siblings(self, twins, s_rems, pair):
        '''
        extract character sequence that occurs at the same index in both strings but contains non-identical characters.
        combs through remaining character sequences after twins are extacted for where characters in both strings appear to align 
        though are different from each other

        Args:
            twins    - identical character sequences extracted from both strings in word pair
            s_rems   - the remaining character sequences once the twins are removed from the pairs of words
            pair     - word pair

        Returns:
            list of siblings
        '''
        sibs = []
        temp_sib = []
        left_branch, right_branch = [], []
        
        limit = min(len(s_rem) for s_rem in s_rems)

        for twin in twins:
            if twin != 'NULL':
                
                #get siblings from right and left branch of twins to ensure accurate morpheme break and minimize duplication
                for s in pair:
                    #get left and right branches of twins
                    if twin in ''.join(self.vowels) and len(twin)==1 and twin in ''.join([tw for tw in twins if tw != twin]):
                        s = [part for part in s.partition(''.join([tw for tw in twins if tw != twin])) if part not in '' and part not in twins]

                        if not s:
                            continue

                        s = s[0]

                    twin_pos = s.find(twin)

                    if s[:twin_pos] not in '':
                        left_branch.append(s[:twin_pos])

                    if s[twin_pos+len(twin):] not in '':
                        right_branch.append(s[twin_pos+len(twin):])
                        
                #get siblings from right banch
                if right_branch:
                    if len(right_branch)>1:
                        distance = abs(len(right_branch[0])-len(right_branch[1]))

                        if right_branch[0]>right_branch[1]:
                            right_branch[0], right_branch[1] = right_branch[1], right_branch[0]

                    else:
                        distance = len(right_branch)

                    if len(twin)>1 or twin not in self.vowels:
                        
                        sib = [right_branch[i][:distance] if distance>=2 else right_branch[i][:limit] if limit>=2 else right_branch[i][:2] for i in range(len(right_branch)) if right_branch[i] not in '']

                    elif len(twin)==1 and twin in self.vowels:
                        
                        sib = [part for i in range(len(right_branch)) for part in right_branch[i].partition(twin) if len(right_branch[i])>2 and len(part)>1 if part not in '' and part not in twins]

                    sibs.extend(sib)
                    sibs = list(set(sibs))

                #get siblings from left branch
                if left_branch:
                    for lb in left_branch:
                        temp_sib.extend(re.sub(r'|'.join(map(re.escape, sorted(twins))), ' ', lb).split(' '))
                        sib = [sib[:limit] if any(twin in lb for twin in twins) else sib[-limit:] for sib in temp_sib if sib not in '']

                        if not sib:
                            continue

                        if sib[-1] not in ''.join(sibs) and sib[-1] not in ''.join(self.vowels):
                            sibs.extend(sib)

                    sibs = list(set(sibs))

            else:
                sibs.append('NULL')
                
        return self.substringSieve(sibs, 2)
        
    def get_alignments(self, pair):
        '''
        Get twins, siblings, and orphans from word pairs. 
        Adapted from https://www.researchgate.net/publication/228566510_Refining_the_SED_heuristic_for_morpheme_discovery_Another_look_at_Swahili 
        
        Args:
            pair - word pair

        Returns:
            Dictionary of twins, siblings, orphans, and associated cost
        '''
        alignments = {}
        
        twins = list(self.get_twins(pair).values())
        s_rems = [re.sub(r'|'.join(map(re.escape, twins)), '', pair[0]), re.sub(r'|'.join(map(re.escape, twins)), '', pair[1])]

        sibs = self.get_siblings(twins, s_rems, pair)                
        orphs = self.get_orphans(s_rems, twins, sibs)

        alignments['twins'] = twins
        alignments['siblings'] = sibs
        alignments['orphans'] = orphs
        
        self.get_cost(alignments)

        return alignments
    
    def merge_dicts(self, dic1, dic2):
        '''
        Helper function to merge dictionaries by appending values from identical keys into one

        Args:
            dic1, dic2 - dictionaries to be merged

        Returns:
            dictionary with the values from two separate dictionaries 
        '''
        d = {}
        
        for k in dic1.keys():
            d[k] = list(d[k] for d in [dic1,dic2])
            
            if k != 'cost':
                d[k] = list(set([item for sublist in d[k] for item in sublist]))
                
        self.get_cost(d)
        
        return d
    
    def get_templates(self, wordlist, return_pairs=False, pair_thresh=None, min_word_len=None):
        '''
        Get templates by collapsing similar alignments. Alignments with similar list of siblings
        are collapsed into one forming a template that represents the set of word pairs from which
        the alignments were created

        Args:
            wordlist     - collection of strings from which alignments are extracted
            return_pairs - whether or not to return word pairs corresponding to specific template
            min_word_len - minimum length of words to be considered for pairing

        Returns:
            Dictionary with twins, siblings, orphans, and cost of associated words
        '''
        
        if isinstance(wordlist, str):
            wordlist = wordlist.split(' ')

        if min_word_len is not None:
            pairs = self.get_pairs(wordlist, threshold=pair_thresh, min_word_len=min_word_len)
            
        else:
            pairs = self.get_pairs(wordlist)

        templates = []
        results = []

        for pair in pairs:
            temp_pair = {}
            alignment = self.get_alignments(pair)

            if not templates:
                templates.append(alignment)

            else:
                #cycle through templates to check whether an alignment is to be merged to an existing template or added as a new one
                for i in range(len(templates)):

                    #check whether an alignment matches any of the templates
                    if (alignment['cost'] == templates[i]['cost'] or len(''.join(alignment['twins'])) == len(''.join(templates[i]['twins']))) and len(alignment['twins'])==len(templates[i]['twins']):

                        #merge existing template with alignment and replace it
                        alignment = self.merge_dicts(alignment, templates[i])
                        templates[i] = alignment

                    #if doesn't match ensure alignment doesn't match last entry and add as new template
                    elif templates[-1]!=alignment:
                        templates.append(alignment)
            
            #add alignment and coresponding word pairs to dict
            temp_pair['pair'] = pair
            temp_pair['template'] = alignment

            #add templates to output dictionary
            if results:
                #check whether siblings in considered template as well as those in alignment are in any of all the siblings in the results
                #if true it indicates that the current word pair already has a corresponding template in the results
                #the pairs are then updated and the template updated with the new collapsed alignments
                sibs_k = [val for sublist in [results[k]['template']['siblings'] for k in range(len(results))] for val in sublist]#siblings in all templates
                
                if (set(alignment['siblings']).issubset(sibs_k)):
                    temp_results = [res for res in results if set(alignment['siblings']).issubset(res['template']['siblings'])]
                    results = list(itertools.filterfalse(lambda x: x in results, temp_results)) + list(itertools.filterfalse(lambda x: x in temp_results, results))
                    
                    for j in range(len(temp_results)):
                        #for first alignment matching an existing template the word pair would include a single tuple of strings
                        if isinstance(temp_results[j]['pair'][0], str):
                            temp_results[j]['pair'] = (temp_results[j]['pair'], pair,)

                        else:
                            temp_results[j]['pair'] = temp_results[j]['pair'] + (pair,)
                        temp_results[j]['template'] = alignment
                    results.extend(temp_results)
                    
                #alignments with no matching siblings in the existing templates are added as new templates in the results
                else:
                    results.append(temp_pair)     

            elif not results:
                results.append(temp_pair)

        #filter duplicate templates from results
        res_dup = [results[l] for l in range(len(results)) if results[l]['template'] not in [res_temp['template'] for res_temp in results[l+1:]]]

        if return_pairs:
            return [results[l]['template'] for l in range(len(results))], results

        return [res_dup[l]['template'] for l in range(len(res_dup))]
    
    def substringSieve(self, string_list, min_sub_size=3):
        '''
        Helper function to filter substrings of other strings in a list of strings

        Args:
            string_list - list of strings to be considered
            min_sub_size - minimum size of substring to be included in sieved list if is part of another

        Returns:
            list of filtered strings
        '''
        string_list.sort(key=lambda s: len(s), reverse=True)
        out = []
        
        for s in string_list:
            if not any([s in o for o in out]):
                out.append(s)
                
            elif any([s in o for o in out]) and len(s)>=min_sub_size and min_sub_size>2:
                out.append(s)
                
        return out
     