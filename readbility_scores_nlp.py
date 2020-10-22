import nltk
nltk.download('punkt') #extra nltk resources we'll need to install

import nltk
from collections import Counter
import re

def sentence_count(txt):
  return len(nltk.sent_tokenize(txt))

for i in range(len(txt)):
    print("Text "+str(i)+" sentence count: "+str(sentence_count(txt[i])))

def tokens_no_nums(txt):
  txt = re.sub('\d', '', txt)
  tokens = nltk.word_tokenize(txt)
  words = [word for word in tokens if word.isalpha()]
  return words

def token_count(txt):
  return len(tokens_no_nums(txt))

for i in range(len(txt)):
    print("Text "+str(i)+" token count: "+str(token_count(txt[i])))

def type_count(txt):
  counter = Counter(tokens_no_nums(txt))
  return len(counter.keys())

for i in range(len(txt)):
    print("Text "+str(i)+" type count: "+str(type_count(txt[i])))

def avg_sentence_length(txt):
  return token_count(txt)/sentence_count(txt)

for i in range(len(txt)):
    print("Text "+str(i)+" average sentence length: "+str(avg_sentence_length(txt[i])))

def type_token_ratio(txt):
  return type_count(txt)/token_count(txt)

for i in range(len(txt)):
    print("Text "+str(i)+" type/token ratio: "+str(type_token_ratio(txt[i])))

# Natural Language Toolkit: Tokenizers
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Christopher Hench <chris.l.hench@gmail.com>
#         Alex Estes
# URL: <http://nltk.sourceforge.net>
# For license information, see LICENSE.TXT

"""
The Sonority Sequencing Principle (SSP) is a language agnostic algorithm proposed
by Otto Jesperson in 1904. The sonorous quality of a phoneme is judged by the
openness of the lips. Syllable breaks occur before troughs in sonority. For more
on the SSP see Selkirk (1984).

The default implementation uses the English alphabet, but the `sonority_hiearchy`
can be modified to IPA or any other alphabet for the use-case. The SSP is a
universal syllabification algorithm, but that does not mean it performs equally
across languages. Bartlett et al. (2009) is a good benchmark for English accuracy
if utilizing IPA (pg. 311).

Importantly, if a custom hiearchy is supplied and vowels span across more than
one level, they should be given separately to the `vowels` class attribute.

References:
- Otto Jespersen. 1904. Lehrbuch der Phonetik.
  Leipzig, Teubner. Chapter 13, Silbe, pp. 185-203.
- Elisabeth Selkirk. 1984. On the major class features and syllable theory.
  In Aronoff & Oehrle (eds.) Language Sound Structure: Studies in Phonology.
  Cambridge, MIT Press. pp. 107-136.
- Susan Bartlett, et al. 2009. On the Syllabification of Phonemes.
  In HLT-NAACL. pp. 308-316.
"""

import warnings

import re
from string import punctuation

from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams


class SyllableTokenizer(TokenizerI):
    """
    Syllabifies words based on the Sonority Sequencing Principle (SSP).

        >>> from nltk.tokenize import SyllableTokenizer
        >>> from nltk import word_tokenize
        >>> SSP = SyllableTokenizer()
        >>> SSP.tokenize('justification')
        ['jus', 'ti', 'fi', 'ca', 'tion']
        >>> text = "This is a foobar-like sentence."
        >>> [SSP.tokenize(token) for token in word_tokenize(text)]
        [['This'], ['is'], ['a'], ['foo', 'bar', '-', 'li', 'ke'], ['sen', 'ten', 'ce'], ['.']]
    """

    def __init__(self, lang="en", sonority_hierarchy=False):
        """
        :param lang: Language parameter, default is English, 'en'
        :type lang: str
        :param sonority_hierarchy: Sonority hierarchy according to the
                                   Sonority Sequencing Principle.
        :type sonority_hierarchy: list(str)
        """
        # Sonority hierarchy should be provided in descending order.
        # If vowels are spread across multiple levels, they should be
        # passed assigned self.vowels var together, otherwise should be
        # placed in first index of hierarchy.
        if not sonority_hierarchy and lang == "en":
            sonority_hierarchy = [
                "aeiouy",  # vowels.
                "lmnrw",  # nasals.
                "zvsf",  # fricatives.
                "bcdgtkpqxhj",  # stops.
            ]

        self.vowels = sonority_hierarchy[0]
        self.phoneme_map = {}
        for i, level in enumerate(sonority_hierarchy):
            for c in level:
                sonority_level = len(sonority_hierarchy) - i
                self.phoneme_map[c] = sonority_level
                self.phoneme_map[c.upper()] = sonority_level

    def assign_values(self, token):
        """
        Assigns each phoneme its value from the sonority hierarchy.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return: List of tuples, first element is character/phoneme and
                 second is the soronity value.
        :rtype: list(tuple(str, int))
        """
        syllables_values = []
        for c in token:
            try:
                syllables_values.append((c, self.phoneme_map[c]))
            except KeyError:
                if c not in punctuation:
                    warnings.warn(
                        "Character not defined in sonority_hierarchy,"
                        " assigning as vowel: '{}'".format(c)
                    )
                    syllables_values.append((c, max(self.phoneme_map.values())))
                    self.vowels += c
                else:  # If it's a punctuation, assing -1.
                    syllables_values.append((c, -1))
        return syllables_values


    def validate_syllables(self, syllable_list):
        """
        Ensures each syllable has at least one vowel.
        If the following syllable doesn't have vowel, add it to the current one.

        :param syllable_list: Single word or token broken up into syllables.
        :type syllable_list: list(str)
        :return: Single word or token broken up into syllables
                 (with added syllables if necessary)
        :rtype: list(str)
        """
        valid_syllables = []
        front = ""
        for i, syllable in enumerate(syllable_list):
            if syllable in punctuation:
                valid_syllables.append(syllable)
                continue
            if not re.search("|".join(self.vowels), syllable):
                if len(valid_syllables) == 0:
                    front += syllable
                else:
                    valid_syllables = valid_syllables[:-1] + [
                        valid_syllables[-1] + syllable
                    ]
            else:
                if len(valid_syllables) == 0:
                    valid_syllables.append(front + syllable)
                else:
                    valid_syllables.append(syllable)

        return valid_syllables


    def tokenize(self, token):
        """
        Apply the SSP to return a list of syllables.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return syllable_list: Single word or token broken up into syllables.
        :rtype: list(str)
        """
        # assign values from hierarchy
        syllables_values = self.assign_values(token)

        # if only one vowel return word
        if sum(token.count(x) for x in self.vowels) <= 1:
            return [token]

        syllable_list = []
        syllable = syllables_values[0][0]  # start syllable with first phoneme
        for trigram in ngrams(syllables_values, n=3):
            phonemes, values = zip(*trigram)
            # Sonority of previous, focal and following phoneme
            prev_value, focal_value, next_value = values
            # Focal phoneme.
            focal_phoneme = phonemes[1]

            # These cases trigger syllable break.
            if focal_value == -1:  # If it's a punctuation, just break.
                syllable_list.append(syllable)
                syllable_list.append(focal_phoneme)
                syllable = ""
            elif prev_value >= focal_value == next_value:
                syllable += focal_phoneme
                syllable_list.append(syllable)
                syllable = ""

            elif prev_value > focal_value < next_value:
                syllable_list.append(syllable)
                syllable = ""
                syllable += focal_phoneme

            # no syllable break
            else:
                syllable += focal_phoneme

        syllable += syllables_values[-1][0]  # append last phoneme
        syllable_list.append(syllable)

        return self.validate_syllables(syllable_list)

import cmudict
d = cmudict.dict()

def nsyl(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        st = SyllableTokenizer()
        return len(st.tokenize(word))

def syl_count(txt):
  tokens = tokens_no_nums(txt)
  syl_tokens = [nsyl(t) for t in tokens]
  return sum(syl_tokens)

for i in range(len(txt)):
    print("Text "+str(i)+" syllable count: "+str(syl_count(txt[i])))

#NUMBER OF DIFFICULT WORDS

def more_2_syl(txt):
  count = 0
  tokens = tokens_no_nums(txt)
  syl_tokens = [nsyl(t) for t in tokens]
  for s in syl_tokens:
    if s > 2:
      count += 1
  return count

for i in range(len(txt)):
    print("Text "+str(i)+" difficult word count: "+str(more_2_syl(txt[i])))

#PERCENTAGE OF DIFFICULT WORDS

def per_more_2_syl(txt):
  return 100*more_2_syl(txt)/token_count(txt)

for i in range(len(txt)):
    print("Text "+str(i)+" difficult word percentage: "+str(per_more_2_syl(txt[i])))

#AVARAGE SYLLABLES PER SENTENCE

def avg_syl_sentence(txt):
  return syl_count(txt)/sentence_count(txt)

for i in range(len(txt)):
    print("Text "+str(i)+" average syllables per sentence: "+str(avg_syl_sentence(txt[i])))

#AVARAGE SYLLABLES PER WORD

def avg_syl_word(txt):
  return syl_count(txt)/token_count(txt)

for i in range(len(txt)):
    print("Text "+str(i)+" average syllables per word: "+str(avg_syl_word(txt[i])))

#COMPARING THE RESULTS
def summary(txt):
  print('- sentence count: '+str(sentence_count(txt)))
  print('- token count: '+str(token_count(txt)))
  print('- type count: '+str(type_count(txt)))
  print('- average sentence length: '+str(avg_sentence_length(txt)))
  print('- type/token ratio: '+str(type_token_ratio(txt)))
  print('- syllable count: '+str(syl_count(txt)))
  print('- words more than 2 syllables: '+str(more_2_syl(txt)))
  print('- percentage of words more than 2 syllables: '+str(per_more_2_syl(txt)))
  print('- average syllables sentence: '+str(avg_syl_sentence(txt)))
  print('- average syllables word: '+str(avg_syl_word(txt)))

for i in range(len(txt)):
    print("TEXT "+str(i))
    print(summary(txt[i]))
    print()

#READABILITY SCORES
#FLESCH READING EASE
def flesch_reading_ease(txt):
  return 206.835 - 1.015 * (token_count(txt)/sentence_count(txt)) - 84.6 * (syl_count(txt)/token_count(txt))

for i in range(len(txt)):
    print("Text "+str(i)+" Flesch Reading Ease: "+str(flesch_reading_ease(txt[i])))

#FLESCH KINCAID GRADE

def flesch_kincaid_grade(txt):
  return 0.39 * (token_count(txt)/sentence_count(txt)) + 11.8 * (syl_count(txt)/token_count(txt)) - 15.59

for i in range(len(txt)):
    print("Text "+str(i)+" Flesch Kincaid Grade: "+str(flesch_kincaid_grade(txt[i])))

#GUNNING FOG INDEX

def gunning_fog_index(txt):
  return 0.4 * ((token_count(txt)/sentence_count(txt) + 100 * (more_2_syl(txt)/token_count(txt))))

for i in range(len(txt)):
    print("Text "+str(i)+" Gunning Fog Index: "+str(gunning_fog_index(txt[i])))

#COLLEMAN LIAU INDEX

def letter_count(txt):
  num_words = 0
  punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  new_txt = ""

  for char in txt:
   if char not in punctuations:
       new_txt = new_txt + char

  for char in new_txt:
    if char == " ":
      pass
    else:  
      num_words += 1 
  return num_words
    
def coleman_liau_index(txt):
  s = (sentence_count(txt) * (100 / token_count(txt)))
  l = (letter_count(txt) * (100 / token_count(txt)))

  return ((0.0588 * l) - (0.296 * s) - 15.8)

#COMPARING RESULTS

def readability_scores(txt):
  print('- flesch reading ease: '+str(flesch_reading_ease(txt)))
  print('- flesch kincaid grade: '+str(flesch_kincaid_grade(txt)))
  print('- gunning fog index: '+str(gunning_fog_index(txt)))
  print('- coleman liau index: '+str(coleman_liau_index(txt)))

for i in range(len(txt)):
    print('Text '+str(i))
    readability_scores(txt[i])
    print()

#MTLD

import spacy
from lexical_diversity import lex_div as ld

def mtld(txt):
  nlp = spacy.load('en')
  doc = nlp(u""+txt)
  txt = ""
  for token in doc:
    txt += (" " + token.lemma_)
  txt = tokens_no_nums(txt)
  return ld.mtld_ma_wrap(txt)

for i in range(len(txt)):
    print("Text "+str(i)+" MLTD: "+str(mtld(txt[i])))

#HDD

def hdd(txt):
  nlp = spacy.load('en')
  doc = nlp(u""+txt)
  txt = ""
  for token in doc:
    txt += (" " + token.lemma_)
  txt = tokens_no_nums(txt)
  return ld.hdd(txt)*100

for i in range(len(txt)):
    print("Text "+str(i)+" HDD: "+str(hdd(txt[i])))

#COMPARING RESULTS

def lex_diversity(txt):
  print('- VOCD: '+str(hdd(txt)))
  print('- MTLD: '+str(mtld(txt)))

for i in range(len(txt)):
    print('Text '+str(i))
    lex_diversity(txt[i])
    print()

#VOTES

import statistics

def flesch_reading_ease_vote(txt):
  resp = flesch_reading_ease(txt)
  if resp >= 90:
    return 0
  elif resp < 90 and resp >= 80:
    return 1
  elif resp < 80 and resp >= 70:
    return 2
  elif resp < 70 and resp >= 60:
    return 3
  elif resp < 60 and resp >= 50:
    return 4
  elif resp < 50 and resp >= 30:
    return 5
  else:
    return 6

def flesch_kincaid_grade_vote(txt):
  resp = flesch_kincaid_grade(txt)
  if resp <= 5.0:
    return 0
  elif resp > 5.0 and resp <= 6.0:
    return 1
  elif resp > 6.0 and resp <= 7.0:
    return 2
  elif resp > 7.0 and resp <= 9.0:
    return 3
  elif resp > 9.0 and resp <= 12.0:
    return 4
  elif resp > 12.0 and resp <= 16.0:
    return 5
  else:
    return 6

def gunning_fog_index_vote(txt):
  resp = gunning_fog_index(txt)
  if resp <= 6.0:
    return 0
  elif resp > 6.0 and resp <= 7.0:
    return 1
  elif resp > 7.0 and resp <= 8.0:
    return 2
  elif resp > 8.0 and resp <= 10.0:
    return 3
  elif resp > 10.0 and resp <= 13.0:
    return 4
  elif resp > 13.0 and resp <= 17.0:
    return 5
  else:
    return 6

def coleman_liau_index_vote(txt):
  resp = coleman_liau_index(txt)
  if resp <= 5.0:
    return 0
  elif resp > 5.0 and resp <= 6.0:
    return 1
  elif resp > 6.0 and resp <= 7.0:
    return 2
  elif resp > 7.0 and resp <= 10.0:
    return 3
  elif resp > 10.0 and resp <= 12.0:
    return 4
  elif resp > 12.0 and resp <= 16.0:
    return 5
  else:
    return 6


def vote_mean(txt):
  results = [flesch_reading_ease_vote(txt), flesch_kincaid_grade_vote(txt), gunning_fog_index_vote(txt),coleman_liau_index_vote(txt)]
  return statistics.mean(results)

def vote_mode(txt):
  results = [flesch_reading_ease_vote(txt), flesch_kincaid_grade_vote(txt), gunning_fog_index_vote(txt),coleman_liau_index_vote(txt)]
  try:
    return statistics.mode(results)
  except:
    return results[0]

def vote_median(txt):
  results = [flesch_reading_ease_vote(txt), flesch_kincaid_grade_vote(txt), gunning_fog_index_vote(txt),coleman_liau_index_vote(txt)]
  return statistics.median(results)

def vote_decode(n):
  if n <= 0:
    return "A1"
  elif n == 1:
    return "A2"
  elif n == 2:
    return "B1"
  elif n == 3:
    return "B2"
  elif n == 4:
    return "C1"
  elif n == 5:
    return "C2"
  else:
    return "Fluent"

import math
import collections

def accuracy(txt, level, vote, round):
  raw_votes = []
  for t in txt:
    if vote == "mode":
      raw_votes.append(vote_mode(t))
    elif vote == "mean":
      raw_votes.append(vote_mean(t))
    else:
      raw_votes.append(vote_median(t))
  
  round_votes = []
  for v in raw_votes:
    if round == "ceil":
      round_votes.append(math.ceil(v))
    else:
      round_votes.append(math.floor(v))
  
  results = []
  levels = []
  for v in round_votes:
    r = vote_decode(v)
    results.append(r == level)
    levels.append(r)
  
  return sum(results)/len(results), Counter(levels)

  