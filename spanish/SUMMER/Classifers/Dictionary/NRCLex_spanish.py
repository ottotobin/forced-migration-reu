#!/usr/bin/env python
"""
@author: metalcorebear

Edited by Bernardo Medeiros, Colin Hwang, Mattea Whitlow, Adeline Roza

This file was taken directly from the NRCLex github and modified to use a spanish lexicon instead of 
an english lexicon.

"""

# Determine word affect based on the NRC emotional lexicon
# Library is built on TextBlob

from textblob import TextBlob
from collections import Counter
import json


def build_word_affect(self):
    # Build word affect function
    affect_list = []
    affect_dict = dict()
    affect_frequencies = Counter()
    lexicon_keys = self.lexicon.keys()
    for word in self.words:
        if word in lexicon_keys:
            affect_list.extend(self.lexicon[word])
            affect_dict.update({word: self.lexicon[word]})
    for word in affect_list:
        affect_frequencies[word] += 1
    sum_values = sum(affect_frequencies.values())
    affect_percent = {'fear': 0.0, 'anger': 0.0, 'anticip': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0,
                      'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
    for key in affect_frequencies.keys():
        affect_percent.update({key: float(affect_frequencies[key]) / float(sum_values)})
    self.affect_list = affect_list
    self.affect_dict = affect_dict
    self.raw_emotion_scores = dict(affect_frequencies)
    self.affect_frequencies = affect_percent


def top_emotions(self):
    emo_dict = self.affect_frequencies
    max_value = max(emo_dict.values())
    top_emotions = []
    for key in emo_dict.keys():
        if emo_dict[key] == max_value:
            top_emotions.append((key, max_value))
    self.top_emotions = top_emotions


class NRCLex:
    """Lexicon source is (C) 2016 National Research Council Canada (NRC) and library is for research purposes only.  Source: http://sentiment.nrc.ca/lexicons-for-research/"""

    with open('nrc_spanish.json') as lex_file:
        lexicon = json.load(lex_file)

    def __init__(self, text):
        self.text = text
        blob = TextBlob(text)
        self.words = list(blob.words)
        self.sentences = list(blob.sentences)
        build_word_affect(self)
        top_emotions(self)

    def append_text(self, text_add):
        self.text = self.text + ' ' + text_add
        blob = TextBlob(self.text)
        self.words = list(blob.words)
        self.sentences = list(blob.sentences)
        build_word_affect(self)