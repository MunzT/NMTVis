import math
import os
import re

from spacy.language import Language

import data
from shared import UPLOAD_FOLDER


class Translation:
    def __init__(self, words=None, log_probs=None, attns=None, candidates=None, is_golden=False, is_unk=None):
        self.words = words
        self.log_probs = log_probs
        self.attns = attns
        self.candidates = candidates
        self.is_golden = is_golden
        self.is_unk = is_unk


    def slice(self):
        return Translation(self.words[1:], self.log_probs[1:], self.attns[1:], self.candidates[1:], self.is_golden,
                           self.is_unk[1:])


    @classmethod
    def from_hypothesis(cls, hypothesis):
        translation = Translation()

        translation.words = hypothesis.words
        translation.log_probs = hypothesis.log_probs
        translation.attns = hypothesis.attns
        translation.candidates = hypothesis.candidates
        translation.is_golden = hypothesis.is_golden
        translation.is_unk = hypothesis.is_unk

        return translation


class Hypothesis:
    def __init__(self, tokens, words, log_probs, state, last_attn_vector, attns=None, candidates=None, is_unk=None):
        self.tokens = tokens
        self.words = words
        self.log_probs = log_probs
        self.state = state
        self.last_attn_vector = last_attn_vector
        self.attns = [[]] if attns is None else attns
        # candidate tokens at each search step
        self.candidates = candidates
        self.is_golden = False
        self.is_unk = is_unk
        self.alpha = 0.6
        self.beta = 0.4


    @property
    def latest_token(self):
        return self.tokens[-1]


    def __len__(self):
        return len(self.tokens)


    @property
    def log_prob(self):
        return sum(self.log_probs)


    def extend(self, token, word, new_log_prob, new_state, last_attn_vector, attn, candidates, is_unk):
        h = Hypothesis(self.tokens + [token], self.words + [word], self.log_probs + [new_log_prob], new_state,
                       last_attn_vector, self.attns + attn, self.candidates + [candidates], self.is_unk + [is_unk])
        h.alpha = self.alpha
        h.beta = self.beta

        return h


    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))


    def __repr__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))


    def score(self):
        return self.log_prob / self.length_norm() + self.coverage_norm()


    def length_norm(self):
        return (5 + len(self.tokens)) ** self.alpha / (5 + 1) ** self.alpha


    def coverage_norm(self):
        # See http://opennmt.net/OpenNMT/translation/beam_search/

        # -1 for SOS token
        X = len(self.attns[0])
        Y = len(self.attns)

        res = 0
        for i in range(X):
            sum_ = 0
            for j in range(Y-1):
                sum_ += self.attns[j][i]
            res += math.log(min(1, sum_)) if sum_ > 0 else 0

        return self.beta * res


def _is_wordlike(tok):
    return tok.orth_ and tok.orth_[0].isalpha()


@Language.component('sentence_division_suppresor')
def sentence_division_suppresor(doc):
    """Spacy pipeline component that prohibits sentence segmentation between two tokens that start with a letter.
    Useful for taming overzealous sentence segmentation in German model, possibly others as well."""
    for i, tok in enumerate(doc[:-1]):
        if _is_wordlike(tok) and _is_wordlike(doc[i + 1]):
            doc[i + 1].is_sent_start = False
    return doc


class Sentence:
    EXPERIMENT_TYPES = ["plain", "beam"]

    def __init__(self, id, source, translation, attention, beam, score):
        self.id = id
        self.source = source
        self.translation = translation
        self.attention = attention
        self.beam = beam
        self.score = score
        self.corrected = False
        self.diff = ""
        self.flagged = False
        self.experiment_metrics = None
        self.experiment_type = "BEAM"


class Document:
    def __init__(self, id, name, unk_map, filepath):
        self.id = id
        self.name = name
        self.sentences = []
        self.keyphrases = []
        self.unk_map = unk_map
        self.filepath = filepath


    def pad_punctuation(self, s):
        s = re.sub('([.,!?()])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        return s


    def replace(self, sentence):
        return sentence\
            .replace('"', "&quot;")\
            .replace("'", "&apos;")\
            .replace('„', "&quot;")\
            .replace('“', "&quot;")


    def load_content(self, filename):
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        with open(os.path.join(UPLOAD_FOLDER, filename), "r", encoding='utf-8') as f:
            content = f.read()
            doc = nlp(content)

            sentences = content.split("\n")
            content = []

            for sent in sentences:
                tokens = nlp(str(sent))
                tokens = [self.replace(str(token).strip()) for token in tokens if not str(token).isspace()]
                sentence = " ".join(tokens)
                sentence = data.bpe.process_line(sentence)
                content.append(sentence)

        return content
