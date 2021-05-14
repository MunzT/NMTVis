import os.path
import pickle
from collections import Counter

from seq2seq.data_loader import LanguagePairLoader


class DomainSpecificExtractor:
    def __init__(self, source_file, src_lang, tgt_lang, train_source_file, train_vocab_file, frequency_threshold=5):
        self.source_file = source_file
        self.frequency_threshold = frequency_threshold
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if train_source_file:
            self.train_source_file = train_source_file
        if train_vocab_file:
            self.train_vocab_file = train_vocab_file


    def extract_keyphrases(self, n_results=20):
        train_vocab = None
        if os.path.isfile(self.train_vocab_file):
            train_vocab = pickle.load(open(self.train_vocab_file, "rb"))
        else:
            train_vocab = Counter()
            train_loader = LanguagePairLoader(self.src_lang, self.tgt_lang, self.train_source_file, self.train_source_file)
            train_in, train_out, train_pairs = train_loader.load()
            for source, _ in train_pairs:
                for word in source.replace("@@ ", "").split(" "):
                    train_vocab[word] += 1
            pickle.dump(train_vocab, open(self.train_vocab_file, "wb"))

        loader = LanguagePairLoader(self.src_lang, self.tgt_lang, self.source_file, self.source_file)
        in_lang, _, pairs = loader.load()

        domain_words = []
        for word in in_lang.word2count:
            if train_vocab[word] < self.frequency_threshold and in_lang.word2count[word] > 0:
                freq = 0
                for source, _ in pairs:
                    if word.lower() in source.lower():
                        freq += 1
                domain_words.append((word, freq))

        domain_words = sorted(domain_words, key=lambda w: in_lang.word2count[w[0]], reverse=True)
        return domain_words[:n_results]