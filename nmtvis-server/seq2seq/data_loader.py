import re
import unicodedata
from collections import OrderedDict

import data
from seq2seq.hp import PAD_token, SOS_token, EOS_token, UNK_token, UNK_text, MIN_LENGTH


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class MyDict(OrderedDict):
    def __missing__(self, key):
        return UNK_token


class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = MyDict({data.BLANK_WORD: PAD_token, data.BOS_WORD: SOS_token, data.EOS_WORD: EOS_token, UNK_text: UNK_token})
        self.word2count = OrderedDict({data.BLANK_WORD: 0, data.BOS_WORD: 0, data.EOS_WORD: 0, UNK_text: 0})
        self.index2word = OrderedDict(
            {PAD_token: data.BLANK_WORD, SOS_token: data.BOS_WORD, EOS_token: data.EOS_WORD, UNK_token: UNK_text})
        self.n_words = 4  # Count default tokens


    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)


    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


    def trim_top(self, vocab_size=50000):
        words = list(self.word2count.keys())
        words.sort(key=lambda w: self.word2count[w], reverse=True)
        words = words[:vocab_size]

        print('keep_words %s / %s = %.4f' % (
            len(words), len(self.word2index), len(words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = MyDict({data.BLANK_WORD: PAD_token, data.BOS_WORD: SOS_token, data.EOS_WORD: EOS_token, UNK_text: UNK_token})
        word2count = OrderedDict({word: self.word2count[word] for word in words})
        self.index2word = OrderedDict(
            {PAD_token: data.BLANK_WORD, SOS_token: data.BOS_WORD, EOS_token: data.EOS_WORD, UNK_token: UNK_text})
        self.n_words = 4  # Count default tokens

        for word in words:
            self.index_word(word)
        self.word2count = word2count


    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = MyDict({"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "UNK": UNK_token})
        self.word2count = OrderedDict()
        self.index2word = OrderedDict({PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"})
        self.n_words = 4  # Count default tokens

        for word in keep_words:
            self.index_word(word)


class LanguagePairLoader:
    def __init__(self, source_lang, target_lang, source_file,  target_file, trim=True):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_file = source_file
        self.target_file = target_file
        self.trim = trim


    def load(self):
        input_lang, output_lang, pairs = self.prepare_data()

        if self.trim:
            input_lang.trim_top()
            output_lang.trim_top()

        # pairs = self.filter(input_lang, output_lang, pairs)

        return input_lang, output_lang, pairs


    def filter_pairs(self, pairs):
        filtered_pairs = []
        for pair in pairs:
            if len(pair[0].split(" ")) >= MIN_LENGTH and len(pair[0].split(" ")) <= data.MAX_LEN \
                    and len(pair[1].split(" ")) >= MIN_LENGTH and len(pair[1].split(" ")) <= data.MAX_LEN:
                filtered_pairs.append(pair)
        return filtered_pairs


    def prepare_data(self):
        input_lang, output_lang, pairs = self.read_langs()
        print("Read %d sentence pairs" % len(pairs))

        pairs = self.filter_pairs(pairs)
        print("Filtered to %d pairs" % len(pairs))

        print("Indexing words...")
        for pair in pairs:
            input_lang.index_words(pair[0])
            output_lang.index_words(pair[1])

        print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
        return input_lang, output_lang, pairs


    def read_langs(self):
        print("Reading lines...")

        # Read the file and split into lines
        source_lines = open(self.source_file).read().strip().split('\n')
        target_lines = open(self.target_file).read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = list(zip(source_lines, target_lines))

        input_lang = Lang(self.source_lang)
        output_lang = Lang(self.target_lang)

        return input_lang, output_lang, pairs


    def filter(self, input_lang, output_lang, pairs):
        keep_pairs = []

        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True

            for word in input_sentence.split(' '):
                if word not in input_lang.word2index:
                    keep_input = False
                    break

            for word in output_sentence.split(' '):
                if word not in output_lang.word2index:
                    keep_output = False
                    break

            # Remove if pair doesn't match input and output conditions
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print(
            "Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs
