import codecs
import os
import torch
from subword_nmt.apply_bpe import BPE
from torchtext import data, datasets

import shared

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
MAX_LEN = 350
MIN_VOCAB_FREQ = 1

tokenizer_fun = lambda s: s.split()
SRC = data.Field(pad_token=BLANK_WORD, batch_first=True, tokenize=tokenizer_fun)
TGT = data.Field(init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD, batch_first=True, tokenize=tokenizer_fun)

def load_dataset(src_lang: str, tgt_lang: str, min_length: int = 0):
    print("Loading dataset...")
    train, val, test = datasets.WMT14.splits(root=os.path.abspath(os.path.join(shared.DATA_FOLDER)),
                            exts=(f'.{src_lang}', f'.{tgt_lang}'), fields=(SRC, TGT),
                            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN and
                                                  len(vars(x)['src']) >= min_length and len(vars(x)['trg']) >= min_length)
    return train, val, test


def load_vocab(src_lang: str, tgt_lang: str, train_data=None, val_data=None, test_data=None):
    vocab_file = os.path.join(shared.VOCAB_FOLDER, f'vocab_{src_lang}-{tgt_lang}.pt')
    if not os.path.isfile(vocab_file):
        print("Building vocab...")
        if train_data is None or val_data is None or test_data is None:
            train_data, val_data, test_data = load_dataset(src_lang=src_lang, tgt_lang=tgt_lang)
        SRC.build_vocab(train_data.src, val_data.src, test_data.src, min_freq=1)
        TGT.build_vocab(train_data.trg, val_data.trg, test_data.trg, min_freq=1)
        torch.save((SRC.vocab, TGT.vocab), vocab_file)
        return SRC.vocab, TGT.vocab
    else:
        print("Loading vocab...")
        src_vocab, tgt_vocab = torch.load(vocab_file)
        SRC.vocab = src_vocab
        TGT.vocab = tgt_vocab
        return src_vocab, tgt_vocab


def check_exist_data(src_lang: str, tgt_lang: str):
    print("Checking dataset...")
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data', "wmt14")):
        print("Getting WMT14 dataset...")
        load_dataset(src_lang=src_lang, tgt_lang=tgt_lang)


def load_bpe(bpe_size: int = 32000):
    bpe_codes = codecs.open(os.path.join(shared.VOCAB_FOLDER, f"bpe.{bpe_size}"), encoding='utf-8')
    return BPE(bpe_codes)
