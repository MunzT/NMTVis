import re

import torch
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer

import data as d
from seq2seq.models import Seq2SeqModel

BATCH_SIZE = 1
TEST_YEAR = 2016

src_lang = 'en'
tgt_lang = 'de'


def load_data(test_year: int = 2014):
	sources = []
	targets = []

	src_file = f'.data/wmt14/newstest{test_year}.tok.bpe.32000.{src_lang}'
	tgt_file = f'.data/wmt14/newstest{test_year}.{tgt_lang}'

	with open(src_file, 'r') as f:
		sources = f.readlines()
	with open(tgt_file, 'r') as f:
		targets = f.readlines()

	assert len(sources) == len(targets)
	print('Loaded', len(sources), 'sentences')
	return sources, targets


print("Loading vocab...")
src_vocab, tgt_vocab = d.load_vocab(src_lang, tgt_lang)
d.SRC.vocab = src_vocab
d.TGT.vocab = tgt_vocab
src_pad_key = d.SRC.vocab.stoi[d.BLANK_WORD]
tgt_pad_key = d.TGT.vocab.stoi[d.BLANK_WORD]

mtok = MosesDetokenizer(lang=tgt_lang)

print("Loading data...")
sources, targets = load_data(test_year=TEST_YEAR)

print('Loading model ...')
model = Seq2SeqModel.load(src_lang=src_lang, tgt_lang=tgt_lang, epoch=20)


print('Starting test...')

i = 0
translations = []
references = []
with torch.no_grad():
	for src_text, tgt_text in zip(sources, targets):
		references.append(mtok.detokenize(tgt_text.split()))

		words, _, _ = model.translate(sentence=src_text, beam_size=3, beam_length=0.6, max_length=350)
		translation = " ".join(words).replace(d.BLANK_WORD, "").replace(d.BOS_WORD, "").replace(d.EOS_WORD, "").\
			replace("@@ ", "").replace(" ` ", "`").replace(" ' ", "'").replace(" ’ ", "").replace(" :", ":")
		translation = re.sub(r"(\d):\s+(\d)", r"\1:\2", translation)
		translations.append(mtok.detokenize(translation.split()))

		i += 1
		if i % 50 == 0:
			print(f"---------------- TRANSLATION {i} ----------------")
			print(translations[-1])
			print(tgt_text)

print('')
print('')
print('')
print("########################################")
print(f"BLEU SCORE {src_lang}-{tgt_lang}", corpus_bleu(hypotheses=translations, references=[references]).score)
