import os
import re

import torch
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer

import data as d
from shared import MODELS_FOLDER
from transformer.models import TransformerTranslator

DEVICE='cuda:0'
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

print('Loading model and Trainer...')
ckpt_path = os.path.join(MODELS_FOLDER, 'transformer', f'trafo_{src_lang}_{tgt_lang}_ensemble.pt')
translator = TransformerTranslator.load(src_lang=src_lang, tgt_lang=tgt_lang, device=DEVICE)

print('Starting test...')

i = 0
translations = []
references = []
with torch.no_grad():
	for src_text, tgt_text in zip(sources, targets):
		references.append(mtok.detokenize(tgt_text.split()))

		words = translator.translate(tok_bpe_sentence=src_text, beam_size=3, beam_length=0.6, max_length=350, device=DEVICE)
		translation = " ".join([d.TGT.vocab.itos[tok] for tok in words]).replace(d.BLANK_WORD, "").replace(d.BOS_WORD, "").\
			replace(d.EOS_WORD, "").replace("@@ ", "").replace(" ` ", "`").replace(" ' ", "'").replace(" ’ ", "").replace(" :", ":")
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
