import torch
from torch.utils.tensorboard import SummaryWriter

import data as d
import seq2seq.hp as hp
from seq2seq.models import Seq2SeqModel, LSTMEncoderRNN, LSTMAttnDecoderRNN
from seq2seq.train import train_iters

SRC_LANG = "de"
TGT_LANG = "en"
device = torch.device('cuda:0')
writer = SummaryWriter(log_dir=f"logs/seq2seq_logs_{SRC_LANG}_{TGT_LANG}")

d.MAX_LEN = 40
print("Loading dataset...")
train, val, test = d.load_dataset(src_lang=SRC_LANG, tgt_lang=TGT_LANG, min_length=hp.MIN_LENGTH)
print("Loading vocabulary...")
src_vocab, tgt_vocab = d.load_vocab(src_lang=SRC_LANG, tgt_lang=TGT_LANG, train_data=train, val_data=val, test_data=test)

print("Training seq2seq model...")
encoder = LSTMEncoderRNN(len(src_vocab), hp.hidden_size, hp.embed_size)
decoder = LSTMAttnDecoderRNN(encoder, hp.attention, hp.hidden_size, len(tgt_vocab))
encoder = encoder.to(device)
decoder = decoder.to(device)
seq2seq_model = Seq2SeqModel(encoder, decoder)

train_iters(seq2seq_model, train, val, SRC_LANG, TGT_LANG,
                n_epochs=hp.n_epochs,
                print_every=hp.print_loss_every_iters,
                evaluate_every=hp.eval_bleu_every_epochs,
                save_every=hp.save_every_epochs,
                learning_rate=hp.learning_rate,
                decoder_learning_ratio=hp.decoder_learning_ratio,
                batch_size=hp.batch_size,
                encoder_optimizer_state=None,
                decoder_optimizer_state=None, train_loss=[], eval_loss=[],
                bleu_scores=[],
                start_epoch=1,
                retrain=False,
                weight_decay=1e-5,
                device=device,
                writer=writer)
