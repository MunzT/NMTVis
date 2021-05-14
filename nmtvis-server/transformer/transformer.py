import data as d
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from pytorch_lightning.metrics.functional import bleu_score
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext.data import BucketIterator, Batch
from transformer.optimizer import Optimizer
from typing import Optional



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].detach().unsqueeze(dim=1)
        return self.dropout(x)


class TransformerDecoderLayerWithAttn(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights


class TransformerDecoderWithAttn(nn.TransformerDecoder):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        attns = []

        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


class TransformerWithAttn(nn.Transformer):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_pad_key: int, tgt_pad_key: int, max_len: int,
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu"):
            super().__init__(d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout,
                 activation, custom_decoder=TransformerDecoderWithAttn(TransformerDecoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout, activation), num_decoder_layers))

            self.src_pad_key = src_pad_key
            self.tgt_pad_key = tgt_pad_key
            self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=max_len+3)
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
            self.classifier = nn.Linear(d_model, tgt_vocab_size)
            self.tgt_vocab_size = tgt_vocab_size
            self.d_model = d_model

            self._init_weights()
    
    @torch.no_grad()
    def _init_weights(self):
        for name, p in self.named_parameters():
            # if "embed" in name:
            #     nn.init.normal_(p, mean=0., std=0.01)
            if "bias" in name:
                nn.init.zeros_(p)
            elif "norm" in name:
                continue
            else:
                nn.init.xavier_uniform_(p)

        # zero padding weights in embedding
        self.src_embedding.weight.data[self.src_pad_key].zero_()
        self.tgt_embedding.weight.data[self.tgt_pad_key].zero_()
    
    def encode(self, src: Tensor):
        src_key_padding_mask = src.eq(self.src_pad_key) # batch x seq length
        src_emb = src.transpose(1,0)  # seq length x batch

        src_emb = self.src_embedding(src_emb) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)

        return self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask), src_key_padding_mask

    def decode_step(self, src_key_padding_mask, encoder_memory: Tensor, tgt_so_far: Tensor):
        tgt_key_padding_mask = tgt_so_far.eq(self.tgt_pad_key) # batch x seq length

        tgt = tgt_so_far.transpose(1,0)  # seq length x batch

        length = tgt.size(dim=0)  # seq length
        tgt_mask = self.generate_square_subsequent_mask(length).to(src_key_padding_mask.device) # length x lengths

        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)

        output, attns = self.decoder(tgt, encoder_memory, tgt_mask=tgt_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=src_key_padding_mask)
        output = self.layer_norm(output)
        output = self.classifier(output)
        return output.transpose(1,0), attns

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        input: batch x src length 
        output: batch x tgt length
        """
        tgt_key_padding_mask = tgt.eq(self.tgt_pad_key) # batch x seq length

        tgt_emb = tgt.transpose(1,0)  # seq length x batch
        length = tgt_emb.size(dim=0)  # seq length
        tgt_mask = self.generate_square_subsequent_mask(length).to(src.device) # length x length

        memory, src_key_padding_mask = self.encode(src) # seq_len x batch x d_model
 
        tgt_emb = self.tgt_embedding(tgt_emb) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        output, attns = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)
        output = self.layer_norm(output)
        output = self.classifier(output) # seq length x batch x vocab
        return output.transpose(1,0), attns  # seq length x batch x vocab -> batch x seq length x voacb

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int):
        batch_size = src.size(dim=0)
        eos_id = d.TGT.vocab.stoi[d.EOS_WORD]
        memory, src_key_padding_mask = self.encode(src)

        tgt_so_far = torch.full((batch_size, 1), fill_value=d.TGT.vocab.stoi[d.BOS_WORD], device=src.device, dtype=torch.long)
        text_so_far = [[]] * batch_size

        # a subsequent mask is intersected with this in decoder forward pass
        finished = 0
        for _ in range(max_len):
            with torch.no_grad():
                logits, _ = self.decode_step(src_key_padding_mask, memory, tgt_so_far)
                logits = F.log_softmax(logits[:, -1, :], dim=-1)
                _, logits = torch.max(logits, dim=-1)
                for batch_idx, token_id in enumerate(logits.tolist()):
                    if token_id == eos_id:
                        finished += 1
                    else:
                        text_so_far[batch_idx].append(d.TGT.vocab.itos[token_id])
                tgt_so_far = torch.cat((tgt_so_far, logits.view(-1, 1)), dim=1)
            if finished == batch_size:
                break

        return text_so_far



global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    # code from http://nlp.seas.harvard.edu/2018/04/03/attention.html 
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class TranslationDataModule(LightningDataModule):
    def __init__(self, batch_size, src_lang, tgt_lang) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang


    def setup(self):
        print("Loading data...")
        self.train_set, self.val_set, self.test_set = d.load_dataset(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        src_vocab, tgt_vocab = d.load_vocab(src_lang=self.src_lang, tgt_lang=self.tgt_lang, train_data=self.train_set, val_data=self.val_set, test_data=self.test_set)
        self.src_pad_key = src_vocab.stoi[d.BLANK_WORD]
        self.tgt_pad_key = tgt_vocab.stoi[d.BLANK_WORD]
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_vocab_size = len(src_vocab)
        print("Done.")


    def train_dataloader(self):
        return BucketIterator(self.train_set, self.batch_size, batch_size_fn=batch_size_fn, sort_within_batch=True,
                              train=True, shuffle=True, repeat=False, sort=False)

    def val_dataloader(self):
        return BucketIterator(self.val_set, 1, sort_within_batch=True, sort=False,
                              train=False, shuffle=False)


    def transfer_batch_to_device(self, batch, device):
        if isinstance(batch, Batch):
            for field in batch.fields:
                batch_field = getattr(batch, field)
                batch_device = super().transfer_batch_to_device(batch_field, device)
                setattr(batch, field, batch_device)
        else:
            batch = super().transfer_batch_to_device(batch, device)
        return batch



class TransformerModel(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_key, tgt_pad_key, max_len: int,
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", smoothing: float = 0.1) -> None:
        super().__init__()

        # setup model and loss
        self.trafo = TransformerWithAttn(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                         src_pad_key=src_pad_key, tgt_pad_key=tgt_pad_key,
                                         max_len=max_len, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                         num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                         dropout=dropout, activation=activation)
        self.loss_fn = LabelSmoothingLoss(tgt_pad_key, smoothing)
        self.save_hyperparameters()


    def freeze_weights(self):
        for param in self.trafo.encoder.parameters():
            param.requires_grad = False


    def forward(self, src, tgt):
        # batch x seq_len
        return self.trafo(src, tgt)


    def configure_optimizers(self):
        return Optimizer(self.parameters(), d_model=self.trafo.d_model, factor=1.0, warmup_steps=8000)


    def training_step(self, batch, batch_idx):
        src, tgt = batch.src, batch.trg # batch x seq_len
        pred, _ = self(src, tgt[:,:-1])
        loss = self.loss_fn(pred, tgt[:,1:])

        with torch.no_grad():
            mask = tgt[:,1:].ne(self.trafo.tgt_pad_key)
            ntokens = mask.view(-1).sum(dim=0).item()
            self.log('tgt_tokens', mask.view(-1).sum(dim=0), reduce_fx=torch.sum)
            acc = (pred.argmax(dim=-1).masked_select(mask) == tgt[:,1:].masked_select(mask)).float().mean()
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('src_tokens', src.ne(self.trafo.src_pad_key).view(-1).sum(dim=0), on_step=True, on_epoch=False, prog_bar=False, logger=True, reduce_fx=torch.sum)
        loss = loss / ntokens
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        src, tgt = batch.src, batch.trg
        beam_text = self.trafo.greedy_decode(src, max_len=d.MAX_LEN)
        tgt_text = []
        for batch_index in range(tgt.size(dim=0)):
            sentence = []
            for token in tgt[batch_index].tolist():
                if token == self.trafo.tgt_pad_key:
                    break
                else:
                    sentence.append(d.TGT.vocab.itos[token])
            tgt_text.append(sentence)

        return {'pred_text': beam_text, 'tgt_text': tgt_text}


    def validation_step_end(self, batch_parts):
        pred_corpus = []
        tgt_corpus = []
        for pred_sentence, tgt_sentence in zip(batch_parts['pred_text'], batch_parts['tgt_text']):
            pred_corpus.append(' '.join(pred_sentence).replace('@@ ', '').split())
            tgt_corpus.append([' '.join(tgt_sentence).replace('@@ ', '').split()])

        return {"pred_corpus": pred_corpus, "tgt_corpus": tgt_corpus}


    def validation_epoch_end(self, validation_step_outputs):
        pred_corpus = []
        tgt_corpus = []
        for out in validation_step_outputs:
            pred_corpus += out['pred_corpus']
            tgt_corpus += out['tgt_corpus']
        score = bleu_score(pred_corpus, tgt_corpus)
        self.log('val_bleu', score)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, tgt_pad_index, smoothing: float):
        super(LabelSmoothingLoss, self).__init__()

        assert 0.0 < smoothing <= 1.0
        self.smoothing = smoothing
        self.tgt_pad_index = tgt_pad_index
        self.criterion = nn.KLDivLoss(reduction="sum")


    def forward(self, preds, labels):
        """
        preds (FloatTensor): (batch_size, seq_len, vocabulary_size)
        labels (LongTensor): (batch_size, seq_len)
        """

        batch_size, num_tokens, vocab_size = preds.size()

        # build smoothed label distribution (batch * seq_len x)
        with torch.no_grad():
            non_label_prob = self.smoothing/(vocab_size - 2)
            label_dist = torch.full((batch_size*num_tokens, vocab_size), fill_value=non_label_prob, dtype=torch.float, device=preds.device)
            # distribute real labels
            label_prob = 1.0 - self.smoothing
            label_dist.scatter_(1, labels.reshape(-1).unsqueeze(dim=1), label_prob)
            # mask padding
            padding_positions = torch.nonzero(labels.reshape(-1) == self.tgt_pad_index, as_tuple=False).squeeze()
            label_dist.index_fill_(0, padding_positions, 0.0)

        return self.criterion(F.log_softmax(preds, dim=-1).view(-1, vocab_size), label_dist)
