import data as d
import os
import pytorch_lightning as pl
import shared
import time
import torch
import torch.nn.functional as F
from document import Hypothesis, Translation
from finetuningdataset import FinetuneDataset
from pytorch_lightning import LightningDataModule
from shared import TranslationModel, MODELS_FOLDER
from torchtext.data import BucketIterator, Batch
from transformer.transformer import TransformerWithAttn, TransformerModel


def src_sentence_to_input_tensor(src_sentence : str, device=torch.device('cpu')):
    # encode bpe tokens as tensor for transformer input
    bpe_src = src_sentence.split()
    src_tensor = torch.empty(len(bpe_src), dtype=torch.long, device=device)
    for token_idx, bpe_token in enumerate(bpe_src):
        src_tensor[token_idx] = d.SRC.vocab.stoi[bpe_token]
    src_tensor = src_tensor.unsqueeze(0)    # batch=1 x src_seq_len x src_vocab_size
    return src_tensor



class FinetuneDataModule(LightningDataModule):
    def __init__(self, batch_size, src_lang, tgt_lang, pairs) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.pairs = pairs


    def setup(self):
        print("Loading data...")
        src_vocab, tgt_vocab = d.load_vocab(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.src_pad_key = src_vocab.stoi[d.BLANK_WORD]
        self.tgt_pad_key = tgt_vocab.stoi[d.BLANK_WORD]
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_vocab_size = len(src_vocab)
        print('Building dataset...')
        self.train_set = FinetuneDataset(pairs=self.pairs, fields=(d.SRC, d.TGT))
        print("Done.")


    def train_dataloader(self):
        return BucketIterator(self.train_set, self.batch_size,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=True, shuffle=True, repeat=False)


    def transfer_batch_to_device(self, batch, device):
        if isinstance(batch, Batch):
            for field in batch.fields:
                batch_field = getattr(batch, field)
                batch_device = super().transfer_batch_to_device(batch_field, device)
                setattr(batch, field, batch_device)
        else:
            batch = super().transfer_batch_to_device(batch, device)
        return batch


class TransformerTranslator(TranslationModel):
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device


    def retrain(self, src_lang, tgt_lang, pairs, last_ckpt, epochs=10, batch_size=32, device='cpu', save_ckpt=True, print_info=True):
        self.model.train()
        self.model.freeze_weights()
        self.model.trafo.encoder.eval()
        for param in self.model.trafo.encoder.parameters():
            param.requires_grad = False

        gpus = None if device == 'cpu' else torch.cuda.device_count()
        dm = FinetuneDataModule(batch_size=batch_size, src_lang=src_lang, tgt_lang=tgt_lang, pairs=pairs)
        dm.prepare_data()
        dm.setup()
        if not self.model.trainer:
            self.model.trainer = pl.Trainer(gpus=gpus, checkpoint_callback=False, logger=False, resume_from_checkpoint=last_ckpt, min_epochs=epochs, max_epochs=epochs, weights_summary=None, progress_bar_refresh_rate=1 if print_info else 0)
        print('Training for', epochs,  'epochs')
        self.model.trainer.fit(self.model, dm)
        # print('done!')
        self.model.eval()
        self.model.trainer.current_epoch = 0 # reset epoch for continued training
        if save_ckpt:
            filename = os.path.join(shared.TRAFO_CHECKPOINT_PATH, f"trafo_{src_lang}_{tgt_lang}_{time.strftime('%Y%m%d-%H%M%S')}_finetuned.pt")
            print('saving to...', filename)
            self.model.trainer.save_checkpoint(filename)
            return filename


    @classmethod
    def load(cls, src_lang="de", tgt_lang='en', device=torch.device('cpu')):
        D_MODEL = 512
        D_FF = 2048
        HEADS = 8
        P_DROP = 0.1
        src_pad_key = d.SRC.vocab.stoi[d.BLANK_WORD]
        tgt_pad_key = d.TGT.vocab.stoi[d.BLANK_WORD]
        model = TransformerModel.load_from_checkpoint(os.path.join(MODELS_FOLDER, 'transformer', f'trafo_{src_lang}_{tgt_lang}_ensemble.pt'),
                                                src_vocab_size=len(d.SRC.vocab), tgt_vocab_size=len(d.TGT.vocab),
                                                src_pad_key=src_pad_key, tgt_pad_key=tgt_pad_key, max_len=d.MAX_LEN,
                                                d_model=D_MODEL, nhead=HEADS, num_encoder_layers=6,
                                                num_decoder_layers=6, dim_feedforward=D_FF, dropout=P_DROP,
                                                activation="relu")

        model.eval()
        return cls(model, device)


    @torch.no_grad()
    def best_translation(self, hyps):
        translation = [d.TGT.vocab.itos[token] for token in hyps[0].tokens] if hyps else [d.BOS_WORD]
        attn = hyps[0].attns if hyps else [[0]]

        return translation[1:], attn[1:]


    @torch.no_grad()
    def translate(self, sentence, beam_size=3, beam_length=0.6, beam_coverage=0.4, attention_override_map=None, correction_map=None, unk_map=None, max_length=d.MAX_LEN):
        self.model.eval()
        hyps = self.beam_search(sentence, beam_size, attention_override_map, correction_map, unk_map,
                                beam_length=beam_length, beam_coverage=beam_coverage, max_length=max_length)
        words, attention = self.best_translation(hyps)

        return words, attention, [Translation.from_hypothesis(h) for h in hyps]


    @torch.no_grad()
    def beam_search(self, input_seq, beam_size=3, attentionOverrideMap=None, correctionMap=None, unk_map=None,
                    beam_length=0.6, beam_coverage=0.4,
                    max_length=d.MAX_LEN):

        self.model.eval()

        src_tensor = src_sentence_to_input_tensor(input_seq, device='cpu')
        enc_output, src_key_padding_mask = self.model.trafo.encode(src_tensor)   # 1 X src_seq_len(BPE)

        beam_search = BeamSearch(self.model.trafo, src_tensor, enc_output, src_key_padding_mask, beam_size, attentionOverrideMap,
                                 correctionMap, unk_map, beam_length=beam_length, beam_coverage=beam_coverage,
                                 max_length=max_length)
        result = beam_search.search()

        return result  # Return a list of indexes, one for each word in the sentence, plus EOS


    @torch.no_grad()
    def greedy_decode(self, input_sentence, max_len=10):
        self.model.eval()
        tokens_so_far = [d.TGT.vocab.stoi[d.BOS_WORD]]
        step = 0

        bpe_src = d.bpe.process_line(input_sentence).split()
        src_tensor = torch.empty(len(bpe_src), dtype=torch.long)
        for token_idx, bpe_token in enumerate(bpe_src):
            src_tensor[token_idx] = d.SRC.vocab.stoi[bpe_token]
        src_tensor = src_tensor.unsqueeze(0)    # batch=1 x src_seq_len
        enc = self.model.trafo.encode(src_tensor)
        
        while tokens_so_far[-1] != d.TGT.vocab.stoi[d.EOS_WORD] and step < max_len:
            output = self.model.trafo.decode_step(src_tensor, enc, torch.tensor(tokens_so_far, dtype=torch.long).view(1,-1))[0]
            new_tokens = output.argmax(dim=-1).view(1, -1)
            step += 1
            tokens_so_far.append(new_tokens.view(-1)[-1].item())

        return tokens_so_far


class BeamSearch:
    def __init__(self, transformer: TransformerWithAttn, src_tensor, encoder_outputs, src_key_padding_mask,
                 beam_size=3, attentionOverrideMap=None, correctionMap=None, unk_map=None, beam_length=0.6,
                 beam_coverage=0.4, max_length=d.MAX_LEN):
        self.model = transformer
        self._src_enc = encoder_outputs
        self.src_key_padding_mask = src_key_padding_mask
        self._src_tensor = src_tensor
        self.beam_size = beam_size
        self.max_length = max_length
        self.attention_override_map = attentionOverrideMap
        self.correction_map = correctionMap
        self.unk_map = unk_map
        self.beam_length = beam_length
        self.beam_coverage = beam_coverage
        self.prefix = self.compute_prefix()
        self.process_corrections()


    def compute_prefix(self):
        if not self.correction_map:
            return
        print(self.correction_map)
        assert len(list(self.correction_map.keys())) == 1

        prefix = list(self.correction_map.keys())[0]
        correction = self.correction_map[prefix]

        prefix = prefix + " " + correction
        raw_words = prefix.split(" ")[1:]
        bpe_words = [d.bpe.process_line(word) if not word.endswith("@@") else word for word in raw_words]
        words = [word for bpe_word in bpe_words for word in bpe_word.split(" ")]
        return words


    def process_corrections(self):
        """Apply BPE to correction map, ignoring words that are already BPE'd"""
        if not self.correction_map:
            return
        prefixes = list(self.correction_map.keys())

        for prefix in prefixes:
            raw_words = self.correction_map.pop(prefix).split(" ")
            bpe_words = [d.bpe.process_line(word) if not word.endswith("@@") else word for word in raw_words]
            words = [word for bpe_word in bpe_words for word in bpe_word.split(" ")]

            for i in range(len(words)):
                curr_prefix = " ".join(prefix.split(" ") + words[:i])
                self.correction_map[curr_prefix] = words[i]


    def to_partial(self, tokens):
        return " ".join([d.TGT.vocab.itos[token] for token in tokens])


    @torch.no_grad()
    def decode_topk(self, tokens_sofar, partials):
        """Decode all current hypotheses on the beam, returning len(hypotheses) x beam_size candidates"""

        # len(latest_tokens) x self.beam_size)
        attns = [None for _ in range(len(tokens_sofar))]
        topk_words = [["" for _ in range(self.beam_size)] for _ in range(len(tokens_sofar))]
        is_unk = [False for _ in range(len(tokens_sofar))]
        topk_ids = [[0 for _ in range(self.beam_size)] for _ in range(len(tokens_sofar))]
        topk_log_probs = [[0 for _ in range(self.beam_size)] for _ in range(len(tokens_sofar))]

        # decode next step
        decoder_input = torch.as_tensor(tokens_sofar, dtype=torch.long)
        new_tgt_probs, dec_attn = self.model.decode_step(self.src_key_padding_mask.repeat(len(tokens_sofar), 1), self._src_enc.repeat(1, len(tokens_sofar), 1), tgt_so_far=decoder_input) # new_tgt_probs: 1 x tgt_seq_len + 1 x tgt_vocab_size
        new_tgt_probs = new_tgt_probs[:,-1,:].squeeze(dim=1)
        # mean attention
        # dec_attn = torch.cat([layer_att.unsqueeze(dim=0) for layer_att in dec_attn], dim=0) # layers x batch x tgt seq len x src seq len
        # dec_attn = torch.mean(dec_attn, dim=0) # batch x tgt seq len x src seq len
        # second last layer attention: created best alignments in our experiments
        dec_attn = dec_attn[-2]  # batch x tgt seq len x src seq len

        # Loop over all hypotheses
        for i, tokens in enumerate(tokens_sofar):
            if self.correction_map and partials[i] in self.correction_map:
                word = self.correction_map[partials[i]]
                print("Corrected {} for partial= {}".format(word, partials[i]))
                if not word in d.TGT.vocab.stoi:
                    topk_words[i][0] = word
                    is_unk[i] = True
                idx = d.TGT.vocab.stoi[word]
                new_tgt_probs[i, idx] = 1000

            new_tgt_probs_i = F.log_softmax(new_tgt_probs[i], dim=-1)   # tgt_vocab_size
            topk_v, topk_i = new_tgt_probs_i.topk(self.beam_size)
            topk_v, topk_i = topk_v.numpy(), topk_i.numpy()

            topk_ids[i] = topk_i.tolist()
            topk_log_probs[i] = topk_v.tolist()
            topk_words[i] = [d.TGT.vocab.itos[id] if not topk_words[i][j] else topk_words[i][j] for j, id in enumerate(topk_ids[i])]

            attns[i] = dec_attn[i, -1, :].tolist() # tgt seq len x src seq len -> src_seq_len

        return topk_ids, topk_words, topk_log_probs, attns, is_unk

    def init_hypothesis(self):
        start_attn = [[0.0]]
        BOS_token = d.TGT.vocab.stoi[d.BOS_WORD]

        if not self.correction_map:
            return [Hypothesis([BOS_token], [d.BOS_WORD], [0.0], # token, word, log_prob
                               None, # decoder state
                               None, # last attn vector
                               start_attn, [[]], [False]) for _ in range(self.beam_size)]  # attns=None, candidates=None, is_unk=None)

        # Assume at most 1 correction prefix at all times
        prefix = [d.TGT.vocab.stoi[token] for token in self.prefix]

        tokens = [d.TGT.vocab.stoi[d.BOS_WORD]]
        candidates = [[]]

        start_attn = [[0 for _ in range(self._src_tensor.size(-1))]]

        #decode
        decoder_input = torch.as_tensor(tokens + prefix, dtype=torch.long).view(1, -1)
        tgt_probs, dec_attn = self.model.decode_step(self.src_key_padding_mask, self._src_enc, tgt_so_far=decoder_input)
        tgt_probs = F.log_softmax(tgt_probs[0], dim=-1) # batch x seq length x vocab -> seq length x voacb

        tokens = tokens + prefix
        topk_v, topk_i = tgt_probs.topk(self.beam_size, dim=-1) # seq length x beam size
        attn = dec_attn[-1][0, :, :] # tgt x src

        for i in range(len(tokens)):
            start_attn.append(attn[i].tolist())
            candidates.append([d.TGT.vocab.itos[i] for i in topk_i[i].tolist()])

        return [Hypothesis(list(tokens), [d.TGT.vocab.itos[token] for token in tokens], [0.0] * len(tokens),
                           None,
                           None,
                           list(start_attn), candidates, [False] * len(tokens)) for _ in range(self.beam_size)]


    @torch.no_grad()
    def search(self):
        hyps = self.init_hypothesis()

        for h in hyps:
            h.alpha = self.beam_length
            h.beta = self.beam_coverage

        result = []

        steps = 0
        while steps < self.max_length * 2 and len(result) < self.beam_size:
            tokens_sofar = [hyp.tokens for hyp in hyps]
            partials = [self.to_partial(hyp.tokens) for hyp in hyps]
            all_hyps = []

            num_beam_source = 1 if steps == 0 else len(hyps)
            topk_ids, topk_words, topk_log_probs, attns, is_unk = self.decode_topk(tokens_sofar, partials)

            for i in range(num_beam_source):
                h, attn = hyps[i], attns[i]
                # print("  ", h.words, ": attn", attn)

                for j in range(self.beam_size):
                    candidates = [d.TGT.vocab.itos[c] for c in (topk_ids[i][:j] + topk_ids[i][j + 1:])]
                    all_hyps.append(
                        h.extend(token=topk_ids[i][j], word=topk_words[i][j], new_log_prob=topk_log_probs[i][j],
                                 new_state=None, last_attn_vector=None, attn=[attn], candidates=candidates,
                                 is_unk=is_unk[i]))

            # Filter
            hyps = []
            for h in self._best_hyps(all_hyps):
                if h.words[-1] == d.EOS_WORD:
                    result.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(result) == self.beam_size:
                    break
            steps += 1

        # print("Beam Search found {} hypotheses for beam_size {}".format(len(result), self.beam_size))
        res = self._best_hyps(result, normalize=True)
        if res:
            res[0].is_golden = True
        return res


    def _best_hyps(self, hyps, normalize=False):
        """Sort the hyps based on log probs and length.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        if normalize:
            return sorted(hyps, key=lambda h: h.score(), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
