import data as d
import os
import shared
import time
import torch
import torch.nn.functional as F
from document import Hypothesis, Translation
from finetuningdataset import FinetuneDataset
from pytorch_lightning import LightningDataModule
from shared import TranslationModel, MODELS_FOLDER
from torchtext.data import BucketIterator, Batch
from tqdm.auto import tqdm
from transformer.transformer import LabelSmoothingLoss, TransformerWithAttn, TransformerModel, TranslationDataModule
from transformer.optimizer import Optimizer
from pytorch_lightning.utilities.cloud_io import load as pl_load


def src_sentence_to_input_tensor(src_sentence : str, device=torch.device('cpu')):
    # encode bpe tokens as tensor for transformer input
    bpe_src = src_sentence.split()
    src_tensor = torch.empty(len(bpe_src), dtype=torch.long)
    for token_idx, bpe_token in enumerate(bpe_src):
        src_tensor[token_idx] = d.SRC.vocab.stoi[bpe_token]
    src_tensor = src_tensor.unsqueeze(0)    # batch=1 x src_seq_len x src_vocab_size
    return src_tensor.to(device)


def _mix_batches(batch_1: torch.LongTensor, batch_2: torch.LongTensor, pad_token_id: int, device: str):
	if batch_1.size(1) > batch_2.size(1):
		# pad 2nd batch
		pad_1 = batch_1.to(device)
		pad_2 = torch.cat((batch_2.to(device), torch.full((batch_2.size(0), batch_1.size(1) - batch_2.size(1)), fill_value=pad_token_id, device=device)), dim=1)
	elif batch_1.size(1) < batch_2.size(1):
		# pad 1st batch
		pad_1 = torch.cat((batch_1.to(device), torch.full((batch_1.size(0), batch_2.size(1) - batch_1.size(1)), fill_value=pad_token_id, device=device)), dim=1)
		pad_2 = batch_2.to(device)
	else:
		pad_1 = batch_1.to(device)
		pad_2 = batch_2.to(device)
	return torch.cat((pad_1, pad_2), dim=0)


class FinetuneDataModule(LightningDataModule):
    def __init__(self, batch_size, src_lang, tgt_lang, pairs, test_pairs=None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.pairs = pairs
        self.test_pairs = test_pairs


    def setup(self):
        print("Loading data...")
        src_vocab, tgt_vocab = d.load_vocab(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        d.SRC.vocab = src_vocab
        d.TGT.vocab = tgt_vocab
        self.src_pad_key = src_vocab.stoi[d.BLANK_WORD]
        self.tgt_pad_key = tgt_vocab.stoi[d.BLANK_WORD]
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_vocab_size = len(src_vocab)
        print('Building dataset...')
        self.train_set = FinetuneDataset(pairs=self.pairs, fields=(d.SRC, d.TGT))
        if self.test_pairs:
            self.test_set = FinetuneDataset(pairs=self.test_pairs, fields=(d.SRC, d.TGT))
        print("Done.")


    def train_dataloader(self):
        return BucketIterator(self.train_set, self.batch_size,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=True, shuffle=True, repeat=False)

    def test_dataloader(self):
        return BucketIterator(self.test_set, self.batch_size,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=False, shuffle=False, repeat=False)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, Batch):
            for field in batch.fields:
                batch_field = getattr(batch, field)
                batch_device = super().transfer_batch_to_device(batch_field, device, dataloader_idx)
                setattr(batch, field, batch_device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch


class TransformerTranslator(TranslationModel):
    def __init__(self, model: TransformerModel, optim: Optimizer, src_lang: str, tgt_lang: str, ckpt: dict, device):
        self.model = model
        self.device = device
        self.ckpt = ckpt
        self.optim = optim
        self._mixin_dm = None
        self._val_dm = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang


    def retrain(self, src_lang, tgt_lang, pairs, last_ckpt, epochs=10, batch_size=32, device='cpu', save_ckpt=True,
                constant_lr: bool = False, accum_grad_steps: int = 1, freeze_encoder: bool = True,
                orig_data_mixin_percentage: float = 0.0, label_smoothing: float = 0.1, exp_name: str = ''):
        
        print("retrain")
        # setup data
        dm = FinetuneDataModule(batch_size=batch_size, src_lang=src_lang, tgt_lang=tgt_lang, pairs=pairs)
        dm.prepare_data()
        dm.setup()

        # setup model and optimizer
            
      
        self.model.train()
        if freeze_encoder:
            self.model.freeze_weights()
            self.model.trafo.encoder.eval()
        loss_fn = LabelSmoothingLoss(d.TGT.vocab.stoi[d.BLANK_WORD], smoothing=label_smoothing)

        # setup data mixin
        batch_size = int(batch_size * (1-orig_data_mixin_percentage))
        mixin_batch_size = int(batch_size * orig_data_mixin_percentage)
        if mixin_batch_size > 0 and not self._mixin_dm:
            self._mixin_dm = TranslationDataModule(batch_size=batch_size, src_lang=src_lang, tgt_lang=tgt_lang, only_val=False)
            self._mixin_dm.setup()

        # train
        print('Training for', epochs,  'epochs')
        train_losses = []
        # test_losses = []
        for epoch in tqdm(range(epochs)):
            # setup dataloaders
            train_loader = dm.train_dataloader()
            if mixin_batch_size > 0:
                mixin_loader = iter(self._mixin_dm.train_dataloader())

            self.optim.zero_grad()
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                # prepare batch data
                if mixin_batch_size > 0:
                    mixin_batch = next(mixin_loader)
                    src = _mix_batches(batch.src, mixin_batch.src, self.model.trafo.src_pad_key, device)
                    tgt = _mix_batches(batch.trg, mixin_batch.trg, self.model.trafo.tgt_pad_key, device)
                else:
                    src = batch.src.to(device)
                    tgt = batch.trg.to(device)

                # forward and backward passes
                pred, _ = self.model(src, tgt[:,:-1])
                loss = loss_fn(pred, tgt[:,1:])
                loss.backward()
                epoch_losses.append(loss.item())

                # optimizer step
                if accum_grad_steps == 1 or (batch_idx != 0 and batch_idx % accum_grad_steps == 0):
                    if constant_lr:
                        self.optim.constant_step()
                    else:
                        self.optim.step()
                    self.optim.zero_grad()
                elif batch_idx == len(train_loader) - 1:
                    if constant_lr:
                        self.optim.constant_step()
                    else:
                        self.optim.step()
                    self.optim.zero_grad()
               
                train_losses.append(sum(epoch_losses)/len(epoch_losses))

        self.model.eval()
        if save_ckpt:
            filename = os.path.join(shared.TRAFO_CHECKPOINT_PATH, f"{exp_name}_{src_lang}_{tgt_lang}_{time.strftime('%Y%m%d-%H%M%S')}_finetuned.pt")
            print('saving to...', filename)
            self.ckpt['state_dict'] = self.model.state_dict()
            self.ckpt['optimizer_states'][0] = self.optim.state_dict()
            if not 'train_losses' in self.ckpt:
                self.ckpt['train_losses'] = []
            # if not 'test_losses' in ckpt:
            #     ckpt['test_losses'] = []
            self.ckpt['train_losses'].append(train_losses)
            # ckpt['test_losses'].append(test_losses)
            self.ckpt['epochs'] = epochs
            self.ckpt['batch_size'] = batch_size
            self.ckpt['constant_lr'] = constant_lr
            self.ckpt['accum_grad_steps'] = accum_grad_steps
            self.ckpt['freeze_encoder'] = freeze_encoder
            self.ckpt['orig_data_mixin_percentage'] = orig_data_mixin_percentage
            self.ckpt['label_smoothing'] = label_smoothing
            torch.save(self.ckpt, filename)

            return filename


    @classmethod
    def load(cls, src_lang="de", tgt_lang='en', device=torch.device('cpu')):
        print("LOADING")
        ckpt = pl_load(f'.data/models/transformer/trafo_{src_lang}_{tgt_lang}_ensemble.pt', map_location=lambda storage, loc: storage)
        model = TransformerModel.load_from_checkpoint(os.path.join(MODELS_FOLDER, 'transformer', f'trafo_{src_lang}_{tgt_lang}_ensemble.pt')).eval().to(device)
        optim_state = ckpt['optimizer_states'][0]
        optim = Optimizer(model.parameters())
        optim.load_state_dict(optim_state)
        return cls(model, optim, src_lang, tgt_lang, ckpt, device)


    @torch.no_grad()
    def best_translation(self, hyps):
        translation = [d.TGT.vocab.itos[token] for token in hyps[0].tokens] if hyps else [d.BOS_WORD]
        attn = hyps[0].attns if hyps else [[0]]

        return translation[1:], attn[1:]


    @torch.no_grad()
    def translate(self, sentence, beam_size=3, beam_length=0.6, beam_coverage=0.4, attention_override_map=None, correction_map=None, unk_map=None, max_length=d.MAX_LEN, attLayer=-2):
        self.model.eval()
        hyps = self.beam_search(sentence, beam_size, attLayer, attention_override_map, correction_map, unk_map,
                                beam_length=beam_length, beam_coverage=beam_coverage, max_length=max_length)
        words, attention = self.best_translation(hyps)

        return words, attention, [Translation.from_hypothesis(h) for h in hyps]


    @torch.no_grad()
    def beam_search(self, input_seq, beam_size=3, attLayer=-2, attentionOverrideMap=None, correctionMap=None, unk_map=None,
                    beam_length=0.6, beam_coverage=0.4,
                    max_length=d.MAX_LEN):

        self.model.eval()

        src_tensor = src_sentence_to_input_tensor(input_seq, device=self.device)
        enc_output, src_key_padding_mask = self.model.trafo.encode(src_tensor)   # 1 X src_seq_len(BPE)

        beam_search = BeamSearch(self.model.trafo, src_tensor, enc_output, src_key_padding_mask, beam_size, attLayer, attentionOverrideMap,
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
        src_tensor = src_tensor.unsqueeze(0).to(self.device)    # batch=1 x src_seq_len
        enc = self.model.trafo.encode(src_tensor)
        
        while tokens_so_far[-1] != d.TGT.vocab.stoi[d.EOS_WORD] and step < max_len:
            output = self.model.trafo.decode_step(src_tensor, enc, torch.tensor(tokens_so_far, dtype=torch.long, device=self.device).view(1,-1))[0]
            new_tokens = output.argmax(dim=-1).view(1, -1)
            step += 1
            tokens_so_far.append(new_tokens.view(-1)[-1].item())

        return tokens_so_far


class BeamSearch:
    def __init__(self, transformer: TransformerWithAttn, src_tensor, encoder_outputs, src_key_padding_mask,
                 beam_size=3, attLayer=-2, attentionOverrideMap=None, correctionMap=None, unk_map=None, beam_length=0.6,
                 beam_coverage=0.4, max_length=d.MAX_LEN):
        self.model = transformer
        self.device = next(self.model.parameters()).device
        self._src_enc = encoder_outputs
        self.src_key_padding_mask = src_key_padding_mask
        self._src_tensor = src_tensor
        self.beam_size = beam_size
        self.attLayer = attLayer
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
        decoder_input = torch.as_tensor(tokens_sofar, dtype=torch.long, device=self.device)
        new_tgt_probs, dec_attn = self.model.decode_step(self.src_key_padding_mask.repeat(len(tokens_sofar), 1),
                                                         self._src_enc.repeat(1, len(tokens_sofar), 1),
                                                         tgt_so_far=decoder_input) # new_tgt_probs: 1 x tgt_seq_len + 1 x tgt_vocab_size
        new_tgt_probs = new_tgt_probs[:,-1,:].squeeze(dim=1)

        # second last layer attention: created best alignments in our experiments
        #temp_dec_attn = dec_attn[-2]  # batch x tgt seq len x src seq len

        if self.attLayer is not None:
            temp_dec_attn = dec_attn[self.attLayer]  # batch x tgt seq len x src seq len
        else:
            # mean attention
            temp_dec_attn = torch.cat([layer_att.unsqueeze(dim=0) for layer_att in dec_attn], dim=0) # layers x batch x tgt seq len x src seq len
            temp_dec_attn = torch.mean(temp_dec_attn, dim=0) # batch x tgt seq len x src seq len

        # Loop over all hypotheses
        for i, tokens in enumerate(tokens_sofar):
            if self.correction_map and partials[i] in self.correction_map:
                word = self.correction_map[partials[i]]
                if not word in d.TGT.vocab.stoi:
                    topk_words[i][0] = word
                    is_unk[i] = True
                idx = d.TGT.vocab.stoi[word]
                new_tgt_probs[i, idx] = 1000

            new_tgt_probs_i = F.log_softmax(new_tgt_probs[i], dim=-1)   # tgt_vocab_size
            topk_v, topk_i = new_tgt_probs_i.topk(self.beam_size)
            topk_v, topk_i = topk_v.cpu().numpy(), topk_i.cpu().numpy()

            topk_ids[i] = topk_i.tolist()
            topk_log_probs[i] = topk_v.tolist()
            topk_words[i] = [d.TGT.vocab.itos[id] if not topk_words[i][j] else topk_words[i][j] for j, id in enumerate(topk_ids[i])]

            attns[i] = temp_dec_attn[i, -1, :].tolist() # tgt seq len x src seq len -> src_seq_len

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
        decoder_input = torch.as_tensor(tokens + prefix, dtype=torch.long, device=self.device).view(1, -1)
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
        while steps < self.max_length and len(result) < self.beam_size:
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
