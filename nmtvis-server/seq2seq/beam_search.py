import data
import data as d
import math
import torch
import torch.nn as nn
from document import Hypothesis
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

UNK_token = -1


class BeamSearch:
    def __init__(self, decoder, encoder_outputs, decoder_hidden, 
                 beam_size=3, attentionOverrideMap=None, correctionMap=None, unk_map=None, beam_length=0.6,
                 beam_coverage=0.4, max_length=d.MAX_LEN):
        self.decoder = decoder
        self.encoder_outputs = encoder_outputs
        self.decoder_hidden = decoder_hidden
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
        assert len(list(self.correction_map.keys())) == 1

        prefix = list(self.correction_map.keys())[0]
        correction = self.correction_map[prefix]

        prefix = prefix + " " + correction
        raw_words = prefix.split(" ")[1:]
        bpe_words = [data.bpe.process_line(word) if not word.endswith("@@") else word for word in raw_words]
        words = [word for bpe_word in bpe_words for word in bpe_word.split(" ")]
        return words


    def process_corrections(self):
        """Apply BPE to correction map, ignoring words that are already BPE'd"""
        if not self.correction_map:
            return
        prefixes = list(self.correction_map.keys())

        for prefix in prefixes:
            raw_words = self.correction_map.pop(prefix).split(" ")
            bpe_words = [data.bpe.process_line(word) if not word.endswith("@@") else word for word in raw_words]
            words = [word for bpe_word in bpe_words for word in bpe_word.split(" ")]

            for i in range(len(words)):
                curr_prefix = " ".join(prefix.split(" ") + words[:i])
                self.correction_map[curr_prefix] = words[i]


    def decode_topk(self, latest_tokens, states, last_attn_vectors, partials):
        """Decode all current hypotheses on the beam, returning len(hypotheses) x beam_size candidates"""

        # len(latest_tokens) x self.beam_size)
        topk_ids = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        topk_log_probs = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        new_states = [None for _ in range(len(states))]
        new_attn_vectors = [None for _ in range(len(states))]
        attns = [None for _ in range(len(states))]
        topk_words = [["" for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        is_unk = [False for _ in range(len(latest_tokens))]

        # Loop over all hypotheses
        for token, state, attn_vector, i in zip(latest_tokens, states, last_attn_vectors, range(len(latest_tokens))):
            decoder_input = Variable(torch.LongTensor([token]))

            if use_cuda:
                decoder_input = decoder_input.cuda()

            attention_override = None
            if self.attention_override_map:
                if partials[i] in self.attention_override_map:
                    attention_override = self.attention_override_map[partials[i]]

            decoder_output, decoder_hidden, decoder_attention, last_attn_vector = self.decoder(decoder_input,
                                                                                               state,
                                                                                               self.encoder_outputs,
                                                                                               attn_vector,
                                                                                               attention_override)
            top_id = decoder_output.data.topk(1)[1]
            if use_cuda:
                top_id = top_id.cpu()

            top_id = top_id.numpy()[0].tolist()[0]

            if top_id == UNK_token:
                print("UNK found partial = {}".format(partials[i]))
            if top_id == UNK_token and self.unk_map and partials[i] in self.unk_map:
                # Replace UNK token based on user given mapping
                word = self.unk_map[partials[i]]
                topk_words[i][0] = word
                print("Replaced UNK token with {}".format(word))
                if word not in data.tgt_vocab.stoi:
                    is_unk[i] = True
                else:
                    idx = data.tgt_vocab.stoi[word]
                    decoder_output.data[0][idx] = 1000
            elif self.correction_map and partials[i] in self.correction_map:
                word = self.correction_map[partials[i]]
                print("Corrected {} for partial= {}".format(word, partials[i]))
                if not word in data.tgt_vocab.stoi:
                    topk_words[i][0] = word
                    is_unk[i] = True
                idx = data.tgt_vocab.stoi[word]
                decoder_output.data[0][idx] = 1000

            decoder_output = nn.functional.log_softmax(decoder_output, dim=-1)
            topk_v, topk_i = decoder_output.data.topk(self.beam_size)
            if use_cuda:
                topk_v, topk_i = topk_v.cpu(), topk_i.cpu()
            topk_v, topk_i = topk_v.numpy()[0], topk_i.numpy()[0]

            topk_ids[i] = topk_i.tolist()
            topk_log_probs[i] = topk_v.tolist()
            topk_words[i] = [data.tgt_vocab.itos[id] if not topk_words[i][j] else topk_words[i][j] for j, id in
                             enumerate(topk_ids[i])]

            new_states[i] = tuple(h.clone() for h in decoder_hidden)
            new_attn_vectors[i] = last_attn_vector.clone()
            attns[i] = decoder_attention.data
            if use_cuda:
                attns[i] = attns[i].cpu()
            attns[i] = attns[i].numpy().tolist()[0]

        return topk_ids, topk_words, topk_log_probs, new_states, new_attn_vectors, attns, is_unk


    def to_partial(self, tokens):
        return " ".join([data.tgt_vocab.itos[token] for token in tokens])


    def init_hypothesis(self):
        start_attn = [[0.0]]
        last_attn_vector = torch.zeros((1, self.decoder.hidden_size))

        if use_cuda:
            last_attn_vector = last_attn_vector.cuda()

        if not self.correction_map:
            return [Hypothesis([data.TGT.vocab.stoi[data.BOS_WORD]], [data.BOS_WORD], [0.0],
                               tuple(h.clone() for h in self.decoder_hidden),
                               last_attn_vector.clone(),
                               start_attn, [[]], [False]) for _ in range(self.beam_size)]

        # Assume at most 1 correction prefix at all times
        prefix = [data.TGT.vocab.stoi[data.BOS_WORD]] + [data.tgt_vocab.stoi[token] for token in self.prefix]
        # We need: hidden state at the end of prefix, last_attn_vector, attention, tokens, candidates

        tokens = []
        candidates = [[]]
        decoder_hidden = tuple(h.clone() for h in self.decoder_hidden)

        start_attn = [[0 for _ in range(self.encoder_outputs.size(1))]]

        for token in prefix:
            decoder_input = Variable(torch.LongTensor([token]))

            # Compute
            if use_cuda:
                decoder_input = decoder_input.cuda()
            decoder_output, decoder_hidden, decoder_attention, last_attn_vector = self.decoder(decoder_input,
                                                                                               decoder_hidden,
                                                                                               self.encoder_outputs,
                                                                                               last_attn_vector)
            # Update
            tokens += [token]

            attn = decoder_attention.data
            if use_cuda:
                attn = attn.cpu()
            attn = attn.numpy().tolist()[0]
            start_attn += attn

            topk_v, topk_i = decoder_output.data.topk(self.beam_size)
            if use_cuda:
                topk_v, topk_i = topk_v.cpu(), topk_i.cpu()
            topk_v, topk_i = topk_v.numpy()[0], topk_i.numpy()[0]

            topk_ids = topk_i.tolist()
            topk_log_probs = topk_v.tolist()
            candidates.append([data.tgt_vocab.itos[i] for i in topk_ids])

        return [Hypothesis(list(tokens), [data.tgt_vocab.itos[token] for token in tokens], [0.0] * len(tokens),
                           tuple(h.clone() for h in decoder_hidden),
                           last_attn_vector.clone(),
                           list(start_attn), candidates, [False] * len(tokens)) for _ in range(self.beam_size)]


    def search(self):
        last_attn_vector = torch.zeros((1, self.decoder.hidden_size))

        if use_cuda:
            last_attn_vector = last_attn_vector.cuda()

        hyps = self.init_hypothesis()

        for h in hyps:
            h.alpha = self.beam_length
            h.beta = self.beam_coverage

        result = []

        steps = 0

        while steps < self.max_length and len(result) < self.beam_size:
            latest_tokens = [hyp.latest_token for hyp in hyps]
            states = [hyp.state for hyp in hyps]
            partials = [self.to_partial(hyp.tokens) for hyp in hyps]
            last_attn_vectors = [hyp.last_attn_vector for hyp in hyps]
            all_hyps = []

            num_beam_source = 1 if steps == 0 else len(hyps)
            topk_ids, topk_words, topk_log_probs, new_states, new_attn_vectors, attns, is_unk = self.decode_topk(
                latest_tokens, states, last_attn_vectors,
                partials)

            for i in range(num_beam_source):
                h, ns, av, attn = hyps[i], new_states[i], new_attn_vectors[i], attns[i]

                for j in range(self.beam_size):
                    candidates = [data.tgt_vocab.itos[c] for c in (topk_ids[i][:j] + topk_ids[i][j + 1:])]

                    all_hyps.append(
                        h.extend(topk_ids[i][j], topk_words[i][j], topk_log_probs[i][j], ns, av, attn, candidates,
                                 is_unk[i]))

            # Filter
            hyps = []
            for h in self._best_hyps(all_hyps)[:self.beam_size]:
                if h.latest_token == data.TGT.vocab.stoi[data.EOS_WORD]:
                    result.append(h)
                else:
                    if len(h) == self.max_length:
                        result.append(h.extend(data.TGT.vocab.stoi[data.EOS_WORD], data.EOS_WORD, 0.0, [], [], [[0.0 for _ in range(len(h.attns[-1]))]], [], False))
                    else:
                        hyps.append(h)
                if len(result) == self.beam_size:
                    break
            steps += 1

        print("Beam Search found {} hypotheses for beam_size {}".format(len(result), self.beam_size))
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
