import data
import data as d
import seq2seq.hp as hp
import shared
import torch
import torch.nn as nn
import torch.nn.functional as F
from document import Translation
from pytorch_lightning.metrics.functional import bleu_score
from seq2seq.beam_search import BeamSearch
from seq2seq.hp import n_layers
from shared import TranslationModel

use_cuda = torch.cuda.is_available()


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=d.MAX_LEN):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))


    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size, max_len)  # B x S

        if use_cuda:
            attn_energies = attn_energies.cuda()

        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.bmm(encoder_output.permute([1, 2, 0]))
            return energy.squeeze(1)

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.bmm(energy.permute(0, 2, 1))  # B x 1 x H bmm B x H x S => B x 1 x S
            return energy.squeeze(1)  # B X S

        elif self.method == 'concat':
            S = encoder_output.size(1)
            B = encoder_output.size(0)

            hidden = hidden.repeat(1, S, 1)  # B x 1 x H => B x S x H
            concat = torch.cat((hidden, encoder_output), 2)  # B x S x 2H

            energy = self.attn(concat).transpose(2, 1)  # B x H x S
            energy = F.tanh(energy)
            v = self.v.repeat(B, 1).unsqueeze(1)
            energy = torch.bmm(v, energy)
            return energy.squeeze(1)


class LSTMEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, n_layers=n_layers, dropout=hp.dropout, bidirectional=True):
        super(LSTMEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = self.hidden_size // self.num_directions

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, n_layers, dropout=self.dropout, bidirectional=True, batch_first=True)


    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over the whole input sequence)
        if use_cuda and hidden: hidden = hidden.cuda()

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # unpack (back to padded)


        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return outputs, hidden


    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden


class LSTMAttnDecoderRNN(nn.Module):
    def __init__(self, encoder, attn_model, hidden_size, output_size, n_layers=n_layers, dropout=hp.dropout):
        super(LSTMAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.output_size = output_size
        self.n_layers = encoder.n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, self.hidden_size)


    def forward(self, input_seq, last_hidden, encoder_outputs, last_attn_vector, attention_override=None):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)

        # B x O => B x H
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)

        # Input feeding
        # B x H cat B x H => B x 2H
        embedded = torch.cat((embedded, last_attn_vector), dim=1)

        embedded = embedded.view(batch_size, 1, 2 * self.hidden_size)  # S= B x 1 x 2H

        # rnn_output: B x 1 x H, hidden: 1 x B x H
        rnn_output, hidden = self.lstm(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        if attention_override is not None:
            attn_weights = torch.tensor(attention_override + [0], dtype=torch.float).view(1, 1, len(attention_override) + 1)

        context = attn_weights.bmm(encoder_outputs)  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(1)  # S=B x 1 x H -> B x H
        context = context.squeeze(1)  # B x S=1 x N -> B x H
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))  # B x H

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights, concat_output


class Seq2SeqModel(TranslationModel):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


    @classmethod
    def load(cls, src_lang, tgt_lang, epoch):
        import os
        checkpoint_name = os.path.join(shared.SEQ2SEQ_CHECKPOINT_PATH, f'seq2seq_{src_lang}_{tgt_lang}_epoch_{epoch}.pt')

        print("Loading seq2seq model from checkpoint", checkpoint_name)
        if not os.path.isfile(checkpoint_name):
            print("No seq2seq checkpoint found - please train model by calling `train_seq2seq.py`")
            exit()
        checkpoint = torch.load(checkpoint_name, map_location=lambda storage, loc: storage)
        encoder = LSTMEncoderRNN(len(data.src_vocab), hp.hidden_size, hp.embed_size)
        decoder = LSTMAttnDecoderRNN(encoder, hp.attention, hp.hidden_size, len(data.tgt_vocab))
        if torch.cuda.is_available():
            print("Using GPU")
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        else:
            print("Using CPU")
        encoder_state = checkpoint["encoder"]
        decoder_state = checkpoint["decoder"]
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
        return cls(encoder, decoder)


    @torch.no_grad()
    def best_translation(self, hyps):
        translation = [data.tgt_vocab.itos[token] for token in hyps[0].tokens] if hyps else [data.BOS_WORD]
        attn = hyps[0].attns if hyps else [0]

        return translation[1:], attn[1:]


    def eval(self):
        self.encoder.eval()
        self.decoder.eval()


    def train(self):
        self.encoder.train()
        self.decoder.train()


    @torch.no_grad()
    def translate(self, sentence, beam_size=3, beam_length=0.6, beam_coverage=0.4, attention_override_map=None,
                  correction_map=None, unk_map=None, max_length=d.MAX_LEN):

        self.eval()
        hyps = self.beam_search(sentence, beam_size, attention_override_map, correction_map, unk_map,
                                beam_length=beam_length, beam_coverage=beam_coverage, max_length=max_length)
        words, attention = self.best_translation(hyps)

        return words, attention, [Translation.from_hypothesis(h) for h in hyps]


    @torch.no_grad()
    def beam_search(self, input_seq, beam_size=3, attentionOverrideMap=None, correctionMap=None, unk_map=None,
                    beam_length=0.6, beam_coverage=0.4,
                    max_length=d.MAX_LEN):
        self.eval()

        input_seqs = [input_indexes_from_sentence(input_seq)]
        input_lengths = [len(seq) for seq in input_seqs]
        input_batches = torch.tensor(input_seqs, dtype=torch.long)

        if use_cuda:
            input_batches = input_batches.cuda()

        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        decoder_hidden = encoder_hidden

        beam_search = BeamSearch(self.decoder, encoder_outputs, decoder_hidden, beam_size,
                                 attentionOverrideMap,
                                 correctionMap, unk_map, beam_length=beam_length, beam_coverage=beam_coverage,
                                 max_length=max_length)
        result = beam_search.search()

        return result  # Return a list of indexes, one for each word in the sentence, plus EOS


    @torch.no_grad()
    def eval_bleu(self, source_test_file="", target_test_file=""):
        self.eval()
        print("Evaluating BLEU")
        references = []

        with open(target_test_file, "r") as f:
            for line in f.readlines():
                references.append([line.strip().replace('@@ ', "").split(" ")])

        source_sentences = []
        with open(source_test_file, "r") as f:
            for line in f.readlines():
                source_sentences.append(line.strip())

        source_sentences = source_sentences

        translations = []
        for sentence in source_sentences:
            translation, _, _ = self.translate(sentence, max_length=d.MAX_LEN)
            translations.append(" ".join(translation[:-1]).replace('@@ ', "").split())

        # import nltk
        # bleu = nltk.translate.bleu_score.corpus_bleu(references, translations)
        bleu = bleu_score(translations, references) 
        return bleu


def input_indexes_from_sentence(sentence):
    # encode string as BPE token list
    bpe_src = sentence.split()
    bpe_src = [tok for tok in bpe_src if tok != '@@@' and tok != '@']
    return [d.SRC.vocab.stoi[word] for word in bpe_src]


def output_indexes_from_sentence(sentence):
    return [d.TGT.vocab.stoi[d.BOS_WORD]] + [d.TGT.vocab.stoi[word] for word in sentence.split(' ')] + [d.TGT.vocab.stoi[d.EOS_WORD]]
