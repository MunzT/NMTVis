import os
import pickle
import random

import nltk
import numpy as np
import spacy
import torch
from nltk.translate import gleu_score
from tqdm import tqdm

import data
import document
from keyphrase_extractor import DomainSpecificExtractor
from scorer import Scorer
from seq2seq.data_loader import LanguagePairLoader
from seq2seq.models import Seq2SeqModel
from seq2seq.train import retrain_iters
from transformer.models import TransformerTranslator


# DEVICE = 'cuda:0'
DEVICE = 'cpu'

reuseCalculatedTranslations = False
reuseInitialTranslations = False

reverse_sort_direction = {"coverage_penalty": True,
                          "coverage_deviation_penalty": True,
                          "confidence": False,
                          "length": True,
                          "ap_in": True,
                          "ap_out": True,
                          "keyphrase_score": True,
                          }


def load_model(src_lang, tgt_lang, model_type, device='cpu'):
    src_vocab, tgt_vocab = data.load_vocab(src_lang=src_lang, tgt_lang=tgt_lang)
    data.src_vocab = src_vocab
    data.tgt_vocab = tgt_vocab
    data.bpe = data.load_bpe()
    if src_lang == 'de':
        document.nlp = spacy.load('de_core_news_sm')
    elif src_lang == 'en':
        document.nlp = spacy.load('en_core_web_sm')
    document.nlp.add_pipe('sentence_division_suppresor', before='parser')

    print(len(src_vocab))
    print(len(tgt_vocab))

    # load vocab + BPE encoding
    if model_type == 'seq':
        print("Loading seq2seq model...")
        model = Seq2SeqModel.load(src_lang=src_lang, tgt_lang=tgt_lang, epoch=20)
    else:
        print("Loading transformer")
        model = TransformerTranslator.load(src_lang=src_lang, tgt_lang=tgt_lang, device=device)
    return model


def compute_bleu(targets, translations):
    references, translations = [[target.replace("@@ ", "").split(" ")] for target in targets], [
        t.replace("@@ ", "").split(" ") for t in translations]

    bleu = nltk.translate.bleu_score.corpus_bleu(references, translations)
    return bleu


def compute_gleu(targets, translations):
    references, translations = [[target.replace("@@ ", "").split(" ")] for target in targets], [
        t.replace("@@ ", "").split(" ") for t in translations]
    return gleu_score.corpus_gleu(references, translations)


def unigram_recall(rare_words, targets, translations):
    numer, denom = 0, 0

    targets = [target.replace("@@ ", "") for target in targets]
    translations = [translation.replace("@@ ", "") for translation in translations]

    for target, translation in zip(targets, translations):
        for rare_word, _ in rare_words:
            denom += target.count(rare_word)
            numer += min(translation.count(rare_word), target.count(rare_word))

    return numer / denom if denom > 0 else 0


def unigram_precision(rare_words, targets, translations):
    numer, denom = 0, 0

    targets = [target.replace("@@ ", "") for target in targets]
    translations = [translation.replace("@@ ", "") for translation in translations]

    for target, translation in zip(targets, translations):
        for rare_word, _ in rare_words:
            denom += translation.count(rare_word)
            numer += min(translation.count(rare_word), target.count(rare_word))

    return numer / denom if denom > 0 else 0


class MetricExperiment:
    def __init__(self, model, src_lang, tgt_lang, model_type, source_file, target_file, test_source_file,
                 test_target_file, dir,
                 evaluate_every=10,
                 num_sentences=400,
                 num_sentences_test=500,
                 reuseCalculatedTranslations=False,
                 reuseInitialTranslations=False,
                 initialTranslationFile="",
                 initialScoreFile="",
                 initialTestTranslationFile="",
                 translationFile="",
                 batch_translate=True,
                 ):

        self.model = model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_type = model_type
        self.source_file = source_file
        self.target_file = target_file
        self.loader = LanguagePairLoader(src_lang, tgt_lang, source_file, target_file)
        self.test_loader = LanguagePairLoader(src_lang, tgt_lang, test_source_file, test_target_file)

        self.extractor = DomainSpecificExtractor(source_file=source_file, src_lang=src_lang, tgt_lang=tgt_lang,
                                                 train_source_file=f".data/wmt14/train.tok.clean.bpe.32000.{src_lang}",
                                                 train_vocab_file=f".data/vocab/train_vocab_{src_lang}.pkl")

        self.target_extractor = DomainSpecificExtractor(source_file=target_file, src_lang=tgt_lang, tgt_lang=src_lang,
                                                        train_source_file=f".data/wmt14/train.tok.clean.bpe.32000.{tgt_lang}",
                                                        train_vocab_file=f".data/vocab/train_vocab_{tgt_lang}.pkl")

        self.scorer = Scorer()
        self.scores = {}
        self.num_sentences = num_sentences
        self.num_sentences_test = num_sentences_test
        self.batch_translate = batch_translate
        self.evaluate_every = evaluate_every
        self.reuseCalculatedTranslations = reuseCalculatedTranslations
        self.reuseInitialTranslations = reuseInitialTranslations

        self.initialTranslationFile = initialTranslationFile
        self.initialScoreFile = initialScoreFile
        self.initialTestTranslationFile = initialTestTranslationFile
        self.translationFile = translationFile

        self.metric_bleu_scores = {}
        self.metric_gleu_scores = {}
        self.metric_precisions = {}
        self.metric_recalls = {}

        self.prefix = "_experiments/retrain_beam3"
        self.dir = dir

    def save_data(self):
        prefix = ("batch_" if self.batch_translate else "beam_") + str(self.evaluate_every) + "_"
        prefix = os.path.join(self.dir, prefix)
        pickle.dump(self.metric_bleu_scores, open(prefix + "metric_bleu_scores.pkl", "wb"))
        pickle.dump(self.metric_gleu_scores, open(prefix + "metric_gleu_scores.pkl", "wb"))
        pickle.dump(self.metric_precisions, open(prefix + "metric_precisions.pkl", "wb"))
        pickle.dump(self.metric_recalls, open(prefix + "metric_recalls.pkl", "wb"))
        print("Saved all scores")

    def save_translation(self, translation, metric, step):
        name = os.path.join(self.dir, metric + "_" + str(step) + self.translationFile)
        pickle.dump(translation, open(name, "wb"))
        print("Saved: " + name)

    def restore_translation(self, metric, step):
        name = os.path.join(self.prefix, metric + "_" + str(step) + self.translationFile)
        with open(name, 'rb') as f:
            return pickle.load(f)

    def save_initialTranslation(self, scores, translations):
        name = os.path.join(self.dir, self.initialTranslationFile)
        pickle.dump(translations, open(name, "wb"))
        name = os.path.join(self.dir, self.initialScoreFile)
        pickle.dump(scores, open(name, "wb"))
        print("Saved: " + name)

    def restore_initialTranslation(self):
        name = os.path.join(self.prefix, self.initialTranslationFile)
        with open(name, 'rb') as f:
            translations = pickle.load(f)
        name = os.path.join(self.prefix, self.initialScoreFile)
        with open(name, 'rb') as f:
            scores = pickle.load(f)
        return translations, scores

    def save_initialTestTranslation(self, translations):
        name = os.path.join(self.dir, self.initialTestTranslationFile)
        pickle.dump(translations, open(name, "wb"))
        print("Saved: " + name)

    def restore_initialTestTranslation(self):
        name = os.path.join(self.prefix, self.initialTestTranslationFile)
        with open(name, 'rb') as f:
            return pickle.load(f)

    def run(self):
        _, _, pairs = self.loader.load()
        random.shuffle(pairs)

        pairs = pairs[:self.num_sentences]
        sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []

        keyphrases = self.extractor.extract_keyphrases(n_results=100)

        target_keyphrases = self.target_extractor.extract_keyphrases(n_results=100)

        # translation and scores for order of retraining
        print('Translating ...')
        if not reuseCalculatedTranslations and not reuseInitialTranslations:
            for i, pair in enumerate(tqdm(pairs)):
                translation, attn, _ = self.model.translate(pair[0])
                translations.append(" ".join(translation[:-1]))

                metrics_scores = self.scorer.compute_scores(pair[0], " ".join(translation[:-1]), attn, keyphrases, "")
                for metric in metrics_scores:
                    if metric not in self.scores:
                        self.scores[metric] = []
                    self.scores[metric].append(metrics_scores[metric])
            self.save_initialTranslation(self.scores, translations)
        else:
            translations, self.scores = self.restore_initialTranslation()

        # initial test set translation
        _, _, test_pairs = self.test_loader.load()
        test_pairs = test_pairs[:self.num_sentences_test]
        test_sources, test_targets, test_translations = [p[0] for p in test_pairs], [p[1] for p in test_pairs], []

        if not reuseCalculatedTranslations and not reuseInitialTranslations:
            print('- not reusing translations: Translating...')
            for i, source in enumerate(tqdm(test_sources)):
                translation, attn, _ = self.model.translate(source)
                test_translations.append(" ".join(translation[:-1]))

            if self.batch_translate:
                test_translations = [t[:-6] for t in self.model.batch_translate(test_sources)]

            self.save_initialTestTranslation(test_translations)
        else:
            test_translations = self.restore_initialTestTranslation()

        metrics = [
            "random",
            "keyphrase_score",
            "coverage_penalty",
            "confidence",
            "length"
        ]

        print("Evaluating metrics...")
        for i, metric in enumerate(tqdm(metrics)):
            self.metric_bleu_scores[metric] = []
            self.metric_gleu_scores[metric] = []
            self.metric_precisions[metric] = []
            self.metric_recalls[metric] = []

            sourcesCopy = sources[:]
            targetsCopy = targets[:]
            translationsCopy = translations[:]

            self.evaluate_metric(self.src_lang, self.tgt_lang, self.model_type, sourcesCopy, targetsCopy,
                                 translationsCopy,
                                 self.scores[metric] if metric != "random" else [],
                                 metric,
                                 target_keyphrases,
                                 test_sources, test_targets, test_translations,
                                 need_sort=True if metric != "random" else False,
                                 reverse=reverse_sort_direction[metric] if metric != "random" else True)
            print()
            print(self.metric_bleu_scores)
            self.save_data()

    def shuffle_list(self, *ls):
        l = list(zip(*ls))

        random.shuffle(l)
        return zip(*l)

    def evaluate_metric(self, src_lang, tgt_lang, model_type, sources, targets, translations, scores, metric,
                        target_keyphrases,
                        test_sources, test_targets, test_translations,
                        need_sort=True, reverse=False):
        print()
        print("Evaluating {}".format(metric))
        base_bleu = compute_bleu(targets, translations)
        print("Base BLEU (of retraining data): {}".format(base_bleu))

        # Sort by metric
        if need_sort:
            sorted_sentences = [(x, y, z) for _, x, y, z in
                                sorted(zip(scores, sources, targets, translations), reverse=reverse)]
            sources, targets, translations = zip(*sorted_sentences)
        else:
            sources, targets, translations = self.shuffle_list(sources, targets, translations)

        n = len(sources)
        encoder_optimizer_state, decoder_optimizer_state = None, None

        pretraining_bleu = compute_bleu(test_targets, test_translations)
        pretraining_gleu = compute_gleu(test_targets, test_translations)
        print()
        print("pretraining BLEU of test set (before retraining)")
        print(pretraining_bleu)

        prerecall = unigram_recall(target_keyphrases, test_targets, test_translations)
        preprecision = unigram_precision(target_keyphrases, test_targets, test_translations)

        self.metric_bleu_scores[metric].append((pretraining_bleu, pretraining_bleu))
        self.metric_gleu_scores[metric].append((pretraining_gleu, pretraining_gleu))
        self.metric_recalls[metric].append((prerecall, prerecall))
        self.metric_precisions[metric].append((preprecision, preprecision))
        self.save_data()

        if isinstance(self.model, TransformerTranslator):
            # create a new checkpoint here that gets overwritten with each ij
            # Neccessary to load trainer state.
            current_ckpt = f'.data/models/transformer/trafo_{src_lang}_{tgt_lang}_ensemble.pt'

        print('Training...')
        for i in tqdm(range(0, n)):
            # retranslate only every 10th sentence
            # evaluets for the 0th, 10th, 20th, ... sentence -> computes for sentences (0..9), (10..19), (20..29);
            # first sentence i = 0; evaluate_every = 10
            if i % self.evaluate_every != 0:
                continue

            if not reuseCalculatedTranslations:

                # Now train, and compute BLEU again
                start = i
                end = min(i + self.evaluate_every, n)

                print()
                print("Correcting {} - {} of {} sentences".format(start, end - 1, n))

                if isinstance(self.model, Seq2SeqModel):
                    # same parameters that are used in the tool
                    encoder_optimizer_state, decoder_optimizer_state = retrain_iters(self.model,
                                                                                     [[x, y] for x, y in
                                                                                      zip(sources[start: end],
                                                                                          targets[start: end])], [],
                                                                                     src_lang, tgt_lang,
                                                                                     batch_size=1,
                                                                                     encoder_optimizer_state=encoder_optimizer_state,
                                                                                     decoder_optimizer_state=decoder_optimizer_state,
                                                                                     print_every=1,
                                                                                     n_epochs=15,
                                                                                     learning_rate=0.0001,
                                                                                     save_ckpt=i == n - 1)
                else:
                    # same parameters that are used in the tool
                    current_ckpt = self.model.retrain(src_lang, tgt_lang,
                                                      [[x, y] for x, y in
                                                       zip(sources[start: end], targets[start: end])],
                                                      last_ckpt=current_ckpt, epochs=15,
                                                      batch_size=1, device=DEVICE, save_ckpt=i == n - 1,
                                                      print_info=False)

                corrected_translations = []

                print(' - Translate using trained model')
                if not self.batch_translate:
                    # Translate trained model
                    for j in tqdm(range(0, len(test_sources))):
                        translation, _, _ = self.model.translate(test_sources[j])
                        corrected_translations.append(" ".join(translation[:-1]))
                else:
                    batch_translations = self.model.batch_translate(test_sources)
                    corrected_translations = [t[:-6] for t in batch_translations]

                self.save_translation(corrected_translations, metric, i)

            else:
                corrected_translations = self.restore_translation(metric, i)

            # Compute posttraining BLEU
            posttraining_bleu = compute_bleu(test_targets, corrected_translations)
            posttraining_gleu = compute_gleu(test_targets, corrected_translations)
            postrecall = unigram_recall(target_keyphrases, test_targets, corrected_translations)
            postprecision = unigram_precision(target_keyphrases, test_targets, corrected_translations)
            print("(Base BLEU {})".format(base_bleu))
            print("Delta Recall {} -> {}".format(prerecall, postrecall))
            print("Delta Precision {} -> {}".format(preprecision, postprecision))
            print("Delta GLEU: {} -> {}".format(pretraining_gleu, posttraining_gleu))
            print("Delta BLEU: {} -> {}".format(pretraining_bleu, posttraining_bleu))

            delta_bleu = posttraining_bleu - pretraining_bleu
            print("Delta: {}".format(delta_bleu))

            self.metric_bleu_scores[metric].append((pretraining_bleu, posttraining_bleu))
            self.metric_gleu_scores[metric].append((pretraining_gleu, posttraining_gleu))
            self.metric_recalls[metric].append((prerecall, postrecall))
            self.metric_precisions[metric].append((preprecision, postprecision))

            self.save_data()

        self.model = load_model(src_lang, tgt_lang, model_type, device=DEVICE)  # reload initial model
        return None
