# Evaluation: Correlation -- Distribution -- Processing in metric order

import os
import pickle
import statistics
from datetime import datetime

import matplotlib
import nltk
import numpy as np
import spacy
from matplotlib.ticker import FormatStrFormatter
from scipy.stats.stats import pearsonr

import data
import document
from keyphrase_extractor import DomainSpecificExtractor
from scorer import Scorer
from seq2seq.charac_ter import cer
from seq2seq.data_loader import LanguagePairLoader

matplotlib.use('Agg')
import matplotlib.pylab as plt

plt.rcParams["axes.titlesize"] = 15
import seaborn as sns

plt.style.use('seaborn-darkgrid')


name_map = {
            "length": "length",
            "coverage_penalty": "coverage penalty",
            "confidence": "confidence",
            "keyphrase_score": "keyphrase",
            }

reverse_sort_direction = {
                  "coverage_penalty": True,
                  "confidence": False,
                  "length": True,
                  "keyphrase_score": True,
                  }

metrics = [
          "coverage_penalty",
          "confidence",
          "length",
          "keyphrase_score"
          ]


def load_model(src_lang, tgt_lang, model_type):
    # load vocab + BPE encoding
    src_vocab, tgt_vocab = data.load_vocab(src_lang=src_lang, tgt_lang=tgt_lang)
    data.src_vocab = src_vocab
    data.tgt_vocab = tgt_vocab
    data.bpe = data.load_bpe()
    if src_lang == 'de':
        document.nlp = spacy.load('de_core_news_sm')
    elif src_lang == 'en':
        document.nlp = spacy.load('en_core_web_sm')
    document.nlp.add_pipe('sentence_division_suppresor', before='parser')

    print(src_lang)
    print(tgt_lang)
    print(model_type)

    if model_type == 'seq':
        print("Loading seq2seq model...")
        from seq2seq.models import Seq2SeqModel
        model = Seq2SeqModel.load(src_lang=src_lang, tgt_lang=tgt_lang, epoch=20)
    else:
        print("Loading transformer")
        from transformer.models import TransformerTranslator
        model = TransformerTranslator.load(src_lang=src_lang, tgt_lang=tgt_lang)

    return model


def compute_cter(target, translation):
    return cer(target.replace("@@ ", "").split(" "), translation.replace("@@ ", "").split(" "))


def compute_bleu(targets, translations):
    references, translations = [[target.replace("@@ ", "").split(" ")] for target in targets], [
        t.replace("@@ ", "").split(" ") for t in translations]

    bleu = nltk.translate.bleu_score.corpus_bleu(references, translations)
    return bleu


class CorrelationExperiment:
    def __init__(self, model, source_file, target_file, source_file2, target_file2, num_sentences=1000, beam_size=3):
        self.model = model
        self.source_file = source_file
        self.target_file = target_file
        self.source_file2 = source_file2
        self.target_file2 = target_file2
        self.num_sentences = num_sentences
        self.beam_size = beam_size

        self.translationList = []
        self.pairs = []
        self.scoresList = []

        self.scorer = Scorer()

        self.metric_to_cter = {}
        self.all_cter_scores = []

        self.metric_to_bad = {}

    # metric order -> correlation
    def plot_correlation(self, dir, prefix, filename):
        palette = sns.color_palette()

        for i, metric in enumerate(metrics):
            f, axes = plt.subplots()
            f.set_figheight(6)
            f.set_figwidth(6)

            x, y = [], []

            score_cter_tuples = []
            cters = []
            for score in self.metric_to_cter[metric]:
                for v in self.metric_to_cter[metric][score]:
                    cters.append(v)
                score_cter_tuples += [(score, v) for v in self.metric_to_cter[metric][score]]
                values = self.metric_to_cter[metric][score]
                x += [score] * len(values)
                y += values

            score_cter_tuples = sorted(score_cter_tuples, key=lambda x: x[0], reverse=reverse_sort_direction[metric])

            self.metric_to_bad[metric] = score_cter_tuples

            axes.set_ylim(-0.1, 1.1)

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            corr, p_val = pearsonr(x, y)

            axes.text(0.05, 0.95, "r = {0:.2f}".format(corr.item()), transform=axes.transAxes, va="top",
                      fontsize=13, weight="bold")

            sns.regplot(x, y, ax=axes, scatter_kws={'alpha': 0.2}, order=1, color=palette[i])

            plt.ylabel("CharacTER", fontsize=17)
            plt.xlabel("Metric: " + name_map[metric], fontsize=17)
            plt.savefig(os.path.join(dir, prefix + "_" + metric + filename))
            plt.close()


    # metric order -> document quality
    def plot_bad(self, dir, prefix, filename):

        palette = sns.color_palette()

        metric_percentage = {}

        mean = statistics.mean(self.all_cter_scores)
        stdev = statistics.stdev(self.all_cter_scores)
        threshold = mean + stdev

        for metric in metrics:
            bad_percentage = []
            curr_bad_count = 0
            for score, cter in self.metric_to_bad[metric]:
                if cter >= threshold:
                    curr_bad_count += 1
                bad_percentage.append(curr_bad_count)
            metric_percentage[metric] = bad_percentage

        for i, metric in enumerate(metrics):
            f, axes = plt.subplots()
            f.set_figheight(6)
            f.set_figwidth(6)

            bad_percentage = metric_percentage[metric]

            x = [100 * i / len(bad_percentage) for i in range(1, len(bad_percentage) + 1)]
            y = [100 * p / max(bad_percentage) for p in bad_percentage]
            line, = plt.plot(x, y, color=palette[i], linewidth=2, alpha=0.9)
            line.set_label(name_map[metric])
            line, = plt.plot(x, x, marker='', linestyle="--", color='black',
                     linewidth=1, alpha=0.9)
            line.set_label("theoretical baseline")

            for m in metrics:
                if m == metric:
                    continue
                line, = plt.plot([100 * i / len(metric_percentage[m]) for i in range(1, len(metric_percentage[m]) + 1)],
                       [100 * p / max(metric_percentage[m]) for p in metric_percentage[m]], marker='', color=palette[metrics.index(m)],
                                 linewidth=1, alpha=0.5, label=name_map[m], linestyle="-")

            plt.legend(loc='upper left', ncol=1, fontsize=12)

            plt.yticks([0, 25, 50, 75, 100], fontsize=15)
            plt.xticks([0, 25, 50, 75, 100], fontsize=15)

            plt.ylabel("% sentences with low quality covered", fontsize=17)
            plt.xlabel("% sentences covered (metric: " + name_map[metric] + ")", fontsize=17)

            plt.savefig(os.path.join(dir, prefix + "_percentages" + "_" + metric + filename))
            print("saved bad")
            plt.close()


    # BLEU values of remaining text
    # Sentences sorted from good to bad according to metric to calculate the BLEU score up to the current
    # sentence.
    # The plot shows the BLEU score when removing bad sentences first until only one good sentence remains.
    def plot_bleu(self, dir, prefix, filename):

        palette = sns.color_palette()

        metric_values = {}

        for metric in metrics:

            sorted_sentences = [(x, y, z) for _, x, y, z in
                                sorted(zip([s[metric] for s in self.scoresList], [p[0] for p in self.pairs],
                                           [p[1] for p in self.pairs],
                                           [" ".join(translation[:-1]) for translation in self.translationList]),
                                       reverse=not reverse_sort_direction[metric])]
            sources, targets, translations = zip(*sorted_sentences)

            values = []
            for i in range(len(sources)):
                s = [targets[i] for i in range(0, i + 1)]
                t = [translations[i] for i in range(0, i + 1)]

                bleu = compute_bleu(s, t)
                values.append(bleu)
            values.reverse()
            metric_values[metric] = values


        for i, metric in enumerate(metrics):
            f, axes = plt.subplots()
            f.set_figheight(6)
            f.set_figwidth(6)

            values = metric_values[metric]

            x = [100 * i / len(values[:-25]) for i in range(1, len(values[:-25]) + 1)]
            y = [100 * p for p in values[:-25]]
            line, = plt.plot(x, y, color=palette[i], linewidth=2, alpha=0.9)
            line.set_label(name_map[metric])

            for m in metrics:
                if m == metric:
                    continue
                line, = plt.plot([100 * i / len(metric_values[m][:-25]) for i in range(1, len(metric_values[m][:-25]) + 1)],
                         [100 * p for p in metric_values[m][:-25]], marker='', color=palette[metrics.index(m)],
                                 linewidth=1, alpha=0.5, label=name_map[m], linestyle="-")
            #line.set_label("other metrics")

            plt.legend(loc='upper left', ncol=1, fontsize=12)

            plt.yticks(fontsize=15)
            plt.xticks([0, 25, 50, 75, 100], fontsize=15)

            plt.ylabel("BLEU", fontsize=17)
            plt.xlabel("% sentences covered (metric: " + name_map[metric] + ")", fontsize=17)

            plt.savefig(os.path.join(dir, prefix + "_" + metric + filename))
            print("saved bleu")
            plt.close()


    # characTER
    # Sentences sorted from bad to good according to metric.
    # The plot shows CharacTER score of the currently processed sentence.
    def plot_cter2(self, dir, prefix, filename):

        palette = sns.color_palette()

        metric_values = {}

        for metric in metrics:

            sorted_sentences = [(x, y, z) for _, x, y, z in
                                sorted(zip([s[metric] for s in self.scoresList],
                                           [p[0] for p in self.pairs], [p[1] for p in self.pairs],
                                           [" ".join(translation[:-1]) for translation in self.translationList]),
                                       reverse=reverse_sort_direction[metric])]
            sources, targets, translations = zip(*sorted_sentences)

            values = []
            for i in range(len(sources)):
                s = targets[i]
                t = translations[i]

                cter = compute_cter(s, t)
                values.append(cter)
            metric_values[metric] = values

        for i, metric in enumerate(metrics):
            f, axes = plt.subplots()
            f.set_figheight(6)
            f.set_figwidth(6)

            axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            values = metric_values[metric]

            x = [100 * i / len(values)  for i in range(1, len(values) + 1)]
            y = [p for p in values]
            plt.plot(x, y, color=palette[i], linewidth=1, alpha=0.9)

            plt.xlim(0, 100)

            def movingaverage(interval, window_size):
                window = np.ones(int(window_size)) / float(window_size)
                return np.convolve(interval, window, 'valid')

            y_av = movingaverage(y, 100)

            plt.plot(x[50:-49], y_av, color='black', linewidth=3, alpha=0.9)

            plt.yticks(fontsize=15)
            plt.xticks([0, 25, 50, 75, 100], fontsize=15)

            plt.ylabel("CharacTER", fontsize=17)
            plt.xlabel("% sentences covered (metric: " + name_map[metric] + ")", fontsize=17)

            plt.savefig(os.path.join(dir, prefix + "_" + metric + filename))
            print("saved characTER (2)")
            plt.close()


    # Average characTER remaining text
    # Sentences sorted from good to bad according to metric to calculate the average CharacTER score up to the current
    # sentence.
    # The plot shows average CharacTER scores when removing bad sentences first until only one good sentence remains.
    def plot_cter(self, dir, prefix, filename):

        palette = sns.color_palette()

        metric_values = {}

        for metric in metrics:

            sorted_sentences = [(x, y, z) for _, x, y, z in
                                sorted(zip([s[metric] for s in self.scoresList],
                                           [p[0] for p in self.pairs], [p[1] for p in self.pairs],
                                           [" ".join(translation[:-1]) for translation in self.translationList]),
                                       reverse=not reverse_sort_direction[metric])]
            sources, targets, translations = zip(*sorted_sentences)

            values = []
            val = 0
            for i in range(len(sources)):
                s = targets[i]
                t = translations[i]

                cter = compute_cter(s, t)
                val += cter
                values.append(val / (i + 1))

            values.reverse()
            metric_values[metric] = values

        for i, metric in enumerate(metrics):
            f, axes = plt.subplots()
            f.set_figheight(6)
            f.set_figwidth(6)

            values = metric_values[metric]

            x = [100 * i / len(values[:-25]) for i in range(1, len(values[:-25]) + 1)]
            y = [p for p in values[:-25]]
            line, = plt.plot(x, y, color=palette[i], linewidth=2, alpha=0.9)
            line.set_label(name_map[metric])

            for m in metrics:
                if m == metric:
                    continue
                line, = plt.plot([100 * i / len(metric_values[m][:-25]) for i in range(1, len(metric_values[m][:-25]) + 1)],
                         [p for p in metric_values[m][:-25]], marker='', color=palette[metrics.index(m)],
                                 linewidth=1, alpha=0.5, label=name_map[m], linestyle="-")
            line.set_label("other metrics")

            plt.legend(loc='upper left', ncol=1, fontsize=12)

            plt.yticks(fontsize=15)
            plt.xticks([0, 25, 50, 75, 100], fontsize=15)

            plt.ylabel("CharacTER", fontsize=17)
            plt.xlabel("% sentences covered (metric: " + name_map[metric] + ")", fontsize=17)

            plt.savefig(os.path.join(dir, prefix + "_" + metric + filename))
            print("saved characTER")
            plt.close()


    # distribution plot
    def plot_distr(self, dir, prefix, filename):
        palette = sns.color_palette()

        bins_map = {"length": 60}

        for i, metric in enumerate(metrics):
            f, axes = plt.subplots()
            f.set_figheight(6)
            f.set_figwidth(6)

            metric_scores = []
            for value in self.metric_to_cter[metric]:
                metric_scores += len(self.metric_to_cter[metric][value]) * [value]

            if metric == "length":
                bins_map["length"] = max(metric_scores) - min(metric_scores) + 1

            plt.ylabel("Density", fontsize=17)
            plt.xlabel("Metric: " + name_map[metric], fontsize=17)

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            bins = bins_map[metric] if metric in bins_map else None
            dist_ax = sns.distplot(metric_scores, ax=axes, color=palette[i], bins=bins, hist_kws={"alpha": 0.2})
            ax2 = dist_ax.twinx()
            sns.boxplot(x=metric_scores, ax=ax2, color=palette[i])
            ax2.set(ylim=(-5, 5))

            plt.savefig(os.path.join(dir, prefix + "_" + metric + filename))
            plt.close()

        f, axes = plt.subplots()
        f.set_figheight(6)
        f.set_figwidth(6)
        plt.ylabel("Density", fontsize=17)
        plt.xlabel("CharacTER", fontsize=17)
        sns.distplot(self.all_cter_scores)
        plt.savefig(os.path.join(dir, prefix + "_" + "cter_dist.png"))
        plt.close()


    def run(self, src_lang, tgt_lang, dir, translationFile, scoresFile, attFile):
        loader = LanguagePairLoader(src_lang, tgt_lang, self.source_file, self.target_file)
        _, _, pairs = loader.load()

        loader2 = LanguagePairLoader(src_lang, tgt_lang, self.source_file2, self.target_file2)
        _, _, pairs2 = loader2.load()

        # concatenate both sets => all 1500 sentences
        pairs = pairs + pairs2

        self.pairs = pairs[:self.num_sentences]

        # Translate sources
        sources, targets, translations = [p[0] for p in self.pairs], [p[1] for p in self.pairs], []

        extractor = DomainSpecificExtractor(source_file=self.source_file, src_lang=src_lang, tgt_lang=tgt_lang,
                                            train_source_file=f".data/wmt14/train.tok.clean.bpe.32000.{src_lang}",
                                            train_vocab_file=f".data/vocab/train_vocab_{src_lang}.pkl")

        keyphrases = extractor.extract_keyphrases(n_results=100)

        self.translationList = []
        attentionList = []
        self.scoresList = []
        prefix = "_experiments/translated_beam3"

        if os.path.isfile(os.path.join(prefix, translationFile)) \
                and os.path.isfile(os.path.join(prefix, scoresFile)) \
                and os.path.isfile(os.path.join(prefix, attFile)):
            print("Translation reloaded")
            with open(os.path.join(prefix, translationFile), 'rb') as f:
                self.translationList = pickle.load(f)
            with open(os.path.join(prefix, attFile), 'rb') as f:
                attentionList = pickle.load(f)
            with open(os.path.join(prefix, scoresFile), 'rb') as f:
                self.scoresList = pickle.load(f)

        else:
            for i, pair in enumerate(self.pairs):
                if i % 10 == 0:
                    print("Translated {} of {}".format(i, len(self.pairs)))

                translation, attn, _ = self.model.translate(pair[0], beam_size=self.beam_size)
                translations.append(" ".join(translation[:-1]))

                scores = self.scorer.compute_scores(pair[0], " ".join(translation), attn, keyphrases, "")

                self.translationList.append(translation)
                attentionList.append(attn)
                self.scoresList.append(scores)

            pickle.dump(self.translationList,
                        open(os.path.join(dir, translationFile), "wb"))
            pickle.dump(self.scoresList,
                        open(os.path.join(dir, scoresFile), "wb"))
            pickle.dump(attentionList,
                        open(os.path.join(dir, attFile), "wb"))

        for i, pair in enumerate(self.pairs):
            if i % 10 == 0:
                print("Processing {} of {}".format(i, len(self.pairs)))

            for metric in self.scoresList[i]:
                #if metric == "coverage_penalty" and self.scoresList[i][metric] > 45: # remove some outliers
                #    continue
                #if metric == "keyphrase_score" and self.scoresList[i][metric] == 0:
                #    continue

                if not metric in self.metric_to_cter:
                    self.metric_to_cter[metric] = {}
                if not self.scoresList[i][metric] in self.metric_to_cter[metric]:
                    self.metric_to_cter[metric][self.scoresList[i][metric]] = []

                cter = compute_cter(pair[1], " ".join(self.translationList[i][:-1]))
                self.all_cter_scores.append(cter)
                self.metric_to_cter[metric][self.scoresList[i][metric]].append(cter)


num_sentences = 1500
beam_size = 3

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

for case in ["seq_de_en", "seq_en_de", "trafo_de_en", "trafo_en_de"]:
    if case == "seq_de_en":
        model_type = "seq"
        src_lang = "de"
        tgt_lang = "en"
    elif case == "seq_en_de":
        model_type = "seq"
        src_lang = "en"
        tgt_lang = "de"
    elif case == "trafo_de_en":
        model_type = "trafo"
        src_lang = "de"
        tgt_lang = "en"
    elif case == "trafo_en_de":
        model_type = "trafo"
        src_lang = "en"
        tgt_lang = "de"

    source_BPE = f".data/evaluation/khresmoi-summary-test.{src_lang}.tok.bpe"
    target_BPE = f".data/evaluation/khresmoi-summary-test.{tgt_lang}.tok.bpe"

    source2_BPE = f".data/evaluation/khresmoi-summary-dev.{src_lang}.tok.bpe"
    target2_BPE = f".data/evaluation/khresmoi-summary-dev.{tgt_lang}.tok.bpe"

    translationFile = f"initTranslationresult_translations_{model_type}_{src_lang}-{tgt_lang}.pkl"
    scoresFile = f"initTranslationresult_scores_{model_type}_{src_lang}-{tgt_lang}.pkl"
    attFile = f"initTranslationresult_attentions_{model_type}_{src_lang}-{tgt_lang}.pkl"

    model = load_model(src_lang, tgt_lang, model_type)

    dir = os.path.join("_experiments", str(case) + "_correlationExperiments_beamSize_" + str(beam_size) + "_" + dt_string)
    os.makedirs(dir)

    exp1 = CorrelationExperiment(model, source_BPE, target_BPE, source2_BPE, target2_BPE, num_sentences, beam_size)
    exp1.run(src_lang, src_lang, dir, translationFile, scoresFile, attFile)
    exp1.plot_distr(dir, str(case), "_metrics_dist.png")
    exp1.plot_correlation(dir, str(case), "_metric_corr.png")
    exp1.plot_bad(dir, str(case), "_metric_bad_progression.png")
    exp1.plot_cter(dir, str(case), "_metric_cter_progression.png")
    exp1.plot_cter2(dir, str(case), "_metric_cter2_progression.png")
    exp1.plot_bleu(dir, str(case), "_metric_bleu_progression.png")
