# reading pkl files and generating a plot for evaluation_retrain_devset and evaluation_retrain_testset

import pickle

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.style.use('seaborn-darkgrid')

palette = sns.color_palette()

# TODO update
p = r'TODO/beam_20_metric_bleu_scores.pkl'

#doc1 = "test set"
doc2 = "test set"
#doc2 = "dev set"
doc1 = "dev set"

metrics = [
          "coverage_penalty",
          "confidence",
          "length",
          "keyphrase_score",
          "random"
          ]

name_map = {
            "length": "length",
            "coverage_penalty": "coverage penalty",
            "confidence": "confidence",
            "keyphrase_score": "keyphrase",
            "random": "random",
            }

with open(p, 'rb') as f:

    data = pickle.load(f)
    print(data)
    print(len(data))

for i, metric in enumerate(metrics):
    if metric in data:
        f, axes = plt.subplots()
        f.set_figheight(6)
        f.set_figwidth(6)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.axhline(y=0, color='dimgray', linestyle='-')

        plt.xlabel('# sentences of {} used for retraining'.format(doc1), fontsize=17)
        plt.ylabel('Δ BLEU of {}'.format(doc2), fontsize=17)

        x = [i * 100 for i in range(int(len(data[list(data.keys())[0]]) / 5 + 1))]
        plt.xticks(x)

        lastLine = None
        for m, value in data.items():
            c = metrics.index(metric)
            print(metric + " - " + m)

            x = [i * 20 for i in range(len(value))]
            delta = [(val[1] - val[0]) * 100 for val in value]

            line, = plt.plot(x, delta, marker='', linestyle="-" if m != "random" else "--",
                     color='black' if m == "random" else (palette[c] if m == metric else 'gray'),
                     linewidth=2 if metric == m else 1.5, alpha=0.9 if (m == metric or m == "random") else 0.5)
            if m == metric or m == "random":
                line.set_label(name_map[m])
            else:
                lastLine = line

        if lastLine:
            lastLine.set_label("other metrics")
        plt.legend(loc='upper left', ncol=1, fontsize=12)
        plt.savefig("bleu_" + metric + ".png")
        plt.close()

f, axes = plt.subplots()
f.set_figheight(6)
f.set_figwidth(6)

plt.axhline(y=0, color='dimgray', linestyle='-')

for metric, value in data.items():

    c = metrics.index(metric)
    print(metric)
    print(len(value))

    x = [i * 20 for i in range(len(value))]
    delta = [(val[1] - val[0]) * 100 for val in value]


    highlight = "coverage_penalty"
    #highlight = "keyphrase_score"
    #highlight = "length"

    plt.plot(x, delta, marker='', linestyle="-" if metric != "random" else "--",
                     color='black' if metric == "random" else palette[c],
                     linewidth=2 if metric == highlight else 1.5 if metric == "random" else 1, alpha=0.9 if metric == highlight or metric == "random" else 0.5,
                     label=name_map[metric])

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('# sentences of {} used for retraining'.format(doc1), fontsize=17)
    plt.ylabel('Δ BLEU of {}'.format(doc2), fontsize=17)

    plt.legend(loc='upper left', ncol=1, fontsize=12)
    plt.savefig('bleu_deltas.png')
