# Computer-Based Evaluation

For our experiments, we chose the domain-specific Khresmoi EN-DE data set by Dušek et al. [1].
It contains 1500 English sentences with complex medical terminology and German translations.
The test set contains 1000 sentences and the development set 500.

In order to prepare the data for our experiments you have to preprocess it.
First, you need to tokenize the respective documents from the data set:
*khresmoi-summary-dev.en*, *khresmoi-summary-dev.de*, *khresmoi-summary-test.en* and *khresmoi-summary-test.de*
In order to achieve this, we used the Tokenizer from [Mosesdecoder](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) with the following command for each document:

```bash
perl mosesdecoder-master\scripts\tokenizer\tokenizer.perl -l en -no-escape < [filename] > [filename].tok
```

Afterward we applied standard BPE with vocab size 32000 using [subword-nmt](https://github.com/rsennrich/subword-nmt):
```bash
./apply_bpe.py -c bpe.32000 < [filename].tok > [filename].tok.bpe
```

We used the *bpe.32000* file located in *nmtvis-server/.data/vocab*.

These files have to be placed in *nmtvis-server/.data/evaluation* and can now be used as input to our evaluation experiments.

We implemented different methods for a computer-based evaluation of our approach considering different aspects.
The corresponding Python scripts are located in *nmtvis-server/*:

- [evaluation_correlation](nmtvis-server/evaluation_correlation.py) performs analysis regarding the improvement of a document when analysing characTER and BLEU scores on both the Khresmoi test and development data.
- [evaluation_retrain_test](nmtvis-server/evaluation_retrain_devset.py) performs analysis regarding the improvement of BLEU scores when retraining and retranslating a document using the Khresmoi development data for retraining and the test data for retranslation.
- [evaluation_retrain_dev](nmtvis-server/evaluation_retrain_testset.py) performs analysis regarding the improvement of BLEU scores when retraining and retranslating a document using the Khresmoi test data for retraining and the development data for retranslation.
- [evaluation_retrain_plot](nmtvis-server/evaluation_retrain_plot.py) generates plots for the results created with *evaluation_retrain_devset* or *evaluation_retrain_testset*

[1] Dušek, Ondřej ; Hajič, Jan; Hlaváčová, Jaroslava; Libovický, Jindřich; Pecina, Pavel; Tamchyna, Aleš; Urešová, Zdeňka, 2017, 
Khresmoi Summary Translation Test Data 2.0, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), 
Faculty of Mathematics and Physics, Charles University, [http://hdl.handle.net/11234/1-2122](http://hdl.handle.net/11234/1-2122). 
