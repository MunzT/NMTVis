# NMTVis

[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--1849-d45815.svg)](https://doi.org/10.18419/darus-1849)
[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--1850-d45815.svg)](https://doi.org/10.18419/darus-1850)
[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--2124-d45815.svg)](https://doi.org/10.18419/darus-2124)

![NMTVis](application.png)

NMTVis is a system that supports users in correcting translations generated with neural machine translation.
Our system supports translation with an LSTM-based and the Transformer architecture.

It provides a web interface for visualizations and to support interactive correction.

You can find an introduction to our system [here](INTRO.md).
If you want to reproduce our evaluation you can follow the steps [here](EVALUATION.md).

## Starting NMTVis

In order to run the system you need [Node.js](https://nodejs.org), [AngularJS](https://angularjs.org), [Python 3](https://www.python.org), and several Python modules defined in requirements.txt (see below for a more detailed description to prepare everything).

You start the server from the *nmtvis-server* directory with
```bash
python server.py -m [seq|trafo] -sl [de|en] -tl [de|en]
```

```bash
-m: model with options LSTM (seq) and Transformer (trafo)
-sl: source langugage with options German (de) and English (en)
-tl: target langugage with options German (de) and English (en)
```

You start the client from the *nmtvis-client* directory with
```bash
ng serve
```

Next, start the application in a browser:
[localhost:4200/documents](http://localhost:4200/documents).

Now, create a user (or use user = admin, password = admin for the default account) and upload documents for translation and correction.
Each document may contain multiple source sentences, each sentence has to be given in a separate line.
After uploading a new document you have to refresh the page.

## Preparing NMTVis

In the following, we describe in more detail how to set everything up to run the system.

### Translation Models

We prepared some trained models: [https://doi.org/10.18419/darus-1850](https://doi.org/10.18419/darus-1850).
They are for German to English translation and vice versa.
You have to place these models in the directories *nmtvis-server/.data/models/* under *transformer* for the Transformer models and *seq2seq* for the LSTM-based models.
Additionally, you have to place the vocabulary files in *nmtvis-server/.data/vocab/*.
Alternatively, you can train the models yourself, however, training will take several days.

### For the server:

First, install all python requirements (we used Python 3.8 and 3.9) for the server (in the *nmtvis-server/* directory ):
```bash
pip3 install -r requirements.txt
```

### For the client:

First, you need Node.js:

On Linux (we used Xubuntu 20.04), you can get it like this:
```bash
sudo apt install curl
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
sudo apt-get install -y nodejs
```
On Windows 10, download and install [Node.js](https://nodejs.org/en/download/).

On MacOS, we recommend using [homebrew](https://brew.sh/index_de):
```bash
brew install node
```

Next, install angular cli:
```bash
npm install -g @angular/cli
```

In case you have permission problems, you can follow [these](https://docs.npmjs.com/resolving-eacces-permissions-errors-when-installing-packages-globally) instructions.

Change to the directory nmtvis-client and install [AngularJS](https://angularjs.org):

```bash
npm install angular
```

## Preliminary Work

This project started as master thesis by Paul Kuznecov (his [NMT model](https://github.com/kuznecpl/nmtvis-model) and [server](https://github.com/kuznecpl/nmtvis-server) code)
and was later extended by an additional translation model and some additional features.

## License

Our project is licensed under the [MIT License](LICENSE.md).

## Citation

When referencing our work, please cite the journal paper *Visualization-Based Improvement of Neural Machine Translation* or conference paper *Visual-Interactive Neural Machine Translation*:

T. Munz, D. V??th, P. Kuznecov, N. T. Vu, and D. Weiskopf. Visualization-based improvement of neural machine translation. Computers Graphics, 2021.

```
@article{nmt2021_2,
  title = {Visualization-based improvement of neural machine translation},
  journal = {Computers & Graphics},
  year = {2021},
  doi = {10.1016/j.cag.2021.12.003},
  author = {Tanja Munz and Dirk V??th and Paul Kuznecov and Ngoc Thang Vu and Daniel Weiskopf},}
```

T. Munz, D. V??th, P. Kuznecov, N. T. Vu, and D. Weiskopf. Visual-interactive neural machine translation. In Proceedings of Graphics Interface 2021, GI 2021, pages 265 ??? 274. Canadian Information Processing Society, 2021.

```
@inproceedings{nmt2021,
  author = {Munz, Tanja and V??th, Dirk and Kuznecov, Paul and Vu, Ngoc Thang and Weiskopf, Daniel},
  title = {Visual-Interactive Neural Machine Translation},
  booktitle = {Proceedings of Graphics Interface 2021},
  series = {GI 2021},
  year = {2021},
  isbn = {978-0-9947868-6-9},
  location = {Virtual Event},
  pages = {265 -- 274},
  numpages = {10},
  doi = {10.20380/GI2021.30},
  publisher = {Canadian Information Processing Society},
}
```