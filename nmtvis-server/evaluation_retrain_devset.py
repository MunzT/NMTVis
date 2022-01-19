# Evaluation with retraining for the development set

from datetime import datetime

from evaluation_retrain_share import *

num_sentences_test = 1000
num_sentences_dev = 500
evaluate_every = 20

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

for case in ["seq_en_de"]:
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

    # input training and test data
    src_BPE_test = f".data/evaluation/khresmoi-summary-test.{src_lang}.tok.bpe"
    tgr_BPE_test = f".data/evaluation/khresmoi-summary-test.{tgt_lang}.tok.bpe"
    src_BPE_dev = f".data/evaluation/khresmoi-summary-dev.{src_lang}.tok.bpe"
    tgr_BPE_dev = f".data/evaluation/khresmoi-summary-dev.{tgt_lang}.tok.bpe"

    # to save translation results
    initialTranslationFile = f"initTranslation_{model_type}_{src_lang}-{tgt_lang}.pkl"
    initialScoreFile = f"initScores_{model_type}_{src_lang}-{tgt_lang}.pkl"
    initialTestTranslationFile = f"initTranslationresult_attentions_{model_type}_{src_lang}-{tgt_lang}.pkl"
    translationFile = f"_translation_{model_type}_{src_lang}-{tgt_lang}.pkl"

    dir = os.path.join("_experiments", str(case) + "_retrainExperiments_" + dt_string)
    print(dir)
    os.makedirs(dir)

    model = load_model(src_lang, tgt_lang, model_type, device=DEVICE)

    exp = MetricExperiment(model,
                           src_lang, tgt_lang, model_type,
                           src_BPE_dev,
                           tgr_BPE_dev,
                           src_BPE_test,
                           tgr_BPE_test,
                           dir,
                           evaluate_every,
                           num_sentences_dev,
                           num_sentences_test,
                           reuseCalculatedTranslations,
                           reuseInitialTranslations,
                           initialTranslationFile,
                           initialScoreFile,
                           initialTestTranslationFile,
                           translationFile,
                           False)

    exp.run()
    exp.save_data()
