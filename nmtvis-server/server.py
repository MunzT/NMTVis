import argparse
import json
import os
import pickle
import random
import spacy
import subprocess
import sys
import torch
import torchtext.data
import torchtext.datasets
from flask import Flask, jsonify, request, send_file, redirect, url_for, Response
from flask_compress import Compress
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)
from flask_sqlalchemy import SQLAlchemy
from hashlib import md5
from simplediff import html_diff
from uuid import uuid4
from werkzeug.utils import secure_filename

import data
import document
from db_models import User, Document as DBDocument
from document import Document, Sentence
from keyphrase_extractor import DomainSpecificExtractor
from scorer import Scorer
from shared import db, DOCUMENTS_FOLDER, DATA_FOLDER, UPLOAD_FOLDER, DB_PATH, ALLOWED_EXTENSIONS, \
    EXPORT_TRANSLATIONS_FOLDER


# Flask config
def setup_flask():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['JWT_SECRET_KEY'] = "supersecretkeyhastochange"
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    CORS(app)
    Compress(app)
    jwt = JWTManager(app)

    # init DB
    with app.app_context():
        db.init_app(app)
        if not os.path.exists(DB_PATH):
            db.create_all()
            admin = User(username="admin", password="admin")
            if db.session.query(User).filter_by(username='admin').count() < 1:
                db.session.add(admin)
                db.session.commit()
    return app


# start flask
app = setup_flask()

@app.route('/api/auth/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if not username:
        return jsonify({"msg": "Missing username parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    maybe_user = User.query.filter_by(username=username).first()
    if not maybe_user or maybe_user.password != password:
        return jsonify({"msg": "Bad username or password"}), 401

    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token, status="success", username=username), 200


@app.route('/api/auth/register', methods=['POST'])
def register():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if not username:
        return jsonify({"msg": "Missing username parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    maybe_user = User.query.filter_by(username=username).first()
    if maybe_user:
        return jsonify({"msg": "Username already registered"}), 409
    new_user = User(username=username, password=password)

    # Create sample document
    if os.path.isfile(DOCUMENTS_FOLDER + "/document-SAMPLE.document"):
        sample_document = get_document("SAMPLE")
        id = uuid4()
        dbDocument = DBDocument(id=id, name="Document", user=new_user, model=model_name)
        save_document(sample_document, id)
        db.session.add(dbDocument)

    db.session.add(new_user)
    db.session.commit()

    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token, status="success", username=username), 200


@app.route('/protected', methods=['GET'])
@jwt_required
def protected():
    # Access the identity of the current user with get_jwt_identity
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200


def addTranslation(root, translation):
    if not translation.words:
        return

    for child in root["children"]:
        if child["name"] == translation.words[0]:
            addTranslation(child, translation.slice())
            return

    # print("--------")
    # print("add trans", len(translation.words), len(translation.attns), len(translation.log_probs))
    # print(translation.words)

    attn = translation.attns[0] if len(translation.attns) > 0 else []
    node = {"name": translation.words[0],
            "logprob": translation.log_probs[0],
            "children": [],
            "attn": attn,
            "candidates": translation.candidates[0],
            "is_golden": translation.is_golden,
            "is_unk": translation.is_unk[0]}
    root["children"].append(node)
    addTranslation(node, translation.slice())


def translationsToTree(translations):
    root = {"name": "root", "logprob": 0, "children": [], "candidates": [], "is_golden": False}

    for translation in translations:
        addTranslation(root, translation)

    if root["children"]:
        return root["children"][0]
    else:
        return root


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/correctTranslation", methods=["POST"])
@jwt_required
def correctTranslation():
    data = request.get_json()
    translation = data["translation"]
    beam = data["beam"]
    document_unk_map = data["document_unk_map"]
    attention = data["attention"]
    document_id = data["document_id"]
    sentence_id = data["sentence_id"]

    document = get_document(document_id)

    extractor = DomainSpecificExtractor(source_file=document.filepath, src_lang=SRC_LANG, tgt_lang=TGT_LANG,
                                        train_source_file=f".data/wmt14/train.tok.clean.bpe.32000.{SRC_LANG}",
                                        train_vocab_file=f".data/vocab/train_vocab_{SRC_LANG}.pkl")
    keyphrases = extractor.extract_keyphrases()

    for key in document_unk_map:
        if key not in document.unk_map:
            document.unk_map[key] = document_unk_map[key]
        else:
            # Merge list values
            document.unk_map[key] = list(set(document.unk_map[key]) | set(document_unk_map[key]))

    sentence = document.sentences[int(sentence_id)]

    if translation != sentence.translation:
        sentence.diff = html_diff(sentence.translation[:-4].replace("@@ ", ""),
                                  translation[:-4].replace("@@ ", ""))
    sentence.translation = translation
    sentence.corrected = True
    sentence.flagged = False
    sentence.attention = attention
    sentence.beam = beam

    scorer = Scorer()
    score = scorer.compute_scores(sentence.source, sentence.translation, attention, keyphrases, "")
    score["order_id"] = sentence.score["order_id"]
    sentence.score = score

    document.sentences[int(sentence_id)] = sentence

    save_document(document, document_id)

    return jsonify({})


@app.route("/api/documents/<document_id>/sentences", methods=["GET"])
@jwt_required
def getSentences(document_id):
    document = get_document(document_id)

    sentences = []

    for sentence in document.sentences:
        sentences.append(
            {"id": str(sentence.id), "source": sentence.source, "translation": sentence.translation,
             "beam": sentence.beam,
             "score": sentence.score,
             "attention": sentence.attention,
             "corrected": sentence.corrected,
             "flagged": sentence.flagged,
             "diff": sentence.diff if hasattr(sentence, "diff") else ""})

    old_etag = request.headers.get('If-None-Match', '')
    data = json.dumps(sentences)
    new_etag = md5(data.encode("utf-8")).hexdigest()

    if old_etag == new_etag:
        return "", 304
    else:
        res = jsonify(sentences)
        res.headers["Etag"] = new_etag
        return res


@app.route("/api/documents", methods=["GET"])
@jwt_required
def getDocuments():

    res = []

    user = User.query.filter_by(username=get_jwt_identity()).first()

    if not user:
        return jsonify([]), 401

    for db_document in user.documents:
        if db_document.model == model_name:
            document = get_document(db_document.id)
            document_map = {"id": db_document.id, "name": db_document.name, "keyphrases": document.keyphrases}
            res.append(document_map)

    return jsonify(res)


def save_document(document, document_id):
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)
    pickle.dump(document, open(DOCUMENTS_FOLDER + "/document-" + str(document_id) + model_name + ".document", "wb"))


def get_document(document_id):
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)
    return pickle.load(open(os.path.join(DOCUMENTS_FOLDER, "document-" + str(document_id) + model_name + ".document"), "rb"))


@app.route("/api/documents/<document_id>/sentences/<sentence_id>", methods=["GET"])
@jwt_required
def getTranslationData(document_id, sentence_id):
    data = request.get_json()

    document = get_document(document_id)

    sentence = document.sentences[int(sentence_id)]

    translation, attn, beam = sentence.translation, sentence.attention, sentence.beam
    document_map = {"inputSentence": sentence.source, "translation": translation, "attention": attn,
                    "beam": beam, "document_unk_map": document.unk_map}
    return jsonify(document_map)


@app.route("/api/documents/<document_id>/sentences/<sentence_id>/corrected", methods=["POST"])
@jwt_required
def setCorrected(document_id, sentence_id):
    data = request.get_json()
    corrected = data["corrected"]

    document = get_document(document_id)

    sentence = document.sentences[int(sentence_id)]
    sentence.corrected = corrected
    save_document(document, document_id)

    return jsonify({"status": "ok"})


@app.route("/api/documents/<document_id>/sentences/<sentence_id>/flagged", methods=["POST"])
@jwt_required
def setFlagged(document_id, sentence_id):
    data = request.get_json()
    flagged = data["flagged"]

    document = get_document(document_id)

    sentence = document.sentences[int(sentence_id)]
    sentence.flagged = flagged
    save_document(document, document_id)

    return jsonify({"status": "ok"})


@app.route("/upload", methods=['POST'])
@jwt_required
def documentUpload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        document_name = request.args.get("document_name")
        id = uuid4()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        user = User.query.filter_by(username=get_jwt_identity()).first()
        dbDocument = DBDocument(id=id, name=document_name, user=user, model=model_name)

        document = Document(str(id), document_name, dict(), filepath)
        sentences = document.load_content(filename)
        sentences = list(filter(None, sentences))  # remove empty lines

        with open(filepath, "w", encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                f.write(sentence.replace("@@ ", "") + "\n" if i < len(sentences) - 1 else "")

        extractor = DomainSpecificExtractor(source_file=filepath, src_lang=SRC_LANG, tgt_lang=TGT_LANG,
                                            train_source_file=f".data/wmt14/train.tok.clean.bpe.32000.{SRC_LANG}",
                                            train_vocab_file=f".data/vocab/train_vocab_{SRC_LANG}.pkl")
        keyphrases = extractor.extract_keyphrases(n_results=30)

        scorer = Scorer()

        print("Translating {} sentences".format(len(sentences)))

        beamSize = 3
        for i, source in enumerate(sentences):
            translation, attn, translations = model.translate(source, beam_size=beamSize, beam_length=0.6,
                                                                      beam_coverage=0.4)
            print("Translated {} of {}".format(i + 1, len(sentences)))

            beam = translationsToTree(translations[:beamSize])

            # print("  ", translation)
            score = scorer.compute_scores(source, " ".join(translation), attn, keyphrases, "")
            score["order_id"] = i
            sentence = Sentence(i, source, " ".join(translation), attn, beam, score)

            document.sentences.append(sentence)

        print("Finished translation")

        keyphrases = [{"name": k, "occurrences": f, "active": False} for (k, f) in keyphrases]
        document.keyphrases = keyphrases
        db.session.add(dbDocument)
        db.session.commit()

        save_document(document, id)

        return jsonify({})
    return jsonify({})


@app.route("/beamUpdate", methods=['POST'])
@jwt_required
def beamUpdate():
    data = request.get_json()
    sentence = data["sentence"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])
    attentionOverrideMap = data["attentionOverrideMap"]
    correctionMap = data["correctionMap"]
    unk_map = data["unk_map"]

    translation, attn, translations = model.translate(sentence, beam_size,
                                                              beam_length=beam_length,
                                                              beam_coverage=beam_coverage,
                                                              attention_override_map=attentionOverrideMap,
                                                              correction_map=correctionMap, unk_map=unk_map)
    beam = translationsToTree(translations)
    res = {}
    res["beam"] = beam

    return jsonify(res)


@app.route("/attentionUpdate", methods=['POST'])
@jwt_required
def attentionUpdate():
    data = request.get_json()
    sentence = data["sentence"]
    attentionOverrideMap = data["attentionOverrideMap"]
    correctionMap = data["correctionMap"]
    unk_map = data["unk_map"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])

    translation, attn, translations = model.translate(sentence, beam_size,
                                                              beam_length=beam_length,
                                                              beam_coverage=beam_coverage,
                                                              attention_override_map=attentionOverrideMap,
                                                              correction_map=correctionMap, unk_map=unk_map)
    beam = translationsToTree(translations)
    res = {}
    res["beam"] = beam

    return jsonify(res)


@app.route("/wordUpdate", methods=['POST'])
@jwt_required
def wordUpdate():
    data = request.get_json()
    sentence = data["sentence"]
    attentionOverrideMap = data["attentionOverrideMap"]
    correctionMap = data["correctionMap"]
    unk_map = data["unk_map"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])

    translation, attn, translations = model.translate(sentence, beam_size,
                                                              beam_length=beam_length,
                                                              beam_coverage=beam_coverage,
                                                              attention_override_map=attentionOverrideMap,
                                                              correction_map=correctionMap, unk_map=unk_map)
    beam = translationsToTree(translations)
    res = {}
    res["beam"] = beam

    return jsonify(res)


@app.route("/translate", methods=['GET', 'POST'])
@jwt_required
def translate():
    data = request.get_json()
    sentence = data["sentence"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])

    translation, attn, translations = model.translate(sentence, beam_size, beam_length=beam_length,
                                                              beam_coverage=beam_coverage, apply_bpe=False)

    res = {}
    res["sentence"] = sentence
    res["translation"] = " ".join(translation)
    res["attention"] = attn

    beam = translationsToTree(translations)
    res["beam"] = beam

    return jsonify(res)


@app.route("/api/documents/<document_id>/retrain", methods=['POST'])
@jwt_required
def retrain(document_id):
    document = get_document(document_id)
    print('retrain', model_name)

    pairs = []
    for sentence in document.sentences:
        # Remove EOS at end
        if sentence.corrected:
            pairs.append([sentence.source.strip(), sentence.translation.replace(data.EOS_WORD, '').strip()])

    if len(pairs) < 2:
        return jsonify({})

    if 'seq' in model_name:
        from seq2seq.train import retrain_iters
        retrain_iters(model, pairs, [], SRC_LANG, TGT_LANG, batch_size=1, print_every=1, n_epochs=2,
                    learning_rate=0.0001)
    elif 'trafo' in model_name:
        model.retrain(SRC_LANG, TGT_LANG, pairs, last_ckpt=f'.data/models/transformer/trafo_{SRC_LANG}_{TGT_LANG}_ensemble.pt', epochs=15, batch_size=1, device='cpu')
    return jsonify({})


@app.route("/api/documents/<document_id>/translate", methods=['POST'])
@jwt_required
def retranslate(document_id):
    document = get_document(document_id)
    scorer = Scorer()
    extractor = DomainSpecificExtractor(source_file=document.filepath, src_lang=SRC_LANG, tgt_lang=TGT_LANG,
                                        train_source_file=f".data/wmt14/train.tok.clean.bpe.32000.{SRC_LANG}",
                                        train_vocab_file=f".data/vocab/train_vocab_{SRC_LANG}.pkl")
    keyphrases = extractor.extract_keyphrases()
    num_changes = 0

    for i, sentence in enumerate(document.sentences):
        sentence, num_changes = retranslateSentenceWithId(i, sentence, scorer, keyphrases, num_changes)

    save_document(document, document_id)
    return jsonify({"numChanges": num_changes})


def retranslateSentenceWithId(i, sentence, scorer, keyphrases, num_changes, beam_size = 3, force = False):
    print("Retranslate: " + str(i))

    if sentence.corrected and not force:
        return sentence, num_changes

    translation, attn, translations = model.translate(sentence.source, beam_size=beam_size)

    beam = translationsToTree(translations)

    score = scorer.compute_scores(sentence.source, " ".join(translation), attn, keyphrases, "")
    score["order_id"] = i

    translation_text = " ".join(translation)
    if translation_text != sentence.translation:
        num_changes += 1
        sentence.diff = html_diff(sentence.translation[:-4].replace("@@ ", ""),
                                  translation_text[:-4].replace("@@ ", ""))
    sentence.translation = translation_text
    sentence.beam = beam
    sentence.score = score
    sentence.attention = attn

    return sentence, num_changes


@app.route("/api/documents/<document_id>/sentences/<sentence_id>/beam_size/<beam_size>/translateSentence", methods=['POST'])
@jwt_required
def retranslateSentence(document_id, sentence_id, beam_size):
    document = get_document(document_id)
    scorer = Scorer()
    extractor = DomainSpecificExtractor(source_file=document.filepath, src_lang=SRC_LANG, tgt_lang=TGT_LANG,
                                        train_source_file=f".data/wmt14/train.tok.clean.bpe.32000.{SRC_LANG}",
                                        train_vocab_file=f".data/vocab/train_vocab_{SRC_LANG}.pkl")
    keyphrases = extractor.extract_keyphrases()
    num_changes = 0

    retranslateSentenceWithId(sentence_id, document.sentences[int(sentence_id)], scorer, keyphrases,
                              num_changes, int(beam_size), True)
    save_document(document, document_id)

    return jsonify({})


@app.route("/api/documents/<document_id>/filterForSimilarSentences/<reference_id>", methods=['POST'])
@jwt_required
def filterForSimilarSentences(document_id, reference_id):
    document = get_document(document_id)
    reference = document.sentences[int(reference_id)].source
    print("Filter for similar sentences: " + str(reference))

    scorer = Scorer()

    keyphrases = []
    for k in document.keyphrases:
        keyphrases.append((k["name"], k["occurrences"]))

    for i, sentence in enumerate(document.sentences):
        score = scorer.compute_scores(sentence.source, " ".join(sentence.translation), sentence.attention,
                                      keyphrases, reference)
        score["order_id"] = i

        sentence.score = score

    for i, sentence in enumerate(document.sentences):
        print(str(i) + "--" + str(sentence.score["similarityToSelectedSentence"]))

    save_document(document, document_id)

    return jsonify({})


@app.route("/api/documents/<document_id>/saveTranslation", methods=['POST'])
@jwt_required
def saveTranslation(document_id):
    document = get_document(document_id)
    sentences = ""
    for i, sentence in enumerate(document.sentences):
        s = sentence.translation[:-4]
        s = s.replace("@@ ", "") .replace("&apos;", "'") .replace("&quot;", '"')

        from nltk import word_tokenize
        s = word_tokenize(s)

        from nltk.tokenize.treebank import TreebankWordDetokenizer
        twd = TreebankWordDetokenizer()
        s = twd.detokenize(s)
        sentences = sentences + s
        sentences = sentences + "\n"

    print(sentences)

    import datetime
    import time
    time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    path = EXPORT_TRANSLATIONS_FOLDER
    fileName = "translation_" + get_jwt_identity() + "_" + time + "_" + str(document_id) + ".txt"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/" + fileName, 'w', encoding='utf-8') as file:
        file.write(sentences)

    return jsonify({})


@app.route("/api/experiments/next", methods=['POST'])
@jwt_required
def nextExperimentSentence():
    user = User.query.filter_by(username=get_jwt_identity()).first()
    data = request.get_json()
    experiment_metrics = data["experimentMetrics"]

    if not user:
        return jsonify({}), 401

    dbDocument = DBDocument.query.filter_by(user=user, name="Sample", model=model_name).first()
    document = get_document(dbDocument.id)
    current_sentence = document.sentences[user.current_experiment_index]
    current_sentence.experiment_metrics = experiment_metrics
    save_document(document, dbDocument.id)

    user.current_experiment_index += 1

    db.session.add(user)
    db.session.commit()

    if user.current_experiment_index >= len(document.sentences):
        return jsonify(status="finished")

    next_sentence = document.sentences[user.current_experiment_index]
    next_index = next_sentence.id
    experiment_type = next_sentence.experiment_type

    return jsonify(status="in_progress", documentId=dbDocument.id, sentenceId=next_index,
                   experimentType=experiment_type)


@app.route("/api/experiments/surveydata", methods=['POST'])
@jwt_required
def sendSurveyData():
    user = User.query.filter_by(username=get_jwt_identity()).first()
    data = request.get_json()

    if not user:
        return jsonify({}), 401

    user.surveydata = json.dumps(data)
    db.session.add(user)
    db.session.commit()

    return jsonify()


@app.route("/api/experiments/experimentdata", methods=['GET'])
@jwt_required
def getExperimentData():
    user = User.query.filter_by(username=get_jwt_identity()).first()
    data = request.get_json()

    if not user:
        return jsonify({}), 401

    users = User.query.all()
    result = []
    for user in users:
        if user.surveydata:
            result.append(json.loads(user.surveydata))

    return jsonify({"survey": result})


if __name__ == '__main__':
    global model
    global model_name
    global SRC_LANG
    global TGT_LANG

    parser = argparse.ArgumentParser(description='The translation server')
    parser.add_argument('-sl', '--src_lang', choices=['de', 'en'], default='de', help='source language')
    parser.add_argument('-tl', '--tgt_lang', choices=['de', 'en'], default='en', help='target language')
    parser.add_argument('-m', '--model', choices=["seq", "trafo"], default='seq', help='translation model (sequence2sequence, transformer)')
    args = parser.parse_args()

    # sanity checks
    assert args.src_lang != args.tgt_lang, "source and target languages can't be the same"
    SRC_LANG = args.src_lang
    TGT_LANG = args.tgt_lang
    # data.check_exist_data(src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    model_name = args.model + "_" + args.src_lang + "_" + args.tgt_lang

    # load vocab + BPE encoding
    src_vocab, tgt_vocab = data.load_vocab(src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    data.src_vocab = src_vocab  # globals
    data.tgt_vocab = tgt_vocab  # globals
    data.bpe = data.load_bpe()
    if args.src_lang == 'de':
        document.nlp = spacy.load('de_core_news_sm')
    elif args.src_lang == 'en':
        document.nlp = spacy.load('en_core_web_sm')
    document.nlp.add_pipe(document.sentence_division_suppresor, name='sent_fix', before='parser')

    print(len(src_vocab))
    print(len(tgt_vocab))

    # load models
    if args.model == 'seq':
        print("Loading seq2seq model...")
        from seq2seq.models import Seq2SeqModel
        model = Seq2SeqModel.load(src_lang=args.src_lang, tgt_lang=args.tgt_lang, epoch=20)
    else:
        print("Loading transformer")
        from transformer.models import TransformerTranslator
        model = TransformerTranslator.load(src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    app.run(host="127.0.0.1", port=5000, use_reloader=False, threaded=True)
