import os
from flask_sqlalchemy import SQLAlchemy

# DB
DB_PATH = os.path.join('.data', 'server.db')
db = SQLAlchemy()

# PATHS
DATA_FOLDER = os.path.join('.data')
VOCAB_FOLDER = os.path.join('.data', 'vocab')
WMT_14_FOLDER = os.path.join(DATA_FOLDER, 'wmt14')
DOCUMENTS_FOLDER = os.path.join(DATA_FOLDER, 'documents')
UPLOAD_FOLDER = os.path.join(DATA_FOLDER, 'uploads')
MODELS_FOLDER = os.path.join(DATA_FOLDER, 'models')
EXPORT_TRANSLATIONS_FOLDER = os.path.join(DATA_FOLDER, 'savedDocuments')

# UPLOAD
ALLOWED_EXTENSIONS = set(['txt'])

# MODELS
SEQ2SEQ_CHECKPOINT_PATH = os.path.join(DATA_FOLDER, 'models', 'seq2seq')
TRAFO_CHECKPOINT_PATH = os.path.join(DATA_FOLDER, 'models', 'transformer')


class TranslationModel:
    @classmethod
    def load(cls):
        """ Model from checkpoint """
        raise NotImplementedError

    def translate(self, sentence, beam_size, beam_length, beam_coverage, attention_override_map,
                  correction_map, unk_map, max_length):
        raise NotImplementedError

