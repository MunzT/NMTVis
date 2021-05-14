from sqlalchemy_utils import PasswordType, UUIDType
from sqlalchemy_utils import force_auto_coercion

from shared import db

force_auto_coercion()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(
        PasswordType(schemes=['pbkdf2_sha512', ])
    )
    documents = db.relationship('Document', backref="user", lazy=True)
    surveydata = db.Column(db.String(5000), nullable=True)
    current_experiment_index = db.Column(db.Integer, default=0)

    def __repr__(self):
        return '<User %r>' % self.username


class Document(db.Model):
    id = db.Column(UUIDType(binary=False), primary_key=True)
    name = db.Column(db.Unicode(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model = db.Column(db.Text, nullable=False)

    @property
    def path(self):
        return "document-" + self.id + ".document"

    def __repr__(self):
        return '<Document %r>' % self.name
