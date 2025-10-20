from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # 로컬 계정(아이디/비번)
    username = db.Column(db.String(100), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)

    # 소셜 계정
    provider = db.Column(db.String(50), nullable=True, index=True)        # 예) 'google'
    provider_id = db.Column(db.String(255), nullable=True, index=True)    # 예) Google sub
    email = db.Column(db.String(255), nullable=True, index=True)
    name = db.Column(db.String(255), nullable=True)
    picture = db.Column(db.String(512), nullable=True)

    protected_images = db.relationship('ProtectedImage', backref='user', lazy=True)

    __table_args__ = (
        db.UniqueConstraint('provider', 'provider_id', name='uq_provider_pid'),
    )

class ProtectedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    protected_filename = db.Column(db.String(255), nullable=False)
    watermark_strength = db.Column(db.Float, default=0.5)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DetectResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    uploaded_filename = db.Column(db.String(255))
    detect_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
