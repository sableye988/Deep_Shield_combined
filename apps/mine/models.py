from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 정수형 ID
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    protected_images = db.relationship('ProtectedImage', backref='user', lazy=True)

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
