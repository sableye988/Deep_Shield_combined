from app import app
from models import db
from flask_migrate import upgrade, Migrate
import subprocess

# Flask-Migrate 초기화
migrate = Migrate(app, db)
db.init_app(app)

# 앱 컨텍스트 안에서 migration 적용
with app.app_context():
    try:
        upgrade()
        print("✅ Database migration applied successfully.")
    except Exception as e:
        print(f"⚠️ Migration failed or already applied: {e}")

# Gunicorn으로 서버 실행
subprocess.run([
    "gunicorn",
    "app:app",
    "--bind",
    "0.0.0.0:$PORT",
    "--workers",
    "1",
    "--log-level",
    "info"
])
