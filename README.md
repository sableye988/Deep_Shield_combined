````markdown
# Deep Shield 실행 가이드

git clone https://github.com/sableye988/Deep_Shield_combined.git
cd Deep_Shield_combined

# 가상환경 생성 & 활성화 (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# FastAPI(워터마크 API)
cd deepfake
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 새 터미널 열고 Flask(Web)
cd apps\webapp
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 환경변수(선택): SESSION_SECRET, GOOGLE_CLIENT_ID/SECRET
# MATE_API 미설정이면 기본 http://127.0.0.1:8000 사용
flask run

## 4. 유의사항

* 두 서버(FastAPI + Flask)를 **모두 실행해야 전체 기능이 정상 동작**합니다.
* DB가 초기화되지 않았다면 다음 명령으로 세팅하세요.
  cd apps\webapp
  $env:FLASK_APP="app.py"
  flask db upgrade
* Alembic 오류나 리비전 불일치로 user 테이블이 안 만들어질 경우 - python -c "from app import app, db; app.app_context().push(); db.create_all(); print('tables:', __import__('sqlalchemy').inspect(db.engine).get_table_names())"
* `assets/hanshin.png`는 워터마크 검증용 참조 이미지이므로 반드시 필요합니다.
