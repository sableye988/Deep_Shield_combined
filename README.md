<<<<<<< HEAD
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
pip install -r requirements.txt
# 환경변수(선택): SESSION_SECRET, GOOGLE_CLIENT_ID/SECRET
# MATE_API 미설정이면 기본 http://127.0.0.1:8000 사용
flask run

## 4. 유의사항

* 두 서버(FastAPI + Flask)를 **모두 실행해야 전체 기능이 정상 동작**합니다.

  * Flask(`mine`)만 실행 시: 로그인, 탐지 페이지까지만 동작
  * FastAPI(`mate`) 실행 시: 방지 및 재검사 페이지 정상 동작
* `assets/hanshin.png`는 워터마크 검증용 참조 이미지이므로 반드시 필요합니다.
=======
# deepfake_blind
>>>>>>> fe9050199043b196cd8ac9a496563002dd1742d7
