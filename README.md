## 개요
Deep_Shield 프로젝트의 **Mine App** 모듈입니다.  
Flask 기반으로 회원 관리, 딥페이크 탐지/방지, 워터마크 기반 재검사 기능을 제공합니다.

---

## 실행 방법

```bash
git clone https://github.com/sableye988/Deep_Shield_combined.git
cd Deep_Shield_combined/apps/mine
git pull origin main


python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


pip install -r requirements.txt


flask db upgrade


flask run


apps/mine/
 ├── app.py                # Flask 메인 실행 파일
 ├── models.py             # DB 모델 정의
 ├── migrations/           # Alembic DB 마이그레이션 파일
 ├── templates/            # HTML 템플릿
 ├── static/               # 정적 파일 (CSS, JS, 업로드 이미지 등)
 │    ├── uploads/         # 사용자가 업로드한 원본 이미지
 │    └── results/         # 워터마크 삽입 결과 이미지
 └── assets/               # 워터마크 참조 이미지 (예: hanshin.png)
