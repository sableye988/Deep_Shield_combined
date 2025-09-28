- Backend : Flask, Flask-SQLAlchemy, Flask-WTF  
- DB : SQLite  
- Frontend : HTML, CSS, Jinja2 Templates  
- 기타 : Pillow (이미지 처리)

---
*cmd, powershell에서 해주세요*


git clone https://github.com/sableye988/DeepShield.git


cd DeepShield

---
가상환경 생성 및 활성화(윈도우)


python -m venv venv


venv\Scripts\activate


---
패키지 설치


pip install -r requirements.txt

---

실행


flask run

---

파일 구조

DeepShield/


├─ app.py              # Flask 앱


├─ models.py           # DB 모델


├─ requirements.txt    # 설치 필요한 패키지 목록


├─ static/             # CSS


└─ templates/          # HTML 템플릿


이 프로젝트는 학습 및 연구 목적입니다.
