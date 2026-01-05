# 🛡️ DeepShield (비가시성 워터마킹 기반 딥페이크 탐지 및 방지 솔루션)

> **"탐지를 넘어 예방까지, 당신의 이미지를 지키는 가장 확실한 방패"**

**DeepShield**는 딥페이크 기술의 악용으로 인한 허위 정보 생성 및 저작권 침해 문제를 해결하기 위해 개발된 웹 기반 솔루션입니다. 기존 서비스들이 사후 탐지에만 집중하는 한계를 넘어, **비가시성 워터마크(Invisible Watermark)** 기술을 통해 이미지 조작을 선제적으로 방어하고, 앙상블 딥러닝 모델을 통해 정밀한 진위 여부를 검증합니다.

---

## 📅 Project Info
* **프로젝트 명:** DeepShield (딥쉴드)
* **개발 기간:** 2025.03 ~ 2025.11
* **주요 기능:** 딥페이크 탐지, 비가시성 워터마크 삽입(방지), 워터마크 재추출 검증, 결과 리포트 제공

---

## 🛠️ Tech Stack

### Frontend
<img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white"/> <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white"/> <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black"/>

### Backend & Server
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white"/> <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white"/> <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white"/> <img src="https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white"/>

### AI & Image Processing
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>

---

## 🏗️ System Architecture

**DeepShield**는 웹 서비스의 유연성과 AI 연산의 고속 처리를 위해 **Flask(Web App)**와 **FastAPI(AI Server)**를 분리한 구조를 채택했습니다.

* **Frontend & Web Server (Flask):** 사용자 인터페이스, 세션 관리, DB 통신 담당.
* **AI API Server (FastAPI):** 워터마크 삽입/추출 및 딥페이크 탐지 알고리즘 비동기 처리.
* **Deployment:** Render 플랫폼을 활용하여 HTTPS 및 로드밸런싱이 적용된 배포 환경 구축.

<img width="637" height="364" alt="image" src="https://github.com/user-attachments/assets/d95d2bd2-bb1a-4f18-bdde-f1458b2ccf07" />



---

## 💡 Key Features & Algorithms

### 1. 딥페이크 방지 (Prevention) : 비가시성 워터마크
육안으로는 원본과 차이를 느낄 수 없도록 이미지를 변환하여 저작권 정보를 숨깁니다.

* **알고리즘:** DWT (Discrete Wavelet Transform, 이산 웨이블릿 변환) 기반
* **프로세스:**
    1.  이미지를 RGB에서 **YCbCr** 색상 공간으로 변환 (색상 채널 CbCr 보존).
    2.  밝기 정보인 **Y 채널에 DWT를 2단계(Level 2) 적용**하여 주파수 대역 분리.
    3.  압축과 변형에 강한 **HL, LH 영역**에 이진화된 워터마크 비트열 삽입.
    4.  역 DWT 변환 및 RGB 복원을 통해 시각적 변화 최소화.

### 2. 딥페이크 탐지 (Detection) : 앙상블 모델
이미지의 미세한 변조 흔적과 전역적인 문맥 정보를 동시에 분석합니다.

* **단계별 검증 시스템:**
    1.  **DB 대조:** 서버에 원본이 존재하는 경우 **PSNR 및 SSIM** 비교를 통해 즉시 변조 판별.
    2.  **AI 분석:** 원본이 없는 경우 3가지 모델의 가중치 앙상블(Weighted Ensemble) 수행.
* **사용 모델:**
    * **XceptionNet:** 딥페이크 데이터셋에서 높은 성능을 보이는 베이스라인 모델.
    * **EfficientNet:** 효율적인 파라미터로 정밀한 특징 추출.
    * **Vision Transformer (ViT):** 이미지의 전역적 문맥(Context)을 분석하여 비국소적 조작 탐지.

### 3. 워터마크 재추출 및 검증
* 조작되거나 훼손된 이미지에서 워터마크를 재추출하여 원본 손상 정도를 시각적으로 보여줍니다.
* 딥페이크 공격(Deepfake Attack) 발생 시 워터마크가 깨지는 현상을 이용해 조작 여부를 입증합니다.

---

## 🖥️ Service Screens

| 메인 페이지 | 탐지 결과 |
| :---: | :---: |
| <img width="639" height="368" alt="image" src="https://github.com/user-attachments/assets/535b057a-25a1-431d-bdd8-944a825d9a2f" /> | <img width="639" height="368" alt="image" src="https://github.com/user-attachments/assets/a1e68917-03ba-4ff7-a5db-3e9c6995cfd4" /> |
| **방지(워터마킹) 결과** | **워터마크 재추출(훼손 확인)** |
| <img width="639" height="368" alt="image" src="https://github.com/user-attachments/assets/d53196be-92b1-410a-b60d-7b7c4d7a06e6" /> | <img width="639" height="368" alt="image" src="https://github.com/user-attachments/assets/fb2962b4-1afd-4c58-836e-7491c3507dbc" /> |

---

## 👥 Team Members

| 이름 | 역할 |
| :---: | :---: |
| **이중근** | 팀장 (PM) |
| **정은성** | 팀원 |
| **이윤우** | 팀원 |
| **양동현** | 팀원 |

---

## 📂 Installation & Usage

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
* Alembic 오류나 리비전 불일치로 user 테이블이 안 만들어질 경우:
  python -c "from app import app, db; app.app_context().push(); db.create_all(); print('tables:', __import__('sqlalchemy').inspect(db.engine).get_table_names())"
  이후 flask db migrate/upgrade로 정리 권장
* `assets/hanshin.png`는 워터마크 검증용 참조 이미지이므로 반드시 필요합니다.
