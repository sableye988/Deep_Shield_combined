````markdown
# Deep Shield 실행 가이드

## 1. Python 환경 준비
- Python **3.12 이상 권장**

### 1-1. 가상환경 생성
```bash
python -m venv venv
````

### 1-2. 가상환경 활성화

* **Windows (PowerShell):**

  ```bash
  venv\Scripts\activate
  ```
* **Windows (Git Bash):**

  ```bash
  source venv/Scripts/activate
  ```
* **macOS / Linux:**

  ```bash
  source venv/bin/activate
  ```

### 1-3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. 데이터베이스 (최초 실행 시)

* DB: **SQLite (`site.db`)**
* 마이그레이션 실행:

  ```bash
  flask db upgrade
  ```

---

## 3. 서버 실행

프로젝트는 **두 개의 서버**가 필요합니다.

* **터미널 1 (FastAPI: 워터마크 API 서버)**

  ```bash
  cd apps/mate
  uvicorn main:app --reload --port 5002
  ```

* **터미널 2 (Flask: 웹사이트 서버)**

  ```bash
  cd apps/mine
  venv\Scripts\activate   # 또는 source venv/bin/activate
  flask run
  ```

  또는

  ```bash
  python app.py
  ```

---

## 4. 유의사항

* 두 서버(FastAPI + Flask)를 **모두 실행해야 전체 기능이 정상 동작**합니다.

  * Flask(`mine`)만 실행 시: 로그인, 탐지 페이지까지만 동작
  * FastAPI(`mate`) 실행 시: 방지 및 재검사 페이지 정상 동작
* `assets/hanshin.png`는 워터마크 검증용 참조 이미지이므로 반드시 필요합니다.

```
원해? 제가 `README.md` 최종본을 이렇게 다듬어서 줄까?
```
