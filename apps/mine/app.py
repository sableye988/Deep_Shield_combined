from flask import Flask, render_template, request, redirect, url_for, session, flash, abort
from models import db, ProtectedImage, User, DetectResult
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_migrate import Migrate 
from PIL import Image, ImageOps
import os
from shutil import copyfile
import logging
import requests

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# 기본 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB
# 배포 시 세션 보안 권장 설정
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# HTTPS 사용 시:
# app.config['SESSION_COOKIE_SECURE'] = True

#은성님 워터마크 API
MATE_API = "http://127.0.0.1:5002"

# CSRF
csrf = CSRFProtect(app)

# 템플릿에서 {{ csrf_token() }} 사용 가능하게
@app.context_processor
def inject_csrf():
    return dict(csrf_token=generate_csrf)

# 허용 확장자
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def ensure_upload_dir():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def build_safe_timestamp_name(prefix: str, original_name: str) -> str:
    safe = secure_filename(original_name)
    _, ext = os.path.splitext(safe)
    ext = ext.lower()
    ts = f"{datetime.utcnow().timestamp()}"
    return f"{prefix}_{ts}{ext}"

def thumb_name(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    return f"{name}_thumb{ext}"

def save_thumbnail(src_path: str, dst_path: str, max_size=(600, 600)):
    try:
        im = Image.open(src_path)
        try:
            im = ImageOps.exif_transpose(im)  # EXIF 회전 보정
        except Exception:
            pass
        im.thumbnail(max_size)
        im.save(dst_path, optimize=True, quality=85)
    except Exception:
        # 썸네일 실패해도 서비스 계속
        pass


# DB 초기화
db.init_app(app)
migrate = Migrate(app, db) 
# ✅ Alembic 연동
# ❌ create_all()는 사용하지 않습니다. (마이그레이션으로 관리)
# with app.app_context():
#     db.create_all()


# 에러 핸들러
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash("파일이 너무 큽니다. 최대 20MB까지 업로드할 수 있어요.")
    return redirect(request.referrer or url_for('index'))


# ---------------------------
# 라우트
@app.route('/')
def index():
    return render_template('index.html')

# 회원가입
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash("이미 존재하는 아이디입니다.")
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash("회원가입이 완료되었습니다.")
        return redirect(url_for('login'))

    return render_template('signup.html')

# 로그인
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash("로그인 성공!")
            return redirect(url_for('mypage'))
        else:
            flash("로그인 실패: 아이디 또는 비밀번호가 틀립니다.")
            return redirect(url_for('login'))

    return render_template('login.html')

# 로그아웃 (GET)
@app.route('/logout')
def logout():
    session.clear()
    flash("로그아웃 되었습니다.")
    return redirect(url_for('index'))

# 탐지
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'user_id' not in session:
            flash("로그인이 필요합니다.")
            return redirect(url_for('login'))

        if 'image' not in request.files:
            flash("업로드된 파일이 없습니다.")
            return redirect(url_for('detect'))

        file = request.files['image']
        if not file or file.filename == '':
            flash("파일을 선택해주세요.")
            return redirect(url_for('detect'))

        if not allowed_file(file.filename):
            flash("이미지 파일만 업로드 가능합니다. (jpg, jpeg, png, gif, webp)")
            return redirect(url_for('detect'))
        if not (file.mimetype or '').startswith('image/'):
            flash("이미지 형식의 파일이 아닙니다.")
            return redirect(url_for('detect'))

        user_id = session['user_id']
        ensure_upload_dir()

        filename = build_safe_timestamp_name('detect', file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 썸네일
        detect_thumb = thumb_name(filename)
        detect_thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], detect_thumb)
        save_thumbnail(filepath, detect_thumb_path)

        # 예시 점수
        import random
        detect_score = round(random.uniform(0, 100), 2)

        new_result = DetectResult(
            user_id=user_id,
            uploaded_filename=filename,
            detect_score=detect_score
        )
        db.session.add(new_result)
        db.session.commit()

        # PRG: 결과를 세션에 담고 리다이렉트
        session['detect_result'] = {
            'uploaded_url': url_for('static', filename='uploads/' + filename),
            'uploaded_thumb_url': url_for('static', filename='uploads/' + detect_thumb),
            'score': detect_score
        }
        return redirect(url_for('detect'))

    result = session.pop('detect_result', None)
    return render_template('detect.html', result=result)


# 방지
@app.route('/prevent', methods=['GET', 'POST'])
def prevent():
    if request.method == 'POST':
        if 'user_id' not in session:
            flash("로그인이 필요합니다.")
            return redirect(url_for('login'))

        if 'image' not in request.files:
            flash("업로드된 파일이 없습니다.")
            return redirect(url_for('prevent'))

        file = request.files['image']
        if not file or file.filename == '':
            flash("파일을 선택해주세요.")
            return redirect(url_for('prevent'))

        if not allowed_file(file.filename):
            flash("이미지 파일만 업로드 가능합니다. (jpg, jpeg, png, gif, webp)")
            return redirect(url_for('prevent'))
        if not (file.mimetype or '').startswith('image/'):
            flash("이미지 형식의 파일이 아닙니다.")
            return redirect(url_for('prevent'))

        # 워터마크 강도(지금은 고정값, 필요 시 폼 입력 받아서 반영)
        strength = 0.5
        user_id = session['user_id']

        ensure_upload_dir()

        # 원본 저장
        original_filename = build_safe_timestamp_name('original', file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # ---- 워터마크 FastAPI로 전송하여 결과 PNG 받기 ----
        try:
            with open(original_path, "rb") as fp:
                # API의 필드명이 'host' 임에 주의
                r = requests.post(
                    f"{MATE_API}/embed_fixed_single_color",
                    files={"host": fp},
                    timeout=120
                )
            if r.status_code != 200:
                flash(f"워터마크 임베드 실패: {r.status_code} {r.text[:200]}")
                return redirect(url_for('prevent'))
        except Exception as e:
            app.logger.exception("FastAPI 호출 실패")
            flash("내부 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            return redirect(url_for('prevent'))

        # API가 PNG를 바이너리로 내려주므로, 결과 확장자는 .png로 저장
        base_noext, _ = os.path.splitext(original_filename)
        protected_filename = f"{base_noext}_protected.png"
        protected_path = os.path.join(app.config['RESULT_FOLDER'], protected_filename)
        with open(protected_path, "wb") as out:
            out.write(r.content)

        # 썸네일
        original_thumb = thumb_name(original_filename)
        protected_thumb = thumb_name(protected_filename)
        save_thumbnail(original_path, os.path.join(app.config['UPLOAD_FOLDER'], original_thumb))
        save_thumbnail(protected_path, os.path.join(app.config['RESULT_FOLDER'], protected_thumb))

        # DB 기록
        new_record = ProtectedImage(
            user_id=user_id,
            original_filename=original_filename,      # 업로드/원본은 uploads 폴더에
            protected_filename=protected_filename,    # 결과는 results 폴더에 (경로 주의)
            watermark_strength=strength
        )
        db.session.add(new_record)
        db.session.commit()

        # PRG: 결과를 세션에 담고 리다이렉트
        session['prevent_result'] = {
            "original_url": url_for('static', filename='uploads/' + original_filename),
            "modified_url": url_for('static', filename='results/' + protected_filename),
            "original_thumb_url": url_for('static', filename='uploads/' + original_thumb),
            "modified_thumb_url": url_for('static', filename='results/' + protected_thumb),
        }
        return redirect(url_for('prevent'))

    result = session.pop('prevent_result', None)
    return render_template('prevent.html', result=result)


# 마이페이지 (페이징)
@app.route('/mypage')
def mypage():
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    dpage = request.args.get('dpage', 1, type=int)
    mpage = request.args.get('mpage', 1, type=int)
    PER_PAGE = 8

    detect_pagination = DetectResult.query.filter_by(user_id=user_id)\
        .order_by(DetectResult.created_at.desc())\
        .paginate(page=dpage, per_page=PER_PAGE, error_out=False)

    modify_pagination = ProtectedImage.query.filter_by(user_id=user_id)\
        .order_by(ProtectedImage.created_at.desc())\
        .paginate(page=mpage, per_page=PER_PAGE, error_out=False)

    detect_history = [
        {
            'id': d.id,
            'date': d.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'thumb_url': url_for('static', filename='uploads/' + thumb_name(d.uploaded_filename)),
            'result': f"{d.detect_score}%"
        }
        for d in detect_pagination.items
    ]

    modify_history = [
        {
            'id': img.id,
            'date': img.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'thumb_url': url_for('static', filename='results/' + thumb_name(img.protected_filename)),
            'strength': int(img.watermark_strength * 100),
            'download_url': url_for('static', filename='results/' + img.protected_filename)
        }
        for img in modify_pagination.items
    ]

    return render_template(
        'mypage.html',
        detect_history=detect_history,
        modify_history=modify_history,
        detect_pagination=detect_pagination,
        modify_pagination=modify_pagination
    )

# 탐지 기록 삭제 (POST + CSRF)
@app.post('/delete_detect/<int:detect_id>')
def delete_detect(detect_id):
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login'))

    rec = DetectResult.query.get_or_404(detect_id)
    if rec.user_id != session['user_id']:
        abort(403)

    try:
        # 파일 삭제
        for fname in [rec.uploaded_filename, thumb_name(rec.uploaded_filename)]:
            if fname:
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

        # DB 삭제
        db.session.delete(rec)
        db.session.commit()
        flash("탐지 기록이 삭제되었습니다.")
    except Exception as e:
        db.session.rollback()
        logging.exception("delete_detect 실패: %s", e)
        flash("삭제 중 오류가 발생했습니다. 다시 시도해주세요.")
    return redirect(url_for('mypage'))

# 변형 기록 삭제 (POST + CSRF)
@app.post('/delete_modify/<int:image_id>')
def delete_modify(image_id):
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login'))

    rec = ProtectedImage.query.get_or_404(image_id)
    if rec.user_id != session['user_id']:
        abort(403)

    try:
        # 파일 삭제
        for base in [rec.protected_filename, rec.original_filename]:
            if base:
                for fname in [base, thumb_name(base)]:
                    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)

        # DB 삭제
        db.session.delete(rec)
        db.session.commit()
        flash("이미지 변형 기록이 삭제되었습니다.")
    except Exception as e:
        db.session.rollback()
        logging.exception("delete_modify 실패: %s", e)
        flash("삭제 중 오류가 발생했습니다. 다시 시도해주세요.")

    return redirect(url_for('mypage'))

# 정보 페이지
@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
