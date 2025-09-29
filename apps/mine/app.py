from flask import Flask, render_template, request, redirect, url_for, session, flash, abort
from models import db, ProtectedImage, User, DetectResult
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_migrate import Migrate
from flask import send_file
from PIL import Image, ImageOps
import os
from shutil import copyfile
import logging
import requests
import json
import io
import numpy as np


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
# app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS 시 활성화

# 은성님 워터마크 API
MATE_API = "http://127.0.0.1:5002"

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
WATERMARK_REF_PATH = os.path.join(ASSETS_DIR, "hanshin.png")

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

def read_psnr_from_png(png_path: str):
    """PNG의 wm_meta에서 psnr_db 읽기 (없으면 None)"""
    try:
        im = Image.open(png_path)
        info = getattr(im, "info", {}) or {}
        meta_str = info.get("wm_meta")
        if not meta_str:
            return None
        meta = json.loads(meta_str)
        return meta.get("psnr_db")
    except Exception:
        return None

# DB 초기화
db.init_app(app)
migrate = Migrate(app, db)

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

# 탐지 (현재는 예시 점수 — 이후 워터마크 기반 판별 로직으로 교체 가능)
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

        # 예시 점수 (임시)
        import random
        detect_score = round(random.uniform(0, 100), 2)

        new_result = DetectResult(
            user_id=user_id,
            uploaded_filename=filename,
            detect_score=detect_score
        )
        db.session.add(new_result)
        db.session.commit()

        # PRG
        session['detect_result'] = {
            'uploaded_url': url_for('static', filename='uploads/' + filename),
            'uploaded_thumb_url': url_for('static', filename='uploads/' + detect_thumb),
            'score': detect_score
        }
        return redirect(url_for('detect'))

    result = session.pop('detect_result', None)
    return render_template('detect.html', result=result)

# 방지 (워터마크 삽입)
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

        # 워터마크 강도(표시용)
        strength = 0.5
        user_id = session['user_id']
        ensure_upload_dir()

        # 원본 저장
        original_filename = build_safe_timestamp_name('original', file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # 팀원 FastAPI 호출
        try:
            with open(original_path, "rb") as fp:
                r = requests.post(
                    f"{MATE_API}/embed_fixed_single_color",
                    files={"host": fp},
                    timeout=120
                )
            if r.status_code != 200:
                flash(f"워터마크 임베드 실패: {r.status_code} {r.text[:200]}")
                return redirect(url_for('prevent'))
        except Exception as e:
            app.logger.exception("FastAPI 호출 실패: %s", e)
            flash("내부 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            return redirect(url_for('prevent'))

        # 결과 PNG 저장
        base_noext, _ = os.path.splitext(original_filename)
        protected_filename = f"{base_noext}_protected.png"
        protected_path = os.path.join(app.config['RESULT_FOLDER'], protected_filename)
        with open(protected_path, "wb") as out:
            out.write(r.content)

        # (선택) PSNR 표시
        psnr_db = read_psnr_from_png(protected_path)
        if psnr_db is not None:
            flash(f"워터마킹 PSNR: {psnr_db:.2f} dB")

        # 썸네일
        original_thumb = thumb_name(original_filename)
        protected_thumb = thumb_name(protected_filename)
        save_thumbnail(original_path, os.path.join(app.config['UPLOAD_FOLDER'], original_thumb))
        save_thumbnail(protected_path, os.path.join(app.config['RESULT_FOLDER'], protected_thumb))

        # DB 기록
        new_record = ProtectedImage(
            user_id=user_id,
            original_filename=original_filename,      # 업로드/원본은 uploads 폴더
            protected_filename=protected_filename,    # 결과는 results 폴더
            watermark_strength=strength
        )
        db.session.add(new_record)
        db.session.commit()

        # PRG
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

    # 결과 PNG에서 PSNR 읽어 표시용으로 추가
    mods = []
    for img in modify_pagination.items:
        protected_path = os.path.join(app.config['RESULT_FOLDER'], img.protected_filename)
        psnr_db = read_psnr_from_png(protected_path)
        mods.append({
            'id': img.id,
            'date': img.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'thumb_url': url_for('static', filename='results/' + thumb_name(img.protected_filename)),
            'psnr': (f"{psnr_db:.2f}" if psnr_db is not None else None),
            'download_url': url_for('download_protected', image_id=img.id)  # ✅ 여기 수정
        })
    modify_history = mods

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
        # 원본은 uploads에서 삭제
        if rec.original_filename:
            for fname in [rec.original_filename, thumb_name(rec.original_filename)]:
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

        # 결과는 results에서 삭제
        if rec.protected_filename:
            for fname in [rec.protected_filename, thumb_name(rec.protected_filename)]:
                fpath = os.path.join(app.config['RESULT_FOLDER'], fname)
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


# 이미지 저장
@app.get('/download/<int:image_id>')
def download_protected(image_id):
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login'))
    rec = ProtectedImage.query.get_or_404(image_id)
    if rec.user_id != session['user_id']:
        abort(403)
    path = os.path.join(app.config['RESULT_FOLDER'], rec.protected_filename)
    if not os.path.exists(path):
        flash("파일을 찾을 수 없습니다.")
        return redirect(url_for('mypage'))
    return send_file(path, as_attachment=True, download_name=rec.protected_filename)


# 재검사
def has_our_wm_meta(png_path: str) -> bool:
    try:
        im = Image.open(png_path)
        info = getattr(im, "info", {}) or {}
        return "wm_meta" in (info or {})
    except Exception:
        return False

def extract_wm_via_api(png_path: str) -> np.ndarray | None:
    try:
        with open(png_path, "rb") as fp:
            r = requests.post(f"{MATE_API}/extract_fixed_color",
                              files={"watermarked_png": fp}, timeout=120)
        if r.status_code != 200:
            return None
        buf = io.BytesIO(r.content)
        return np.array(Image.open(buf).convert("L"), dtype=np.float32)
    except Exception:
        return None

def load_ref_wm_resized(target_shape) -> np.ndarray | None:
    try:
        im = Image.open(WATERMARK_REF_PATH).convert("L")
    except Exception:
        return None
    h, w = target_shape
    im = im.resize((w, h), Image.BILINEAR)
    return np.array(im, dtype=np.float32)

def ncc_similarity_percent(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    denom = (a.std() * b.std()) + 1e-12
    ncc = float((a * b).mean() / denom)
    return max(0.0, min(1.0, (ncc + 1.0) / 2.0)) * 100.0


#재검사 페이지
@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("업로드된 파일이 없습니다.")
            return redirect(url_for('verify'))

        file = request.files['image']
        if not file or file.filename == '':
            flash("파일을 선택해주세요.")
            return redirect(url_for('verify'))

        # PNG만 지원 (wm_meta가 PNG에 저장됨)
        is_png_ext = file.filename.lower().endswith('.png')
        is_png_mime = (file.mimetype or '').lower() == 'image/png'
        if not (is_png_ext and is_png_mime):
            flash("이 페이지는 DeepShield 서비스로 워터마킹된 PNG만 지원합니다. 방지 기능으로 저장한 PNG 파일을 업로드해주세요.")
            return redirect(url_for('verify'))

        ensure_upload_dir()
        filename = build_safe_timestamp_name('verify', file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        verdict = "워터마크 없음 → 딥페이크 의심"
        similarity = None

        if has_our_wm_meta(filepath):
            wm_est = extract_wm_via_api(filepath)
            if wm_est is not None:
                ref = load_ref_wm_resized(wm_est.shape)
                if ref is not None:
                    similarity = round(ncc_similarity_percent(wm_est, ref), 2)
                    if similarity >= 70:
                        verdict = "워터마크 정상 → 원본 가능성 높음"
                    elif similarity >= 40:
                        verdict = "워터마크 손상 → 조작 의심"
                    else:
                        verdict = "워터마크 불일치 → 딥페이크 의심"
                else:
                    verdict = "참조 워터마크 없음"
            else:
                verdict = "워터마크 추출 실패 → 딥페이크 의심"

        session['verify_result'] = {
            'uploaded_url': url_for('static', filename='uploads/' + filename),
            'similarity': similarity,
            'verdict': verdict
        }
        return redirect(url_for('verify'))

    result = session.pop('verify_result', None)
    return render_template('verify.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)
