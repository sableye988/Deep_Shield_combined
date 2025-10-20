from flask import Flask, render_template, request, redirect, url_for, session, flash, abort, send_file
from models import db, ProtectedImage, User, DetectResult
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_migrate import Migrate
from PIL import Image, ImageOps
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import logging
import requests
import json
import io
import numpy as np

app = Flask(__name__)

# ── 환경변수 기반 보안키(배포에서 필수) ──
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret')

# 프록시 뒤 HTTPS 인식(렌더 배포 시 권장)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# ── 기본 설정 ──
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB

# 세션/쿠키
app.config['SESSION_COOKIE_HTTPONLY'] = True
# 프런트/백엔드 같은 도메인이면 Lax로도 충분
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# 프런트가 다른 도메인(예: GitHub Pages/Netlify)에서 열고,
# 백엔드가 onrender.com인 "교차 도메인"이라면 아래 두 줄 사용:
# app.config['SESSION_COOKIE_SAMESITE'] = 'None'
# app.config['SESSION_COOKIE_SECURE'] = True

# ── 외부 API (워터마크 FastAPI) ──
MATE_API = "https://deep-shield-combined-api.onrender.com"

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
WATERMARK_REF_PATH = os.path.join(ASSETS_DIR, "hanshin.png")

# ── OAuth(구글) ──
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# ── CSRF ──
csrf = CSRFProtect(app)

@app.context_processor
def inject_csrf():
    return dict(csrf_token=generate_csrf)

# ── 파일 유틸 ──
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

# ── DB 초기화 ──
db.init_app(app)
migrate = Migrate(app, db)

# ── 에러 핸들러 ──
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash("파일이 너무 큽니다. 최대 20MB까지 업로드할 수 있어요.")
    return redirect(request.referrer or url_for('index'))

# ── 라우트 ──
@app.route('/')
def index():
    return render_template('index.html')

# 과거 경로 호환: /login, /signup 접근 시 구글 로그인으로 보냄
@app.get('/login')
def login_redirect_to_google():
    return redirect(url_for('google_login'))

@app.get('/signup')
def signup_redirect_to_google():
    flash("구글 로그인만 지원합니다.")
    return redirect(url_for('google_login'))

# 구글 로그인
@app.get("/auth/google/login")
def google_login():
    # 환경변수 미설정 시 빠른 안내
    if not os.environ.get("GOOGLE_CLIENT_ID") or not os.environ.get("GOOGLE_CLIENT_SECRET"):
        flash("서버에 GOOGLE_CLIENT_ID/SECRET 환경변수가 설정되어 있지 않습니다.")
        return redirect(url_for('index'))

    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

# 구글 콜백
@app.get("/auth/google/callback")
def google_callback():
    token = google.authorize_access_token()
    userinfo = token.get("userinfo") or {}
    sub = userinfo.get("sub")
    if not sub:
        flash("Google 사용자 식별자(sub)를 가져오지 못했습니다.")
        return redirect(url_for('index'))

    # provider + provider_id 로 upsert
    user = User.query.filter_by(provider='google', provider_id=sub).first()

    if not user:
        user = User(
            provider='google',
            provider_id=sub,
            email=userinfo.get("email"),
            name=userinfo.get("name"),
            picture=userinfo.get("picture"),
        )
        db.session.add(user)
        db.session.commit()

    session['user_id'] = user.id
    session['username'] = user.name or user.email or user.username or 'user'
    flash("구글 계정으로 로그인되었습니다.")
    return redirect(url_for('mypage'))

# 로그아웃
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
            return redirect(url_for('login_redirect_to_google'))

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

        # (임시) 탐지 점수: 실제 모델 붙이면 교체
        import random
        detect_score = round(random.uniform(0, 100), 2)

        new_result = DetectResult(
            user_id=user_id,
            uploaded_filename=filename,
            detect_score=detect_score
        )
        db.session.add(new_result)
        db.session.commit()

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
            return redirect(url_for('login_redirect_to_google'))

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
            original_filename=original_filename,
            protected_filename=protected_filename,
            watermark_strength=strength
        )
        db.session.add(new_record)
        db.session.commit()

        session['prevent_result'] = {
            "original_url": url_for('static', filename='uploads/' + original_filename),
            "modified_url": url_for('static', filename='results/' + protected_filename),
            "original_thumb_url": url_for('static', filename='uploads/' + original_thumb),
            "modified_thumb_url": url_for('static', filename='results/' + protected_thumb),
        }
        return redirect(url_for('prevent'))

    result = session.pop('prevent_result', None)
    return render_template('prevent.html', result=result)

# 마이페이지
@app.route('/mypage')
def mypage():
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login_redirect_to_google'))

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

    mods = []
    for img in modify_pagination.items:
        protected_path = os.path.join(app.config['RESULT_FOLDER'], img.protected_filename)
        psnr_db = read_psnr_from_png(protected_path)
        mods.append({
            'id': img.id,
            'date': img.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'thumb_url': url_for('static', filename='results/' + thumb_name(img.protected_filename)),
            'psnr': (f"{psnr_db:.2f}" if psnr_db is not None else None),
            'download_url': url_for('download_protected', image_id=img.id)
        })
    modify_history = mods

    return render_template(
        'mypage.html',
        detect_history=detect_history,
        modify_history=modify_history,
        detect_pagination=detect_pagination,
        modify_pagination=modify_pagination
    )

# 탐지 기록 삭제
@app.post('/delete_detect/<int:detect_id>')
def delete_detect(detect_id):
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login_redirect_to_google'))

    rec = DetectResult.query.get_or_404(detect_id)
    if rec.user_id != session['user_id']:
        abort(403)

    try:
        for fname in [rec.uploaded_filename, thumb_name(rec.uploaded_filename)]:
            if fname:
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

        db.session.delete(rec)
        db.session.commit()
        flash("탐지 기록이 삭제되었습니다.")
    except Exception as e:
        db.session.rollback()
        logging.exception("delete_detect 실패: %s", e)
        flash("삭제 중 오류가 발생했습니다. 다시 시도해주세요.")
    return redirect(url_for('mypage'))

# 변형 기록 삭제
@app.post('/delete_modify/<int:image_id>')
def delete_modify(image_id):
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login_redirect_to_google'))

    rec = ProtectedImage.query.get_or_404(image_id)
    if rec.user_id != session['user_id']:
        abort(403)

    try:
        if rec.original_filename:
            for fname in [rec.original_filename, thumb_name(rec.original_filename)]:
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

        if rec.protected_filename:
            for fname in [rec.protected_filename, thumb_name(rec.protected_filename)]:
                fpath = os.path.join(app.config['RESULT_FOLDER'], fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

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

# 결과 PNG 다운로드
@app.route('/download/<int:image_id>', methods=['GET'])
def download_protected(image_id):
    if 'user_id' not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for('login_redirect_to_google'))
    rec = ProtectedImage.query.get_or_404(image_id)
    if rec.user_id != session['user_id']:
        abort(403)
    path = os.path.join(app.config['RESULT_FOLDER'], rec.protected_filename)
    if not os.path.exists(path):
        flash("파일을 찾을 수 없습니다.")
        return redirect(url_for('mypage'))
    return send_file(path, as_attachment=True, download_name=rec.protected_filename)

# ── 재검사 유틸 ──
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

# 재검사
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
