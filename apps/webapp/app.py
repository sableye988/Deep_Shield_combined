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
from threading import Lock

import os
import logging
import json
import io
import time
import numpy as np

# ---------- 네트워크 세션 ----------
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
SESSION.headers.update({"Connection": "keep-alive"})
_adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504]),
)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

# ---------- Flask ----------
app = Flask(__name__)
app.logger.setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.INFO)

# ── 보안키 ──
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret')

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# ── 기본 설정 ──
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 업로드 용량 제한(20MB)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# 세션/쿠키
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# ── 외부 FastAPI ──
MATE_API_URL = os.getenv("MATE_API_URL", "https://deep-shield-combined-api.onrender.com")# ★ 딥페이크/워터마크 API 베이스

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
WATERMARK_REF_PATH = os.path.join(ASSETS_DIR, "hanshin.png")

# 워터마크 참조 이미지 메모리 캐싱
try:
    with open(WATERMARK_REF_PATH, "rb") as _f:
        WM_BYTES = _f.read()
    WM_REF_IM = Image.open(io.BytesIO(WM_BYTES)).convert("L")
    WM_H, WM_W = WM_REF_IM.height, WM_REF_IM.width
except Exception as _e:
    WM_BYTES = None
    WM_REF_IM = None
    WM_H = WM_W = None
    app.logger.exception("워터마크 참조 이미지 로드 실패: %s", _e)

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
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        im.thumbnail(max_size)
        im.save(dst_path, optimize=True, quality=85)
    except Exception:
        pass  # 썸네일 실패해도 서비스 계속

def read_psnr_from_png(png_path: str):
    """PNG의 info['wm_meta']에 저장된 PSNR(dB) 읽기(없을 수 있음)."""
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

# ── 로컬 지표 계산 유틸 ──
def _psnr(a: np.ndarray, b: np.ndarray, maxval: float = 255.0) -> float:
    import math
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float('inf')
    return 20.0 * math.log10(maxval) - 10.0 * math.log10(mse)

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def _ber_fixed(est: np.ndarray, gt: np.ndarray, thr: float = 128.0) -> float:
    estb = (est >= thr).astype(np.uint8)
    gtb  = (gt  >= thr).astype(np.uint8)
    H, W = gtb.shape
    estb = estb[:H, :W]
    return float(np.mean(estb != gtb))

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

@app.route('/info')
def info():
    return render_template('info.html')

@app.get('/login')
def login_redirect_to_google():
    return redirect(url_for('google_login'))

@app.get('/signup')
def signup_redirect_to_google():
    flash("구글 로그인만 지원합니다.")
    return redirect(url_for('google_login'))

@app.get("/auth/google/login")
def google_login():
    if not os.environ.get("GOOGLE_CLIENT_ID") or not os.environ.get("GOOGLE_CLIENT_SECRET"):
        flash("서버에 GOOGLE_CLIENT_ID/SECRET 환경변수가 설정되어 있지 않습니다.")
        return redirect(url_for('index'))
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.get("/auth/google/callback")
def google_callback():
    token = google.authorize_access_token()
    userinfo = token.get("userinfo") or {}
    sub = userinfo.get("sub")
    if not sub:
        flash("Google 사용자 식별자(sub)를 가져오지 못했습니다.")
        return redirect(url_for('index'))
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

@app.route('/logout')
def logout():
    session.clear()
    flash("로그아웃 되었습니다.")
    return redirect(url_for('index'))

# ── 탐지 ──
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

        detect_thumb = thumb_name(filename)
        save_thumbnail(filepath, os.path.join(app.config['UPLOAD_FOLDER'], detect_thumb))

        # (변경) 외부 FastAPI /api/detect 호출로 통일  # ★
        try:
            with open(filepath, "rb") as fp:
                files = {"file": ("image.png", fp, "image/png")}
                data = {"username": session.get('username', 'guest')}
                t0 = time.perf_counter()
                r = SESSION.post(f"{MATE_API}/api/detect", files=files, data=data, timeout=(10, 120))
                app.logger.info("/api/detect: %.3fs %s %s",
                                time.perf_counter()-t0, r.status_code, r.headers.get("content-type"))
            if r.status_code != 200:
                app.logger.error("detect api failed: /api/detect %s %s", r.status_code, r.text[:200])
                raise RuntimeError(f"/api/detect {r.status_code} {r.text[:200]}")
            res = r.json()
            prob_fake = float(res.get('expert_result', {}).get('final_prob', 0.5))
            detect_score = round(prob_fake * 100.0, 2)
            verdict = res.get('warning') or res.get('simple_result', {}).get('prediction')
        except Exception as e:
            app.logger.exception("탐지 호출 실패: %s", e)
            flash("탐지 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            return redirect(url_for('detect'))

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
            'score': detect_score,
            'verdict': verdict
        }
        return redirect(url_for('detect'))

    result = session.pop('detect_result', None)
    return render_template('detect.html', result=result)

# ── 방지: 워터마크 삽입 ──
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

        user_id = session['user_id']
        ensure_upload_dir()

        # 원본 저장
        original_filename = build_safe_timestamp_name('original', file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # ---------- 외부 FastAPI 호출 ----------
        try:
            if WM_BYTES is None:
                flash("내부 워터마크 참조 이미지를 불러오지 못했습니다.")
                return redirect(url_for('prevent'))

            with open(original_path, "rb") as host_fp:
                files = {
                    "host": ("host.png", host_fp, "image/png"),
                    "wm":   ("wm.png",   io.BytesIO(WM_BYTES), "image/png"),
                }
                t0 = time.perf_counter()
                r = SESSION.post(f"{MATE_API}/embed_blind_fixed", files=files, timeout=(10, 180))
                app.logger.info("/embed_blind_fixed: %.3fs %s %s",
                                time.perf_counter()-t0, r.status_code, r.headers.get("content-type"))

            if r.status_code != 200 or "image/png" not in (r.headers.get("content-type","").lower()):
                app.logger.warning("embed failed: %s %s | %s", r.status_code, r.headers.get("content-type"), (r.text or "")[:200])
                flash(f"워터마크 임베드 실패: {r.status_code} {r.text[:200]}")
                return redirect(url_for('prevent'))
        except Exception as e:
            app.logger.exception("embed_blind_fixed 호출 실패: %s", e)
            flash("내부 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            return redirect(url_for('prevent'))

        # 결과 PNG 저장
        base_noext, _ = os.path.splitext(original_filename)
        protected_filename = f"{base_noext}_protected.png"
        protected_path = os.path.join(app.config['RESULT_FOLDER'], protected_filename)
        with open(protected_path, "wb") as out:
            out.write(r.content)

        # PSNR
        psnr_db = read_psnr_from_png(protected_path)
        if psnr_db is not None:
            flash(f"워터마킹 PSNR: {psnr_db:.2f} dB")

        # 썸네일
        original_thumb = thumb_name(original_filename)
        protected_thumb = thumb_name(protected_filename)
        save_thumbnail(original_path, os.path.join(app.config['UPLOAD_FOLDER'], original_thumb))
        save_thumbnail(protected_path, os.path.join(app.config['RESULT_FOLDER'], protected_thumb))

        # DB
        new_record = ProtectedImage(
            user_id=user_id,
            original_filename=original_filename,
            protected_filename=protected_filename,
            watermark_strength=0.5
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

# ---------- 추출 호출 유틸 ----------
def try_extract_wm_and_metrics_via_api(png_path: str):
    if WM_BYTES is None:
        return None, None, "참조 워터마크 없음"
    try:
        with open(png_path, "rb") as img_fp:
            files = {
                "watermarked_img": ("image.png", img_fp, "image/png"),
                "wm":              ("wm.png",    io.BytesIO(WM_BYTES), "image/png"),
            }
            r = SESSION.post(f"{MATE_API}/extract_blind_fixed", files=files, timeout=(10, 180))
        app.logger.info("extract_blind_fixed: %s %s", r.status_code, r.headers.get("content-type"))
        if r.status_code == 200 and "image/png" in (r.headers.get("content-type","").lower()):
            return r.content, r.headers, None
        return None, None, f"/extract_blind_fixed -> {r.status_code} {(r.text or '')[:200]}"
    except Exception as e:
        return None, None, f"/extract_blind_fixed 예외: {e}"

# ── 재검사 페이지 ──
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

        # PNG 권장 알림
        is_png_ext = file.filename.lower().endswith('.png')
        is_png_mime = (file.mimetype or '').lower() == 'image/png'
        if not (is_png_ext and is_png_mime):
            flash("이 페이지는 방지 기능으로 생성한 PNG 파일을 권장합니다.")

        ensure_upload_dir()
        filename = build_safe_timestamp_name('verify', file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        verdict = "워터마크 없음 → 딥페이크 의심"
        similarity = None
        wm_preview_url = None
        psnr = ncc = ber = None

        t0 = time.perf_counter()
        extracted_png, hdrs, fail_reason = try_extract_wm_and_metrics_via_api(filepath)
        app.logger.info("verify total: %.3fs", time.perf_counter()-t0)

        if extracted_png is not None:
            # 추출 결과 저장
            base_noext, _ = os.path.splitext(os.path.basename(filepath))
            wm_preview_name = f"{base_noext}_wm_extracted.png"
            wm_preview_path = os.path.join(app.config['RESULT_FOLDER'], wm_preview_name)
            with open(wm_preview_path, "wb") as out:
                out.write(extracted_png)
            wm_preview_url = url_for('static', filename=f"results/{wm_preview_name}")

            if hdrs:
                try: psnr = float(hdrs.get("X-PSNR"))
                except: pass
                try: ncc  = float(hdrs.get("X-NCC"))
                except: pass
                try: ber  = float(hdrs.get("X-BER"))
                except: pass

            if (psnr is None or ncc is None or ber is None) and WM_REF_IM is not None:
                try:
                    rec_im = Image.open(wm_preview_path).convert("L")
                    gt_im  = WM_REF_IM
                    rec = np.array(rec_im, dtype=np.float32)
                    gt  = np.array(gt_im,  dtype=np.float32)
                    psnr = _psnr(rec, gt)
                    ncc  = _ncc(rec, gt)
                    ber  = _ber_fixed(rec, gt, 128.0)
                    app.logger.info("local metrics: PSNR=%.2f NCC=%.3f BER=%.4f", psnr, ncc, ber)
                except Exception as e:
                    app.logger.warning("local metric calc fail: %s", e)

            if ncc is not None:
                similarity = round(max(0.0, min(1.0, (ncc + 1.0) / 2.0)) * 100.0, 2)

            # 판정
            if similarity is not None:
                if similarity >= 70:
                    verdict = "워터마크 정상 → 원본 가능성 높음"
                elif similarity >= 40:
                    verdict = "워터마크 손상 → 조작 의심"
                else:
                    verdict = "워터마크 불일치 → 딥페이크 의심"
            else:
                verdict = "유사도 계산 불가 → 딥페이크 의심"
        else:
            if fail_reason:
                flash(f"워터마크 추출 실패 원인: {fail_reason}")
            verdict = "워터마크 추출 실패 → 딥페이크 의심"

        session['verify_result'] = {
            'uploaded_url': url_for('static', filename='uploads/' + filename),
            'similarity': similarity,
            'verdict': verdict,
            'wm_preview_url': wm_preview_url,
            'psnr': (None if psnr is None else f"{psnr:.2f}"),
            'ncc':  (None if ncc  is None else f"{ncc:.3f}"),
            'ber':  (None if ber  is None else f"{ber:.4f}"),
        }
        return redirect(url_for('verify'))

    result = session.pop('verify_result', None)
    return render_template('verify.html', result=result)

# ── 마이페이지 ──
@app.route('/mypage', endpoint='mypage')
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

# ── 삭제/다운로드 ──
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
        app.logger.exception("delete_detect 실패: %s", e)
        flash("삭제 중 오류가 발생했습니다. 다시 시도해주세요.")
    return redirect(url_for('mypage'))

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
        app.logger.exception("delete_modify 실패: %s", e)
        flash("삭제 중 오류가 발생했습니다. 다시 시도해주세요.")
    return redirect(url_for('mypage'))

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

# ── API 워밍업 ──
_warmup_lock = Lock()
_warmed_up = False

@app.before_request
def warm_up_mate_api_once():
    global _warmed_up
    if _warmed_up:
        return
    with _warmup_lock:
        if _warmed_up:
            return
        try:
            r = SESSION.get(f"{MATE_API}/health", timeout=(2, 5))
            app.logger.info("MATE_API health: %s %s", getattr(r, "status_code", None), getattr(r, "text", "")[:80])
        except Exception as e:
            app.logger.info("MATE_API warmup failed (ignored): %s", e)
        _warmed_up = True

if __name__ == '__main__':
    app.run(debug=True)
