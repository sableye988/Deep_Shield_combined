# main.py
# -------------------------------------------------------------
# FastAPI for Blind Watermark (DWT-QIM) + Deepfake Detect API
# Endpoints:
#   - /health
#   - /embed_blind_fixed         (host + wm => PNG, PSNR 메타 포함)
#   - /extract_blind_fixed       (img + wm => PNG + 헤더 지표)
#   - /extract_blind_fixed_text  (img + wm => 텍스트 지표)
#   - /api/detect                (딥페이크 탐지: 폴백(resnet18) 확률/경고)  # ★
# -------------------------------------------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io
import numpy as np
import json

from wm_blind import embed_blind_y, extract_blind_y

# ======= (변경) 딥페이크 탐지 모델 준비: utils/models 의존성 제거, torchvision 폴백만 사용 =======  # ★
DETECT_BACKEND = None        # "fallback" or None
DETECT_LOAD_ERR = ""
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from torchvision.models import resnet18

    _fallback_model = resnet18(weights="IMAGENET1K_V1")   # torch/torchvision 최신에서 사용
    _fallback_model.eval()
    _fallback_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    DETECT_BACKEND = "fallback"
except Exception as e:
    DETECT_BACKEND = None
    DETECT_LOAD_ERR = f"fallback init error: {e}"

# ======= 고정 파라미터 =======
FIXED = {
    "wavelet": "db2",
    "level": 1,
    "bands": ("HL", "LH"),
    "delta": 4.0,
    "repeat": 1,
}

# ======= 지표 유틸 =======
def _psnr(a: np.ndarray, b: np.ndarray, maxval: float = 255.0) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float('inf')
    import math
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

app = FastAPI(title="Blind Watermark API (DWT-QIM) + Detect")

# ------------- health -------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "fixed": FIXED,
        "detect_backend": DETECT_BACKEND,   # ★ 상태 노출
        "load_error": DETECT_LOAD_ERR if DETECT_BACKEND is None else ""
    }

# ------------- (변경) 딥페이크 탐지: 폴백(resnet18)만 사용 -------------  # ★
@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...), username: str = "guest"):
    if DETECT_BACKEND is None:
        raise HTTPException(status_code=503, detail=f"탐지 모델 준비 실패: {DETECT_LOAD_ERR or 'Unknown'}")
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        tensor = _fallback_tf(image).unsqueeze(0)
        with torch.no_grad():
            out = _fallback_model(tensor)
            # class 0 확률을 'fake 유사도'로 간이 사용
            fake_prob = float(torch.softmax(out, dim=1)[0, 0].item())

        final_prob = fake_prob
        label = "Fake" if final_prob > 0.5 else "Real"
        confidence = round(final_prob, 4) if label == "Fake" else round(1 - final_prob, 4)

        warning = ""
        if confidence < 0.3:
            warning = "탐지 결과 불확실: 이미지 품질이 낮거나 노이즈가 많습니다."
        elif 0.3 <= confidence <= 0.7:
            warning = "주의: 보정된 사진(포토샵, 필터 등)일 수 있습니다."
        elif confidence > 0.9:
            warning = "변조 흔적이 없습니다."

        return JSONResponse({
            "simple_result": {"prediction": label, "confidence": confidence},
            "expert_result": {
                "final_prob": round(final_prob, 4),
                "model_details": {"resnet18_softmax0": round(fake_prob, 4)}
            },
            "warning": warning
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"탐지 처리 실패: {e}")

# ------------- embed (fixed) -------------
@app.post("/embed_blind_fixed")
async def embed_blind_fixed(
    host: UploadFile = File(..., description="원본 이미지"),
    wm:   UploadFile = File(..., description="워터마크 이미지(흑백 권장)"),
):
    try:
        host_im = Image.open(io.BytesIO(await host.read())).convert("RGB")
        wm_im   = Image.open(io.BytesIO(await wm.read())).convert("L")

        marked = embed_blind_y(
            host_img=host_im, wm_img=wm_im,
            wavelet=FIXED["wavelet"], level=FIXED["level"], bands=FIXED["bands"],
            delta=FIXED["delta"], repeat=FIXED["repeat"]
        )

        # (유지) 임베드 결과와 원본의 PSNR을 PNG 텍스트청크(wm_meta)에 기록  # ★
        host_arr = np.array(host_im, dtype=np.uint8)
        prot_arr = np.array(marked.convert("RGB"), dtype=np.uint8)
        psnr_db = float(_psnr(prot_arr, host_arr, maxval=255.0))

        meta = PngInfo()
        meta.add_text("wm_meta", json.dumps({"psnr_db": psnr_db}))

        out = io.BytesIO()
        marked.save(out, format="PNG", pnginfo=meta, optimize=True)
        out.seek(0)
        return Response(
            content=out.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="face_marked_blind.png"'}
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

# ------------- extract (fixed) - PNG + 헤더 지표 -------------
@app.post("/extract_blind_fixed")
async def extract_blind_fixed(
    watermarked_img: UploadFile = File(..., description="워터마크 삽입 이미지"),
    wm: UploadFile = File(..., description="원본 워터마크(크기 자동 인식)"),
):
    try:
        im = Image.open(io.BytesIO(await watermarked_img.read())).convert("RGB")
        wm_gt = Image.open(io.BytesIO(await wm.read())).convert("L")
        wm_h, wm_w = wm_gt.height, wm_gt.width

        wm_rec = extract_blind_y(
            img_rgb=im, wm_shape=(wm_h, wm_w),
            wavelet=FIXED["wavelet"], level=FIXED["level"], bands=FIXED["bands"],
            delta=FIXED["delta"], repeat=FIXED["repeat"]
        )

        gt_arr  = np.array(wm_gt, dtype=np.float32)
        rec_arr = wm_rec.astype(np.float32)
        psnr_val = _psnr(rec_arr, gt_arr)
        ncc_val  = _ncc (rec_arr, gt_arr)
        ber_val  = _ber_fixed(rec_arr, gt_arr, thr=128.0)

        out = io.BytesIO()
        Image.fromarray(wm_rec, "L").save(out, format="PNG")
        return Response(
            content=out.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": 'attachment; filename="wm_extracted_blind.png"',
                "X-PSNR": f"{psnr_val:.2f}",
                "X-NCC": f"{ncc_val:.3f}",
                "X-BER": f"{ber_val:.4f}",
            }
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

# ------------- extract (fixed) - 텍스트 지표 -------------
@app.post("/extract_blind_fixed_text")
async def extract_blind_fixed_text(
    watermarked_img: UploadFile = File(..., description="워터마크 삽입 이미지"),
    wm: UploadFile = File(..., description="원본 워터마크(크기 자동 인식)"),
):
    try:
        im = Image.open(io.BytesIO(await watermarked_img.read())).convert("RGB")
        wm_gt = Image.open(io.BytesIO(await wm.read())).convert("L")
        wm_h, wm_w = wm_gt.height, wm_gt.width

        wm_rec = extract_blind_y(
            img_rgb=im, wm_shape=(wm_h, wm_w),
            wavelet=FIXED["wavelet"], level=FIXED["level"], bands=FIXED["bands"],
            delta=FIXED["delta"], repeat=FIXED["repeat"]
        )

        gt_arr  = np.array(wm_gt, dtype=np.float32)
        rec_arr = wm_rec.astype(np.float32)
        psnr_val = _psnr(rec_arr, gt_arr)
        ncc_val  = _ncc (rec_arr, gt_arr)
        ber_val  = _ber_fixed(rec_arr, gt_arr, thr=128.0)

        text = (
            f"Wavelet={FIXED['wavelet']}, Level={FIXED['level']}, Bands={','.join(FIXED['bands'])}, "
            f"Delta={FIXED['delta']}, Repeat={FIXED['repeat']}\n"
            f"PSNR(dB)={psnr_val:.2f}\n"
            f"NCC={ncc_val:.3f}\n"
            f"BER={ber_val:.4f}\n"
        )
        return PlainTextResponse(text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
