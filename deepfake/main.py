# main.py
# -------------------------------------------------------------
# FastAPI for Blind Watermark (DWT-QIM)
# Endpoints:
#   - /health
#   - /embed_blind         (host + wm => PNG)
#   - /extract_blind       (img + wm_h + wm_w => PNG)
#   - /extract_blind_by_wm (img + wm => PNG + 헤더로 PSNR/NCC/BER 제공)
# -------------------------------------------------------------
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image
import io
import numpy as np

from wm_blind import embed_blind_y, extract_blind_y

app = FastAPI(title="Blind Watermark API (DWT-QIM)")

# ------------- metrics utils -------------
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

# ------------- health -------------
@app.get("/health")
def health():
    return {"ok": True}

# ------------- embed (blind) -------------
@app.post("/embed_blind")
async def embed_blind(
    host: UploadFile = File(..., description="원본 이미지 (JPG/PNG 등)"),
    wm:   UploadFile = File(..., description="워터마크 이미지 (흑백 권장)"),
    wavelet: str = Form("db2"),
    level:   int = Form(2),
    bands:   str = Form("HL,LH"),   # "HL,LH" or "HL,LH,HH"
    delta:  float = Form(4.0),
    repeat: int   = Form(3),
):
    try:
        host_im = Image.open(io.BytesIO(await host.read())).convert("RGB")
        wm_im   = Image.open(io.BytesIO(await wm.read())).convert("L")
        bands_tuple = tuple([b.strip() for b in bands.split(",") if b.strip()])

        marked = embed_blind_y(
            host_img=host_im, wm_img=wm_im,
            wavelet=wavelet, level=level, bands=bands_tuple,
            delta=delta, repeat=repeat
        )
        out = io.BytesIO()
        marked.save(out, format="PNG")
        return Response(
            content=out.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="face_marked_blind.png"'}
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

# ------------- extract (blind) with wm size -------------
@app.post("/extract_blind")
async def extract_blind(
    watermarked_img: UploadFile = File(..., description="블라인드 방식으로 워터마크 삽입된 이미지"),
    wm_h: int = Form(..., description="워터마크 높이"),
    wm_w: int = Form(..., description="워터마크 너비"),
    wavelet: str = Form("db2"),
    level:   int = Form(2),
    bands:   str = Form("HL,LH"),
    delta:  float = Form(4.0),
    repeat: int   = Form(3),
):
    try:
        im = Image.open(io.BytesIO(await watermarked_img.read())).convert("RGB")
        bands_tuple = tuple([b.strip() for b in bands.split(",") if b.strip()])

        wm_rec = extract_blind_y(
            img_rgb=im, wm_shape=(wm_h, wm_w),
            wavelet=wavelet, level=level, bands=bands_tuple,
            delta=delta, repeat=repeat
        )
        out = io.BytesIO()
        Image.fromarray(wm_rec, "L").save(out, format="PNG")
        return Response(
            content=out.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="wm_extracted_blind.png"'}
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

# ------------- extract (blind) by wm image + return metrics in headers -------------
@app.post("/extract_blind_by_wm")
async def extract_blind_by_wm(
    watermarked_img: UploadFile = File(..., description="블라인드 방식으로 워터마크 삽입된 이미지"),
    wm: UploadFile = File(..., description="원본 워터마크 이미지(크기 자동 인식, PSNR 비교용)"),
    wavelet: str = Form("db2"),
    level:   int = Form(2),
    bands:   str = Form("HL,LH"),
    delta:  float = Form(4.0),
    repeat: int   = Form(3),
):
    """
    - 추출 PNG를 파일로 반환.
    - 응답 헤더에 X-PSNR, X-NCC, X-BER(고정128) 제공.
    """
    try:
        im = Image.open(io.BytesIO(await watermarked_img.read())).convert("RGB")
        wm_im_gt = Image.open(io.BytesIO(await wm.read())).convert("L")
        wm_h, wm_w = wm_im_gt.height, wm_im_gt.width
        bands_tuple = tuple([b.strip() for b in bands.split(",") if b.strip()])

        # 추출 (uint8 0/255)
        wm_rec = extract_blind_y(
            img_rgb=im, wm_shape=(wm_h, wm_w),
            wavelet=wavelet, level=level, bands=bands_tuple,
            delta=delta, repeat=repeat
        )

        # 지표 계산
        gt_arr  = np.array(wm_im_gt, dtype=np.float32)
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
