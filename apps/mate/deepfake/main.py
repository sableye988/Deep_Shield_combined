import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from PIL import Image

from wm_core import (
    # 그레이스케일
    embed_pipeline, extract_pipeline,
    array_to_png_bytes_with_meta, extract_meta_from_png,
    # 컬러(Y 채널 유지)
    embed_pipeline_y, extract_pipeline_y,
    array_rgb_to_png_bytes_with_meta, extract_meta_from_png_rgb
)

app = FastAPI(title="DWT-SVD Invisible Watermark API (Fixed Params + Color Y-channel)", version="1.2.0")

# 고정 파라미터
WAVELET_DEFAULT = "db2"
LEVEL_DEFAULT   = 2
BAND_DEFAULT    = "HL"
ALPHA_DEFAULT   = 0.12

# 서버에 두는 기본 워터마크 파일 경로 (클라이언트는 host만 업로드)
WATERMARK_DEFAULT_PATH = Path("hanshin.png")

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# [추천] 컬러 유지(Y 채널) 고정 파라미터
# ---------------------------
@app.post("/embed_fixed_single_color")
async def embed_fixed_single_color(
    host: UploadFile = File(..., description="원본 이미지만 업로드 (워터마크는 서버의 hanshin.png 사용)")
):
    if not WATERMARK_DEFAULT_PATH.exists():
        return JSONResponse(status_code=500, content={
            "detail": f"기본 워터마크 파일을 찾을 수 없습니다: {WATERMARK_DEFAULT_PATH.resolve()}"
        })

    host_img = Image.open(host.file)
    wm_img   = Image.open(str(WATERMARK_DEFAULT_PATH))

    out_rgb, meta, psnr_db = embed_pipeline_y(
        host_img=host_img, wm_img=wm_img,
        wavelet=WAVELET_DEFAULT, level=LEVEL_DEFAULT,
        band=BAND_DEFAULT, alpha=ALPHA_DEFAULT
    )
    meta_with_psnr = {
        **meta, "psnr_db": psnr_db, "fixed_params": {
            "wavelet": WAVELET_DEFAULT, "level": LEVEL_DEFAULT,
            "band": BAND_DEFAULT, "alpha": ALPHA_DEFAULT
        }, "wm_src": str(WATERMARK_DEFAULT_PATH)
    }

    png_bytes = array_rgb_to_png_bytes_with_meta(out_rgb, meta_with_psnr)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="face_marked.png"'}
    )

@app.post("/extract_fixed_color")
async def extract_fixed_color(
    watermarked_png: UploadFile = File(..., description="워터마크 삽입된 컬러 PNG(내부 wm_meta 포함)")
):
    data = await watermarked_png.read()
    bio  = io.BytesIO(data)
    meta, im_rgb = extract_meta_from_png_rgb(bio)

    wm_est_arr = extract_pipeline_y(watermarked_img=im_rgb, meta=meta)
    out = io.BytesIO()
    Image.fromarray(wm_est_arr.astype('uint8'), mode='L').save(out, format="PNG")
    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="wm_extracted.png"'}
    )

# ---------------------------
# (옵션) 그레이스케일 고정 파라미터 (원래 버전 호환)
# ---------------------------
@app.post("/embed_fixed_single")
async def embed_fixed_single(
    host: UploadFile = File(..., description="원본 이미지만 업로드 (워터마크는 서버의 hanshin.png 사용)")
):
    if not WATERMARK_DEFAULT_PATH.exists():
        return JSONResponse(
            status_code=500,
            content={"detail": f"기본 워터마크 파일을 찾을 수 없습니다: {WATERMARK_DEFAULT_PATH.resolve()}"}
        )

    host_img = Image.open(host.file)
    wm_img   = Image.open(str(WATERMARK_DEFAULT_PATH))

    watermarked_arr, meta, psnr_db = embed_pipeline(
        host_img=host_img,
        wm_img=wm_img,
        wavelet=WAVELET_DEFAULT,
        level=LEVEL_DEFAULT,
        band=BAND_DEFAULT,
        alpha=ALPHA_DEFAULT
    )
    meta_with_psnr = {
        **meta,
        "psnr_db": psnr_db,
        "fixed_params": {
            "wavelet": WAVELET_DEFAULT,
            "level": LEVEL_DEFAULT,
            "band": BAND_DEFAULT,
            "alpha": ALPHA_DEFAULT,
        },
        "wm_src": str(WATERMARK_DEFAULT_PATH)
    }
    png_bytes = array_to_png_bytes_with_meta(watermarked_arr, meta_with_psnr)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="face_marked.png"'}
    )

@app.post("/extract_fixed")
async def extract_fixed(
    watermarked_png: UploadFile = File(..., description="워터마크 삽입 PNG(내부에 wm_meta 포함)")
):
    data = await watermarked_png.read()
    bio  = io.BytesIO(data)
    meta, im = extract_meta_from_png(bio)
    wm_est_arr = extract_pipeline(watermarked_img=im, meta=meta)

    out = io.BytesIO()
    Image.fromarray(wm_est_arr.astype('uint8'), mode='L').save(out, format="PNG")
    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="wm_extracted.png"'}
    )
