import io
import json
import math
from typing import Dict

import numpy as np
from PIL import Image, PngImagePlugin
import pywt

# =========================
# 유틸: 이미지 I/O
# =========================
def pil_to_array_gray(img: Image.Image) -> np.ndarray:
    """PIL → 0~255 float32 그레이스케일 ndarray"""
    return np.array(img.convert('L'), dtype=np.float32)

def psnr(reference: np.ndarray, test: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)

# =========================
# 2D DWT (레벨 N)
# =========================
def dwt2_levelN(image: np.ndarray, wavelet: str = 'db2', level: int = 2):
    """
    반환: [cA_n, (cH_n,cV_n,cD_n), ..., (cH_1,cV_1,cD_1)]
    pywt 표기: cH≈HL, cV≈LH, cD≈HH
    """
    return pywt.wavedec2(image, wavelet=wavelet, level=level)

def idwt2_levelN(coeffs, wavelet: str = 'db2') -> np.ndarray:
    return pywt.waverec2(coeffs, wavelet=wavelet)

def get_subband_by_name(coeffs, level: int, band: str):
    """
    band ∈ {'LL','HL','LH','HH'}
    coeffs: [cA_N, (cH_N,cV_N,cD_N), ..., (cH_1,cV_1,cD_1)]
    """
    N = len(coeffs) - 1
    if level < 1 or level > N:
        raise ValueError(f"유효하지 않은 level={level}. (1~{N})")

    if band.upper() == 'LL':
        if level != N:
            raise ValueError("LL은 최상위 근사(cA_N). level 없이 사용하세요.")
        return coeffs[0]

    idx = 1 + (N - level)
    cH, cV, cD = coeffs[idx]
    if band.upper() == 'HL':   # cH
        return cH
    elif band.upper() == 'LH': # cV
        return cV
    elif band.upper() == 'HH': # cD
        return cD
    raise ValueError("band는 {'LL','HL','LH','HH'} 중 하나여야 합니다.")

def set_subband_by_name(coeffs, level: int, band: str, new_band):
    N = len(coeffs) - 1
    if band.upper() == 'LL':
        if level != N:
            raise ValueError("LL 교체는 cA_N에 대해서만 허용합니다.")
        coeffs = list(coeffs)
        coeffs[0] = new_band
        return tuple(coeffs)

    idx = 1 + (N - level)
    cH, cV, cD = coeffs[idx]
    if band.upper() == 'HL':
        cH = new_band
    elif band.upper() == 'LH':
        cV = new_band
    elif band.upper() == 'HH':
        cD = new_band
    else:
        raise ValueError("band는 {'LL','HL','LH','HH'} 중 하나여야 합니다.")
    coeffs = list(coeffs)
    coeffs[idx] = (cH, cV, cD)
    return tuple(coeffs)

# =========================
# SVD 삽입/추출 (그레이스케일)
# =========================
def svd_embed(subband: np.ndarray, wm: np.ndarray, alpha: float):
    """
    subband: 타깃 서브밴드(2D), wm: 워터마크(2D) → subband 크기로 리사이즈 후
    S' = S + α·S_w 를 특이값에 삽입
    """
    sh = subband.shape
    wm_resized = np.array(
        Image.fromarray(wm.astype(np.float32)).resize((sh[1], sh[0]), Image.BILINEAR),
        dtype=np.float32
    )
    U_s, S_s, Vt_s = np.linalg.svd(subband, full_matrices=False)
    U_w, S_w, Vt_w = np.linalg.svd(wm_resized, full_matrices=False)

    L = min(len(S_s), len(S_w))
    S_s_new = S_s.copy()
    S_s_new[:L] = S_s[:L] + alpha * S_w[:L]

    subband_new = (U_s @ np.diag(S_s_new) @ Vt_s).astype(np.float32)

    key = {
        "S_sub_original": S_s.astype(np.float32),  # 반블라인드 추출용
        "U_w": U_w.astype(np.float32),
        "Vt_w": Vt_w.astype(np.float32),
        "wm_shape": list(wm_resized.shape),
        "alpha": float(alpha),
    }
    return subband_new, key

def svd_extract(subband_marked: np.ndarray, subband_original: np.ndarray, key: Dict) -> np.ndarray:
    alpha = float(key["alpha"])
    U_w = key["U_w"]
    Vt_w = key["Vt_w"]
    wm_shape = tuple(key["wm_shape"])

    _, S_m, _ = np.linalg.svd(subband_marked,   full_matrices=False)
    _, S_o, _ = np.linalg.svd(subband_original, full_matrices=False)
    L = min(len(S_m), len(S_o), U_w.shape[0], Vt_w.shape[0])
    S_w_est = (S_m[:L] - S_o[:L]) / (alpha + 1e-12)
    wm_est = (U_w[:, :L] @ np.diag(S_w_est) @ Vt_w[:L, :])
    wm_est = wm_est[:wm_shape[0], :wm_shape[1]]
    wm_est = np.clip(wm_est, 0, 255).astype(np.float32)
    return wm_est

def embed_pipeline(host_img: Image.Image, wm_img: Image.Image,
                   wavelet: str, level: int, band: str, alpha: float):
    host = pil_to_array_gray(host_img)
    wm   = pil_to_array_gray(wm_img)

    coeffs = dwt2_levelN(host, wavelet=wavelet, level=level)
    sb = get_subband_by_name(coeffs, level=level, band=band).astype(np.float32)

    sb_marked, key = svd_embed(sb, wm, alpha)
    coeffs_marked = set_subband_by_name(coeffs, level=level, band=band, new_band=sb_marked)
    watermarked = idwt2_levelN(coeffs_marked, wavelet=wavelet)

    meta = {
        "method": "svd",
        "wavelet": wavelet, "level": level, "band": band, "alpha": alpha,
        "key": {
            "S_sub_original": key["S_sub_original"],
            "U_w": key["U_w"],
            "Vt_w": key["Vt_w"],
            "wm_shape": key["wm_shape"],
            "alpha": key["alpha"],
        },
        "orig_subband": sb.astype(np.float32)  # 반블라인드: 원본 서브밴드 저장
    }
    return watermarked, meta, psnr(host, watermarked)

def extract_pipeline(watermarked_img: Image.Image, meta: Dict):
    assert meta.get("method") == "svd", "SVD 메타가 아닙니다."
    wavelet = meta["wavelet"]; level = meta["level"]; band = meta["band"]
    key     = meta["key"]

    wm_gray = pil_to_array_gray(watermarked_img)
    coeffs_m = dwt2_levelN(wm_gray, wavelet=wavelet, level=level)
    sb_m = get_subband_by_name(coeffs_m, level=level, band=band).astype(np.float32)

    sb_orig = meta["orig_subband"].astype(np.float32)
    return svd_extract(sb_m, sb_orig, key)

# =========================
# PNG 메타 유틸 (그레이스케일 버전)
# =========================
def array_to_png_bytes_with_meta(arr: np.ndarray, meta: dict) -> bytes:
    """0~255 ndarray → PNG bytes, PNG 텍스트 메타 'wm_meta'에 JSON 저장"""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode='L')
    pnginfo = PngImagePlugin.PngInfo()
    # numpy 배열(JSON 직렬화용 변환)
    cleaned_meta = {
        **meta,
        "key": {
            "S_sub_original": meta["key"]["S_sub_original"].tolist(),
            "U_w": meta["key"]["U_w"].tolist(),
            "Vt_w": meta["key"]["Vt_w"].tolist(),
            "wm_shape": meta["key"]["wm_shape"],
            "alpha": meta["key"]["alpha"],
        },
        "orig_subband": meta["orig_subband"].tolist()
    }
    pnginfo.add_text("wm_meta", json.dumps(cleaned_meta, ensure_ascii=False))
    buf = io.BytesIO()
    im.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()

def extract_meta_from_png(fileobj):
    """
    PNG의 wm_meta(JSON)를 dict로 복원하고, 같은 이미지의 PIL Image(그레이스케일)도 반환
    """
    im = Image.open(fileobj).convert('L')
    info = getattr(im, "info", {}) or {}
    wm_meta_str = info.get("wm_meta")
    if wm_meta_str is None:
        raise ValueError("PNG에 wm_meta가 없습니다. 해당 방식으로 생성된 파일인지 확인하세요.")
    meta = json.loads(wm_meta_str)
    # 리스트 → np.array 복원
    meta["key"]["S_sub_original"] = np.array(meta["key"]["S_sub_original"], dtype=np.float32)
    meta["key"]["U_w"] = np.array(meta["key"]["U_w"], dtype=np.float32)
    meta["key"]["Vt_w"] = np.array(meta["key"]["Vt_w"], dtype=np.float32)
    meta["orig_subband"] = np.array(meta["orig_subband"], dtype=np.float32)
    return meta, im

# =========================
# 컬러 유지(Y 채널 워터마킹) 버전
# =========================
def array_rgb_to_png_bytes_with_meta(arr_rgb: np.ndarray, meta: dict) -> bytes:
    """RGB ndarray → PNG bytes, wm_meta(JSON) 저장"""
    arr_rgb = np.clip(arr_rgb, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr_rgb, mode='RGB')
    pnginfo = PngImagePlugin.PngInfo()
    cleaned = {
        **meta,
        "key": {
            "S_sub_original": meta["key"]["S_sub_original"].tolist(),
            "U_w": meta["key"]["U_w"].tolist(),
            "Vt_w": meta["key"]["Vt_w"].tolist(),
            "wm_shape": meta["key"]["wm_shape"],
            "alpha": meta["key"]["alpha"],
        },
        "orig_subband": meta["orig_subband"].tolist()
    }
    pnginfo.add_text("wm_meta", json.dumps(cleaned, ensure_ascii=False))
    buf = io.BytesIO()
    im.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()

def extract_meta_from_png_rgb(fileobj):
    """컬러 PNG에서 wm_meta 복원 + PIL RGB 이미지 반환"""
    im = Image.open(fileobj).convert('RGB')
    info = getattr(im, "info", {}) or {}
    wm_meta_str = info.get("wm_meta")
    if wm_meta_str is None:
        raise ValueError("PNG에 wm_meta가 없습니다. 컬러 버전으로 생성된 파일인지 확인하세요.")
    meta = json.loads(wm_meta_str)
    meta["key"]["S_sub_original"] = np.array(meta["key"]["S_sub_original"], dtype=np.float32)
    meta["key"]["U_w"]           = np.array(meta["key"]["U_w"], dtype=np.float32)
    meta["key"]["Vt_w"]          = np.array(meta["key"]["Vt_w"], dtype=np.float32)
    meta["orig_subband"]         = np.array(meta["orig_subband"], dtype=np.float32)
    return meta, im

def embed_pipeline_y(host_img: Image.Image, wm_img: Image.Image,
                     wavelet='db2', level=2, band='HL', alpha=0.12):
    """RGB → YCbCr 분해, Y만 워터마킹 후 RGB로 복원"""
    rgb  = host_img.convert('RGB')
    Y, Cb, Cr = rgb.convert('YCbCr').split()
    Y_arr = np.array(Y, dtype=np.float32)

    wm_arr = np.array(wm_img.convert('L'), dtype=np.float32)

    coeffs = dwt2_levelN(Y_arr, wavelet=wavelet, level=level)
    sb = get_subband_by_name(coeffs, level=level, band=band).astype(np.float32)

    sb_marked, key = svd_embed(sb, wm_arr, alpha)
    coeffs_marked = set_subband_by_name(coeffs, level=level, band=band, new_band=sb_marked)
    Y_marked = idwt2_levelN(coeffs_marked, wavelet=wavelet)
    Y_marked = np.clip(Y_marked, 0, 255).astype(np.uint8)

    out_rgb = Image.merge('YCbCr', (Image.fromarray(Y_marked, 'L'), Cb, Cr)).convert('RGB')
    psnr_db = psnr(Y_arr, Y_marked.astype(np.float32))  # Y 채널 기준 PSNR

    meta = {
        "method": "svd_y",
        "wavelet": wavelet, "level": level, "band": band, "alpha": alpha,
        "key": {
            "S_sub_original": key["S_sub_original"],
            "U_w": key["U_w"],
            "Vt_w": key["Vt_w"],
            "wm_shape": key["wm_shape"],
            "alpha": key["alpha"],
        },
        "orig_subband": sb.astype(np.float32)
    }
    return np.array(out_rgb, dtype=np.uint8), meta, psnr_db

def extract_pipeline_y(watermarked_img: Image.Image, meta: dict):
    """컬러 PNG에서 Y 채널만 꺼내 동일 파이프라인으로 추출"""
    assert meta.get("method") == "svd_y", "컬러(Y채널) 메타가 아닙니다."
    wavelet = meta["wavelet"]; level = meta["level"]; band = meta["band"]
    key     = meta["key"]

    rgb = watermarked_img.convert('RGB')
    Y, _, _ = rgb.convert('YCbCr').split()
    Y_arr = np.array(Y, dtype=np.float32)

    coeffs_m = dwt2_levelN(Y_arr, wavelet=wavelet, level=level)
    sb_m = get_subband_by_name(coeffs_m, level=level, band=band).astype(np.float32)

    sb_orig = meta["orig_subband"].astype(np.float32)
    return svd_extract(sb_m, sb_orig, key)
