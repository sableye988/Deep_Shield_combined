# wm_blind.py
# -------------------------------------------------------------
# Blind watermarking (no key/meta):
#  - 2-level (or 1-level) DWT on Y channel
#  - QIM(Quantization Index Modulation) into HL/LH/HH subbands
#  - Extract needs only params + watermark size (또는 원본 wm 이미지)
# -------------------------------------------------------------
from typing import Tuple, Sequence
import numpy as np
from PIL import Image
import pywt

# ---------- color & DWT helpers ----------
def rgb_to_y(image_rgb: Image.Image):
    ycbcr = image_rgb.convert("YCbCr")
    Y, Cb, Cr = ycbcr.split()
    return np.array(Y, dtype=np.float32), Cb, Cr

def y_to_rgb(Y: np.ndarray, Cb: Image.Image, Cr: Image.Image) -> Image.Image:
    Y8 = np.clip(Y, 0, 255).astype(np.uint8)
    return Image.merge("YCbCr", (Image.fromarray(Y8, "L"), Cb, Cr)).convert("RGB")

def dwt2_levels(Y: np.ndarray, wavelet="db2", level=2):
    return pywt.wavedec2(Y, wavelet=wavelet, level=level)

def idwt2_levels(coeffs, wavelet="db2"):
    return pywt.waverec2(coeffs, wavelet=wavelet)

def get_band(coeffs, level: int, band_name: str) -> np.ndarray:
    """
    coeffs = [cAn, (cHn,cVn,cDn), ..., (cH1,cV1,cD1)]
    band_name ∈ {"HL","LH","HH"} at given 'level'
    HL -> cH, LH -> cV, HH -> cD
    """
    assert band_name in ("HL", "LH", "HH")
    idx = 1 + (len(coeffs) - 1 - level)  # level=2 -> index 1
    cH, cV, cD = coeffs[idx]
    return {"HL": cH, "LH": cV, "HH": cD}[band_name]

def set_band(coeffs, level: int, band_name: str, new_arr: np.ndarray):
    idx = 1 + (len(coeffs) - 1 - level)
    cH, cV, cD = coeffs[idx]
    if band_name == "HL":   cH = new_arr
    elif band_name == "LH": cV = new_arr
    else:                   cD = new_arr
    coeffs[idx] = (cH, cV, cD)

# ---------- QIM core ----------
def qim_embed_vals(vals: np.ndarray, bits: np.ndarray, delta: float) -> np.ndarray:
    """
    Scalar QIM:
      place |vals| into bins centered at Δ/4 (bit=0) and 3Δ/4 (bit=1) within each Δ interval
    """
    out = vals.copy()
    sgn = np.sign(out); sgn[sgn == 0] = 1.0
    u = np.abs(out)

    q = np.floor(u / delta)
    c0 = delta * (q + 0.25)
    c1 = delta * (q + 0.75)
    choose = np.where(bits.astype(bool), c1, c0)
    u_new = choose
    return sgn * u_new

def qim_extract_bits(vals: np.ndarray, delta: float) -> np.ndarray:
    u = np.abs(vals)
    r = u - delta * np.floor(u / delta)          # remainder in [0, Δ)
    d0 = np.abs(r - 0.25 * delta)
    d1 = np.abs(r - 0.75 * delta)
    bits = (d1 < d0).astype(np.uint8)            # closer center wins
    return bits

# ---------- public APIs ----------
def embed_blind_y(
    host_img: Image.Image,
    wm_img: Image.Image,
    *,
    wavelet: str = "db2",
    level: int = 2,
    bands: Sequence[str] = ("HL", "LH"),
    delta: float = 4.0,
    repeat: int = 3,
) -> Image.Image:
    """
    Blind embed on Y channel with DWT-QIM.
    No meta/keys are stored. Recovery uses only params and wm size.
    """
    if repeat < 1:
        raise ValueError("repeat must be >=1")

    Y, Cb, Cr = rgb_to_y(host_img)
    coeffs = dwt2_levels(Y, wavelet, level)

    wm_arr = np.array(wm_img.convert("L"), dtype=np.float32)
    wm_bits = (wm_arr >= 128.0).astype(np.uint8).ravel()
    n_bits = wm_bits.size

    pools = [get_band(coeffs, level, b) for b in bands]
    capacity = sum(p.size for p in pools)
    need = n_bits * repeat
    if need > capacity:
        raise ValueError(f"Capacity {capacity} < needed {need}. Reduce wm size or repeat, or add bands.")

    # Flatten & embed first 'need' positions deterministically
    flats = [p.reshape(-1) for p in pools]
    flat_all = np.concatenate(flats, axis=0)

    idxs = np.arange(need, dtype=np.int64)
    seq_bits = np.repeat(wm_bits, repeat)
    flat_all[idxs] = qim_embed_vals(flat_all[idxs], seq_bits, delta)

    # Put back
    offset = 0
    for k, p in enumerate(pools):
        count = p.size
        slice_mod = flat_all[offset:offset+count].reshape(p.shape)
        set_band(coeffs, level, bands[k], slice_mod)
        offset += count

    Y_new = idwt2_levels(coeffs, wavelet)
    return y_to_rgb(Y_new, Cb, Cr)

def extract_blind_y(
    img_rgb: Image.Image,
    *,
    wm_shape: Tuple[int, int],
    wavelet: str = "db2",
    level: int = 2,
    bands: Sequence[str] = ("HL", "LH"),
    delta: float = 4.0,
    repeat: int = 3,
):
    """
    Blind extract: needs only (wm_h, wm_w) and the same params.
    Returns uint8 watermark (0/255).
    """
    if repeat < 1:
        raise ValueError("repeat must be >=1")

    H_wm, W_wm = wm_shape
    n_bits = H_wm * W_wm

    Y, _, _ = rgb_to_y(img_rgb)
    coeffs = dwt2_levels(Y, wavelet, level)
    pools = [get_band(coeffs, level, b) for b in bands]
    capacity = sum(p.size for p in pools)
    need = n_bits * repeat
    if need > capacity:
        raise ValueError(f"Capacity {capacity} < needed {need}.")

    flat_all = np.concatenate([p.reshape(-1) for p in pools], axis=0)
    idxs = np.arange(need, dtype=np.int64)
    seq_bits = qim_extract_bits(flat_all[idxs], delta)  # length 'need'

    # majority vote across 'repeat'
    votes = (seq_bits.reshape(-1, repeat).mean(axis=1) >= 0.5).astype(np.uint8)
    return (votes.reshape(H_wm, W_wm) * 255).astype(np.uint8)
