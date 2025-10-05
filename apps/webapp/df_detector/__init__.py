# apps/webapp/df_detector/__init__.py
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image

from .detection import deepfake_model_detect
from .metrics import compute_psnr_ssim
from .exif_utils import is_photoshop_like

# 기본값 + config.json 병합
_DEF = {"deepfake_threshold": 0.5, "psnr_threshold": 35.0, "ssim_threshold": 0.95}
cfg_path = Path(__file__).with_name("config.json")
if cfg_path.exists():
    try:
        _CFG = {**_DEF, **json.loads(cfg_path.read_text(encoding="utf-8"))}
    except Exception:
        _CFG = _DEF
else:
    _CFG = _DEF

def run_detection(host_img: Image.Image) -> Dict[str, Any]:
    """단일 이미지 딥페이크 추정 결과를 반환"""
    rgb = host_img.convert("RGB")

    # 모델 추론 (torch 미설치면 deepfake_model_detect가 0.5 반환)
    prob_fake = float(deepfake_model_detect(rgb))

    label = "Fake" if prob_fake > 0.5 else "Real"
    confidence = prob_fake if label == "Fake" else (1.0 - prob_fake)
    confidence = float(round(confidence, 4))

    # 경고 메시지 (DF-Detector 뉘앙스 유지)
    warning = ""
    if confidence < 0.3:
        warning = "탐지 결과 불확실: 이미지 품질이 낮거나 노이즈가 많습니다."
    elif 0.3 <= confidence <= 0.7:
        warning = "주의: 보정된 사진(포토샵, 필터 등)일 수 있습니다."
    elif confidence > 0.9:
        warning = "변조 흔적이 없습니다."

    # EXIF 기반 보정 가능성 플래그
    exif_photoshop = is_photoshop_like(rgb)

    return {
        "simple_result": {"prediction": label, "confidence": confidence},
        "expert_result": {
            "final_prob": float(round(prob_fake, 4)),
            "exif_photoshop_like": bool(exif_photoshop),
        },
        "warning": warning,
    }

def compute_quality_metrics(orig_arr: np.ndarray, edited_arr: np.ndarray) -> Tuple[float, float]:
    """(선택) PSNR/SSIM 계산용"""
    return compute_psnr_ssim(orig_arr, edited_arr)

