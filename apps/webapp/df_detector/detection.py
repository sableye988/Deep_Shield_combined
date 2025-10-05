import io
import numpy as np
from PIL import Image, ImageChops

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _ela_score(pil_img: Image.Image, quality: int = 90) -> float:
    """
    ELA(Error Level Analysis) 간이 점수.
    이미지를 JPEG으로 재압축한 뒤 원본과 차이의 평균을 0~1로 정규화.
    값이 클수록 조작 흔적 가능성이 높다고 가정.
    """
    rgb = pil_img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    diff = ImageChops.difference(rgb, recompressed)
    arr = np.asarray(diff).astype(np.float32)
    return float(np.clip(arr.mean() / 255.0 * 2.0, 0.0, 1.0))


if TORCH_AVAILABLE:
    # (나중에 torch 설치하면 이 경로가 자동 활성화됨)
    model = resnet18(pretrained=True)  # 최신 torchvision이면 weights=... 권장 경고가 뜰 수 있음(동작엔 문제 없음)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


def deepfake_model_detect(img: Image.Image) -> float:
    if not TORCH_AVAILABLE:
        return _ela_score(img)  # ← 0.5 고정 대신 ELA 기반 점수 반환
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    prob = float(out.softmax(1)[0][0])  # ⚠️ resnet18은 1000클래스라 이 값은 임시 지표
    return prob
