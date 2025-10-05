import os
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from PIL import Image

if TORCH_AVAILABLE:
    model = resnet18(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

def deepfake_model_detect(img: Image.Image) -> float:
    if not TORCH_AVAILABLE:
        return 0.5
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    prob = float(out.softmax(1)[0][0])
    return prob
