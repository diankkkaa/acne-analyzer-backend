from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.config import settings


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            base = models.resnet34(weights=None)
        except TypeError:
            base = models.resnet34(pretrained=False)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.proj(x)
        return F.normalize(x, dim=1)


_device: torch.device | None = None
_model: EmbeddingNet | None = None
_prototypes: torch.Tensor | None = None
_class_names: tuple[str, ...] = ()


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model() -> None:
    global _device, _model
    _device = _resolve_device()
    net = EmbeddingNet()
    path = Path(settings.MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"MODEL_PATH does not exist or is not a file: {path.resolve()}")
    
    try:
        checkpoint = torch.load(path, map_location=_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=_device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        net.load_state_dict(checkpoint["model_state"])
    else:
        net.load_state_dict(checkpoint)

    net.to(_device)
    net.eval()
    _model = net

def _ensure_model() -> tuple[EmbeddingNet, torch.device]:
    if _model is None or _device is None:
        raise RuntimeError("Model not loaded; call load_model() first.")
    return _model, _device


def _discover_class_folders(support_root: Path) -> list[Path]:
    folders = sorted(
        [p for p in support_root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )
    if not folders:
        raise FileNotFoundError(f"No class subfolders in {support_root.resolve()}")
    return folders


@torch.inference_mode()
def build_prototypes() -> None:
    global _prototypes, _class_names
    model, device = _ensure_model()
    support_root = Path(settings.SUPPORT_SET_PATH)
    if not support_root.is_dir():
        raise FileNotFoundError(f"SUPPORT_SET_PATH is not a directory: {support_root.resolve()}")

    names: list[str] = []
    protos: list[torch.Tensor] = []
    for folder in _discover_class_folders(support_root):
        files = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        if len(files) < 1:
            raise FileNotFoundError(f"No images found in {folder.resolve()}")

        embs: list[torch.Tensor] = []
        for img_path in files[:5]:
            with Image.open(img_path) as pil_img:
                img = pil_img.convert("RGB")
            tensor = val_transform(img).unsqueeze(0).to(device)
            emb = model(tensor)
            embs.append(emb.squeeze(0))

        stacked = torch.stack(embs, dim=0)
        proto = stacked.mean(dim=0)
        proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
        names.append(folder.name)
        protos.append(proto)

    _class_names = tuple(names)
    _prototypes = torch.stack(protos, dim=0).to(device)


def _ensure_prototypes() -> torch.Tensor:
    if _prototypes is None:
        raise RuntimeError("Prototypes not built; call build_prototypes() first.")
    return _prototypes


@torch.inference_mode()
def predict(image: Image.Image) -> dict[str, str | float]:
    model, device = _ensure_model()
    prototypes = _ensure_prototypes().to(device)
    if not _class_names:
        raise RuntimeError("Class names not set; call build_prototypes() first.")

    if image.mode != "RGB":
        image = image.convert("RGB")
    x = val_transform(image).unsqueeze(0).to(device)
    emb = model(x)

    sims = torch.mm(emb, prototypes.t()).squeeze(0)
    probs = F.softmax(sims, dim=0)
    idx = int(torch.argmax(sims).item())
    confidence = float(probs[idx].item())
    return {
        "predicted_class": _class_names[idx],
        "confidence": confidence,
    }


def initialize_for_inference() -> None:
    load_model()
    build_prototypes()
