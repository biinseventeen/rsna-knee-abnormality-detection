from dataclasses import dataclass
from pathlib import Path

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

DINO_MODEL_ID = "facebook/dinov2-small"


@dataclass(frozen=True)
class InferenceConfig:
    image_size: int = 224
    slices_per_stack: int = 3
    feature_dim: int = 384
    device: str = "auto"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
