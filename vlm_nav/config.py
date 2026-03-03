"""Configuration dataclasses and JSON IO for the mainline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class VLMConfig:
    """Settings for communicating with the Vision Language Model server."""
    base_url: str = "http://127.0.0.1:8080"
    model: str = "qwen3-vl"
    seed: Optional[int] = None
    request_timeout_sec: float = 60.0
    n_probs: int = 64
    temperature: float = -1.0
    top_k: int = 0
    top_p: float = 1.0
    online_action_samples: int = 1
    use_mock_when_unreachable: bool = True


@dataclass
class NavigationConfig:
    """Physical movement parameters and decision-making thresholds."""
    forward_step_m: float = 0.20
    scan_angle_deg: float = 30.0
    camera_height_m: float = 0.6
    # Two-stage safety & entropy thresholds.
    entropy_threshold: float = 2.1
    confidence_threshold: float = 0.35
    max_consecutive_fallbacks: int = 4


@dataclass
class HabitatConfig:
    """Habitat simulator environment and dataset settings."""
    config_path: str = "benchmark/nav/objectnav/objectnav_hm3d.yaml"
    dataset_data_path: str = (
        "/home/liuxh/vln/entopo/habitat-lab/data/datasets/objectnav/hm3d/v2/"
        "objectnav_hm3d_v2/val_mini/val_mini.json.gz"
    )
    dataset_scenes_dir: str = "/home/liuxh/vln/entopo/habitat-sim/data/versioned_data/hm3d-0.2"
    scene_dataset: str = (
        "/home/liuxh/vln/entopo/habitat-sim/data/versioned_data/hm3d-0.2/hm3d/"
        "hm3d_annotated_basis.scene_dataset_config.json"
    )
    split: str = "val"
    rgb_width: int = 512
    rgb_height: int = 512
    gpu_device_id: int = 0


@dataclass
class NavExperimentConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    habitat: HabitatConfig = field(default_factory=HabitatConfig)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _filter_dataclass_kwargs(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in payload.items() if k in names}


def default_config() -> NavExperimentConfig:
    return NavExperimentConfig()


def load_config(config_path: str) -> NavExperimentConfig:
    cfg = default_config()
    path = Path(config_path)
    if not path.exists():
        return cfg

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    merged = _deep_update(asdict(cfg), payload)
    return NavExperimentConfig(
        vlm=VLMConfig(**_filter_dataclass_kwargs(VLMConfig, merged["vlm"])),
        navigation=NavigationConfig(**_filter_dataclass_kwargs(NavigationConfig, merged.get("navigation", merged.get("stage2", {})))),
        habitat=HabitatConfig(**_filter_dataclass_kwargs(HabitatConfig, merged["habitat"])),
    )


def save_config(cfg: NavExperimentConfig, config_path: str) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=True)
        f.write("\n")
