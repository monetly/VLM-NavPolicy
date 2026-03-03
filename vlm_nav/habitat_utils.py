"""Habitat environment config helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import NavExperimentConfig


def ensure_hm3d_scene_alias(scenes_dir: str) -> None:
    """Create hm3d_v0.2 -> hm3d symlink when datasets expect legacy naming."""
    root = Path(scenes_dir)
    target = root / "hm3d"
    alias = root / "hm3d_v0.2"
    if target.exists() and not alias.exists():
        alias.symlink_to(target, target_is_directory=True)


def _resolve_dataset_data_path(base_data_path: str, scene_id: Optional[str]) -> str:
    if not scene_id:
        return base_data_path

    base = Path(base_data_path)
    content_path = base.parent / "content" / f"{scene_id}.json.gz"
    if content_path.exists():
        return str(content_path)
    return base_data_path


def effective_turn_angle_deg(scan_angle_deg: float) -> int:
    """Habitat turn_angle is integer; round user value to nearest valid int."""
    value = int(round(float(scan_angle_deg)))
    return max(1, value)


def build_habitat_overrides(
    exp_cfg: NavExperimentConfig,
    split: Optional[str] = None,
    scene_id: Optional[str] = None,
    max_episode_steps: int = 500,
) -> List[str]:
    hcfg = exp_cfg.habitat
    scfg = exp_cfg.navigation
    turn_angle_deg = effective_turn_angle_deg(scfg.scan_angle_deg)

    dataset_data_path = _resolve_dataset_data_path(hcfg.dataset_data_path, scene_id)

    overrides = [
        f"habitat.dataset.split={split or hcfg.split}",
        f"habitat.dataset.data_path={dataset_data_path}",
        f"habitat.dataset.scenes_dir={hcfg.dataset_scenes_dir}",
        f"habitat.environment.max_episode_steps={max_episode_steps}",
        "habitat.task.end_on_success=False",
        f"habitat.simulator.scene_dataset={hcfg.scene_dataset}",
        f"habitat/simulator/sensor_setups@habitat.simulator.agents.main_agent=rgb_agent",
        f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width={hcfg.rgb_width}",
        f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height={hcfg.rgb_height}",
        f"habitat.simulator.agents.main_agent.height={scfg.camera_height_m}",
        f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position=[0,{scfg.camera_height_m},0]",
        f"habitat.simulator.forward_step_size={scfg.forward_step_m}",
        f"habitat.simulator.turn_angle={turn_angle_deg}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={hcfg.gpu_device_id}",
        "habitat.simulator.habitat_sim_v0.allow_sliding=True",
        "+habitat/task/measurements@habitat.task.measurements.collisions=collisions",
        "~habitat.simulator.agents.main_agent.sim_sensors.depth_sensor",
    ]
    return overrides


def collision_count(metrics) -> int:
    value = metrics.get("collisions")
    if isinstance(value, dict):
        return int(value.get("count", 0))
    if isinstance(value, (int, float)):
        return int(value)
    return 0
