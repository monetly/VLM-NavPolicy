#!/usr/bin/env python3
"""Probe 1..6 macro-action outcomes from one open Habitat starting viewpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
from habitat.config.default import get_config
import habitat
from habitat_sim.utils import common as hs_common

from vlm_nav.actions import NUMBERED_MACRO_ACTIONS, macro_option_to_habitat
from vlm_nav.config import load_config
from vlm_nav.habitat_utils import build_habitat_overrides, collision_count, ensure_hm3d_scene_alias


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _yaw_deg_from_rotation(rot) -> float:
    fwd = hs_common.quat_rotate_vector(rot, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    return _wrap_deg(np.degrees(np.arctan2(float(fwd[0]), float(-fwd[2]))))


def _quat_from_yaw_deg(yaw_deg: float):
    return hs_common.quat_from_angle_axis(
        np.deg2rad(float(yaw_deg)),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


def _set_agent_state(sim, position: np.ndarray, rotation) -> None:
    try:
        sim.set_agent_state(position, rotation, reset_sensors=True)
    except TypeError:
        sim.set_agent_state(position, rotation)


def _create_env_with_gpu_fallback(exp_cfg, max_episode_steps: int):
    tried = []
    seen = set()
    candidates = [int(exp_cfg.habitat.gpu_device_id), 1, 0]
    last_exc: Optional[Exception] = None
    for gid in candidates:
        if gid in seen:
            continue
        seen.add(gid)
        tried.append(gid)
        cfg = copy.deepcopy(exp_cfg)
        cfg.habitat.gpu_device_id = int(gid)
        overrides = build_habitat_overrides(cfg, max_episode_steps=max_episode_steps)
        habitat_cfg = get_config(config_path=cfg.habitat.config_path, overrides=overrides)
        try:
            env = habitat.Env(habitat_cfg)
            return env, int(gid)
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to create Habitat env on gpu ids {tried}: {last_exc}") from last_exc


def _choose_open_start(sim, samples: int = 320) -> Tuple[np.ndarray, float, float]:
    pf = sim.pathfinder
    has_clearance = hasattr(pf, "distance_to_closest_obstacle")
    has_island = hasattr(pf, "island_radius")

    best_point = None
    best_score = -1e9
    best_clear = 0.0

    for _ in range(max(50, int(samples))):
        p = np.asarray(pf.get_random_navigable_point(), dtype=np.float32).reshape(3)
        clear = float(pf.distance_to_closest_obstacle(p, 5.0)) if has_clearance else 0.0
        island = float(pf.island_radius(p)) if has_island else 0.0
        score = clear + 0.05 * island
        if score > best_score:
            best_score = score
            best_point = p
            best_clear = clear

    if best_point is None:
        raise RuntimeError("Failed to sample any navigable point.")

    # Choose a heading that keeps short forward probes navigable.
    best_yaw = 0.0
    best_heading_score = -1e9
    for yaw in range(-180, 180, 15):
        rot = _quat_from_yaw_deg(float(yaw))
        fwd = hs_common.quat_rotate_vector(rot, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        fwd[1] = 0.0
        n = float(np.linalg.norm(fwd))
        if n < 1e-6:
            continue
        fwd = fwd / n
        probes = 0.0
        for dist in (0.5, 1.0, 1.5):
            q = np.asarray(best_point, dtype=np.float32).copy()
            q += fwd * float(dist)
            q[1] = best_point[1]
            probes += 1.0 if sim.pathfinder.is_navigable(q) else 0.0
        if probes > best_heading_score:
            best_heading_score = probes
            best_yaw = float(yaw)

    return best_point, best_yaw, float(best_clear)


def _run_option_probe(env, option_id: int, start_pos: np.ndarray, start_rot) -> Dict[str, object]:
    sim = env.sim
    _set_agent_state(sim, start_pos, start_rot)

    before_state = sim.get_agent_state()
    before_pos = np.asarray(before_state.position, dtype=np.float32).reshape(3)
    before_yaw = _yaw_deg_from_rotation(before_state.rotation)
    fwd0 = hs_common.quat_rotate_vector(before_state.rotation, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    right0 = hs_common.quat_rotate_vector(before_state.rotation, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    fwd0_xz = np.asarray([fwd0[0], fwd0[2]], dtype=np.float32)
    right0_xz = np.asarray([right0[0], right0[2]], dtype=np.float32)
    fwd0_xz = fwd0_xz / max(1e-6, float(np.linalg.norm(fwd0_xz)))
    right0_xz = right0_xz / max(1e-6, float(np.linalg.norm(right0_xz)))

    collided = False
    used_primitives: List[str] = []
    for primitive in macro_option_to_habitat(option_id):
        before_c = collision_count(env.get_metrics())
        env.step(primitive)
        used_primitives.append(str(primitive))
        after_c = collision_count(env.get_metrics())
        if after_c > before_c:
            collided = True
            break

    after_state = sim.get_agent_state()
    after_pos = np.asarray(after_state.position, dtype=np.float32).reshape(3)
    after_yaw = _yaw_deg_from_rotation(after_state.rotation)

    delta = after_pos - before_pos
    delta_xz = np.asarray([delta[0], delta[2]], dtype=np.float32)
    forward_m = float(np.dot(delta_xz, fwd0_xz))
    right_m = float(np.dot(delta_xz, right0_xz))
    displacement_m = float(np.linalg.norm(delta_xz))
    yaw_delta_deg = _wrap_deg(after_yaw - before_yaw)

    return {
        "option_id": int(option_id),
        "primitive_sequence_used": used_primitives,
        "collided": bool(collided),
        "yaw_delta_deg": float(yaw_delta_deg),
        "forward_m": float(forward_m),
        "right_m": float(right_m),
        "displacement_m": float(displacement_m),
        "delta_world_xyz": [float(delta[0]), float(delta[1]), float(delta[2])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe macro action outcomes in Habitat open space")
    parser.add_argument("--config", default="configs/vlm_nav.json")
    parser.add_argument("--output-dir", default="outputs/action_probe")
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--sample-points", type=int, default=320)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_hm3d_scene_alias(cfg.habitat.dataset_scenes_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    env, used_gpu_id = _create_env_with_gpu_fallback(cfg, max_episode_steps=int(args.max_episode_steps))
    with env:
        obs = env.reset()
        sim = env.sim
        episode = env.current_episode
        point, yaw_deg, clearance_m = _choose_open_start(sim, samples=int(args.sample_points))
        start_rot = _quat_from_yaw_deg(float(yaw_deg))
        _set_agent_state(sim, point, start_rot)
        obs0 = sim.get_sensor_observations()
        if "rgb" in obs0:
            Image.fromarray(obs0["rgb"][..., :3].astype(np.uint8)).save(images_dir / "start.png")

        results = []
        prompt_lines: List[str] = []
        for item in NUMBERED_MACRO_ACTIONS:
            option_id = int(item.option_id)
            probe = _run_option_probe(env, option_id, point, start_rot)
            results.append(probe)

            _set_agent_state(sim, point, start_rot)
            for primitive in macro_option_to_habitat(option_id):
                env.step(primitive)
            obs_after = sim.get_sensor_observations()
            if "rgb" in obs_after:
                Image.fromarray(obs_after["rgb"][..., :3].astype(np.uint8)).save(
                    images_dir / f"option_{option_id}.png"
                )

            prompt_lines.append(
                (
                    f"{option_id}: measured yaw_delta={probe['yaw_delta_deg']:+.1f} deg, "
                    f"forward={probe['forward_m']:+.2f} m, right={probe['right_m']:+.2f} m, "
                    f"displacement={probe['displacement_m']:.2f} m, collided={probe['collided']}"
                )
            )

    payload = {
        "scene_id": str(getattr(episode, "scene_id", "")),
        "episode_id": str(getattr(episode, "episode_id", "")),
        "gpu_device_id": int(used_gpu_id),
        "start_position_xyz": [float(point[0]), float(point[1]), float(point[2])],
        "start_yaw_deg": float(yaw_deg),
        "start_clearance_m": float(clearance_m),
        "turn_angle_deg": float(cfg.stage2.scan_angle_deg),
        "forward_step_m": float(cfg.stage2.forward_step_m),
        "results": results,
        "prompt_lines": prompt_lines,
        "images_dir": str(images_dir),
    }

    out_json = output_dir / "open_space_action_probe.json"
    out_txt = output_dir / "open_space_action_probe.prompt.txt"
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    out_txt.write_text("\n".join(f"- {line}" for line in prompt_lines) + "\n", encoding="utf-8")

    print("action_probe_done")
    print(f"output_json: {out_json}")
    print(f"output_prompt: {out_txt}")
    print(f"images_dir: {images_dir}")


if __name__ == "__main__":
    main()

