#!/usr/bin/env python3
"""Mainline run: draw bottom direction icons on RGB, VLM choose option, execute macro actions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Set, Tuple
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from habitat.config.default import get_config
import habitat
from habitat.utils.visualizations import maps as habitat_maps
from PIL import Image

from vlm_nav.agent import NavigationAgent
from vlm_nav.config import load_config
from vlm_nav.habitat_utils import (
    build_habitat_overrides,
    effective_turn_angle_deg,
    ensure_hm3d_scene_alias,
)


def _parse_indices(raw: Optional[str]) -> Set[int]:
    if not raw:
        return set()
    values = set()
    for chunk in raw.split(","):
        token = chunk.strip()
        if token:
            values.add(int(token))
    return values


def _episode_matches(scene_id: Optional[str], selected_indices: Set[int], seen_idx: int, episode) -> bool:
    if scene_id and scene_id not in str(getattr(episode, "scene_id", "")):
        return False
    if not selected_indices:
        return True
    if seen_idx in selected_indices:
        return True
    episode_id = str(getattr(episode, "episode_id", "")).strip()
    return bool(episode_id.isdigit() and int(episode_id) in selected_indices)


def _load_trace_positions(trace_csv_path: Path) -> List[Tuple[float, float, float]]:
    if not trace_csv_path.exists():
        return []
    points: List[Tuple[float, float, float]] = []
    with trace_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append((float(row["x"]), float(row["y"]), float(row["z"])))
            except Exception:  # pylint: disable=broad-except
                continue
    return points


def _render_topdown_trajectory(env, trace_csv_path: Path, output_png_path: Path) -> Optional[str]:
    points = _load_trace_positions(trace_csv_path)
    if not points:
        return None

    topdown = habitat_maps.get_topdown_map_from_sim(
        env.sim,
        map_resolution=1024,
        draw_border=True,
        meters_per_pixel=0.05,
    )
    colored = habitat_maps.colorize_topdown_map(topdown)
    h, w = int(colored.shape[0]), int(colored.shape[1])

    grid_points: List[Tuple[int, int]] = []
    for x, _y, z in points:
        try:
            gx, gy = habitat_maps.to_grid(
                realworld_x=float(z),
                realworld_y=float(x),
                grid_resolution=(h, w),
                sim=env.sim,
            )
        except Exception:  # pylint: disable=broad-except
            continue
        if 0 <= gx < h and 0 <= gy < w:
            grid_points.append((int(gx), int(gy)))

    if not grid_points:
        return None
    if len(grid_points) >= 2:
        habitat_maps.draw_path(colored, grid_points, color=10, thickness=2)

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colored).save(output_png_path)
    return str(output_png_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mainline arrow-overlay VLM navigation in Habitat")
    parser.add_argument("--config", default="configs/vlm_nav.json")
    parser.add_argument("--run-name", default="mainline_run")
    parser.add_argument("--split", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--max-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--vlm-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vlm_seed is not None:
        cfg.vlm.seed = int(args.vlm_seed)
    requested_turn = float(cfg.navigation.scan_angle_deg)
    effective_turn = float(effective_turn_angle_deg(cfg.navigation.scan_angle_deg))
    if abs(requested_turn - effective_turn) > 1e-9:
        print(
            f"note: habitat.simulator.turn_angle requires integer; "
            f"requested {requested_turn:.3f} deg -> using {effective_turn:.0f} deg"
        )
    ensure_hm3d_scene_alias(cfg.habitat.dataset_scenes_dir)

    overrides = build_habitat_overrides(
        exp_cfg=cfg,
        split=args.split,
        scene_id=args.scene_id,
        max_episode_steps=args.max_steps,
    )
    habitat_cfg = get_config(
        config_path=cfg.habitat.config_path,
        overrides=overrides,
    )

    output_dir = Path("outputs") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    try:
        with habitat.Env(habitat_cfg) as env:
            agent = NavigationAgent(cfg)
            selected_indices = _parse_indices(args.episode_indices)

            dataset_len = len(getattr(env, "episodes", []))
            max_seen = max(dataset_len, args.max_episodes * 4) if dataset_len > 0 else args.max_episodes * 4

            processed = 0
            seen = 0
            while processed < args.max_episodes and seen < max_seen:
                obs = env.reset()
                episode = env.current_episode

                if not _episode_matches(args.scene_id, selected_indices, seen, episode):
                    seen += 1
                    continue

                scene_name = Path(str(episode.scene_id)).stem
                episode_uid = f"ep{seen:04d}_{scene_name}_{episode.episode_id}"

                metrics = agent.run_episode(
                    env=env,
                    max_steps=args.max_steps,
                    output_dir=output_dir,
                    episode_uid=episode_uid,
                    reset_env=False,
                    initial_obs=obs,
                )
                trace_csv_path = Path(str(metrics.get("trace_csv", "")))
                topdown_png = output_dir / f"{episode_uid}_topdown.png"
                topdown_path = _render_topdown_trajectory(
                    env=env,
                    trace_csv_path=trace_csv_path,
                    output_png_path=topdown_png,
                )

                row = {
                    "seen_index": seen,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "topdown_map": topdown_path or "",
                    **metrics,
                }
                summary_rows.append(row)
                processed += 1
                seen += 1
    except Exception as exc:
        text = str(exc)
        if "WindowlessContext" in text or "unable to find CUDA device" in text:
            raise RuntimeError(
                "Habitat EGL context init failed. Run on a machine/session with working EGL+GPU "
                "or adjust gpu_device_id in config."
            ) from exc
        raise

    if not summary_rows:
        raise RuntimeError("No episode matched filters. Try removing --scene-id/--episode-indices.")

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = output_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print("mainline_run_finished")
    print(f"episodes: {len(summary_rows)}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
