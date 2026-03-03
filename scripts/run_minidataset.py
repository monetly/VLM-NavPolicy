#!/usr/bin/env python3
"""Run one VLM endpoint on mini-dataset and export per-sample + summary results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image

from vlm_nav.config import VLMConfig, load_config
from vlm_nav.habitat_utils import effective_turn_angle_deg
from vlm_nav.ground_overlay import DEFAULT_ACTIONS, render_prob_summary
from vlm_nav.vlm_client import VLMScorer


def _load_rows(metadata_csv: Path) -> List[Dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _coarse_label(option_id: str) -> str:
    s = str(option_id).strip().upper()
    if s in {"A", "B"}:
        return "turn_left"
    if s == "C":
        return "move_forward"
    if s in {"D", "E"}:
        return "turn_right"
    return "observe"


def _build_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    n = max(1, len(rows))
    ent_sum = 0.0
    opt_counts: Counter[str] = Counter()
    coarse_match = 0
    by_cat = defaultdict(lambda: {"ent": 0.0, "match": 0, "count": 0})

    for row in rows:
        cat = str(row["category"])
        ent = float(row["entropy"])
        opt = str(row["option"])
        ent_sum += ent
        opt_counts[opt] += 1
        coarse_match += int(row["match_coarse"])
        by_cat[cat]["ent"] += ent
        by_cat[cat]["match"] += int(row["match_coarse"])
        by_cat[cat]["count"] += 1

    by_category = {
        cat: {
            "avg_entropy": v["ent"] / max(1, v["count"]),
            "coarse_match_rate": v["match"] / max(1, v["count"]),
            "count": v["count"],
        }
        for cat, v in by_cat.items()
    }
    return {
        "num_samples": len(rows),
        "avg_entropy": ent_sum / n,
        "option_counts": dict(opt_counts),
        "coarse_match_rate": coarse_match / n,
        "by_category": by_category,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VLM on mini-dataset")
    ap.add_argument("--config", default="configs/vlm_nav.json")
    ap.add_argument("--metadata-csv", default="data/mini_dataset/metadata.csv")
    ap.add_argument("--output-dir", default="outputs/minidataset_run")
    ap.add_argument("--model-tag", default="qwen")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3-vl")
    ap.add_argument("--turn-angle-deg", type=float, default=None)
    ap.add_argument("--forward-step-m", type=float, default=None)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-images", dest="save_images",
                    action="store_true", default=True)
    ap.add_argument("--no-save-images", dest="save_images",
                    action="store_false")
    args = ap.parse_args()

    rows = _load_rows(Path(args.metadata_csv))
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError("No rows loaded from mini-dataset metadata.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlay"
    prob_dir = out_dir / "prob_vis"
    if args.save_images:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = load_config(args.config)
    turn_deg = float(effective_turn_angle_deg(
        args.turn_angle_deg or exp_cfg.navigation.scan_angle_deg))
    fwd_m = float(args.forward_step_m or exp_cfg.navigation.forward_step_m)

    scorer = VLMScorer(VLMConfig(
        base_url=args.base_url, model=args.model,
        seed=args.seed, request_timeout_sec=args.timeout_sec,
        n_probs=64, use_mock_when_unreachable=False,
    ))

    results: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        rgb = np.array(Image.open(row["path"]).convert("RGB"), dtype=np.uint8)
        dist = scorer.score_direction(
            rgb, return_annotated=args.save_images,
            turn_angle_deg=turn_deg, forward_step_m=fwd_m,
            camera_height_m=float(exp_cfg.navigation.camera_height_m),
        )

        frame = f"{idx+1:03d}_{Path(row['path']).stem}.png"
        overlay_path = prob_path = ""
        if dist.annotated_rgb is not None and args.save_images:
            overlay_path = str(overlay_dir / frame)
            Image.fromarray(dist.annotated_rgb).save(overlay_path)
            vis = render_prob_summary(
                dist.annotated_rgb, DEFAULT_ACTIONS,
                dist.prob_by_option, dist.selected_option)
            prob_path = str(prob_dir / frame)
            Image.fromarray(vis).save(prob_path)

        coarse = _coarse_label(dist.selected_option)
        rec: Dict[str, object] = {
            "model_tag": args.model_tag,
            "category": row.get("category", ""),
            "path": row.get("path", ""),
            "scene_id": row.get("scene_id", ""),
            "episode_id": row.get("episode_id", ""),
            "step": row.get("step", ""),
            "follower_action": row.get("follower_action", ""),
            "option": dist.selected_option,
            "action_name": dist.selected_action_name,
            "coarse": coarse,
            "entropy": float(dist.entropy),
            "mock": int(dist.used_mock),
            "overlay": overlay_path,
            "prob_vis": prob_path,
            "match_coarse": int(coarse == row.get("follower_action", "")),
        }
        for oid in ("A", "B", "C", "D", "E"):
            rec[f"P_{oid}"] = float(dist.prob_by_option.get(oid, 0.0))
        results.append(rec)
        print(f"[{args.model_tag}] {idx+1:02d}/{len(rows)} "
              f"-> opt={dist.selected_option} H={dist.entropy:.3f}")

    summary = _build_summary(results)
    summary.update(model_tag=args.model_tag, base_url=args.base_url,
                   model=args.model, seed=args.seed,
                   turn_angle_deg=turn_deg, forward_step_m=fwd_m)

    csv_path = out_dir / "per_sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True); f.write("\n")

    print("done"); print(f"csv: {csv_path}"); print(f"json: {json_path}")


if __name__ == "__main__":
    main()
