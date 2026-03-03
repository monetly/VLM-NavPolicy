"""Two-stage navigation agent with entropy safety net.

Per step:
1) Stage 1: VLM Y/N safety check on raw RGB
2) If N → fallback turn (rotate left by scan_angle_deg)
3) Stage 2: Ground trajectory overlay → VLM A-E direction
4) Entropy / confidence guard → if uncertain, fallback turn
5) Execute selected macro-action primitives in Habitat
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
from PIL import Image

from .actions import option_to_habitat, MACRO_ACTIONS, MacroAction
from .config import NavExperimentConfig
from .habitat_utils import collision_count, effective_turn_angle_deg
from .ground_overlay import DEFAULT_ACTIONS, render_prob_summary
from .vlm_client import VLMScorer


class NavigationAgent:
    """Two-stage agent: safety (Y/N) → direction (A-E) + entropy guard."""

    def __init__(self, cfg: NavExperimentConfig):
        self.cfg = cfg
        self.scorer = VLMScorer(cfg.vlm)
        self._actions: Tuple[MacroAction, ...] = tuple(DEFAULT_ACTIONS)
        self._turn_deg = float(effective_turn_angle_deg(cfg.navigation.scan_angle_deg))

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _log(handle, payload: Dict[str, object]) -> None:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    @staticmethod
    def _agent_xyz(env) -> Optional[Tuple[float, float, float]]:
        sim = getattr(env, "sim", None)
        if sim is None:
            return None
        try:
            state = sim.get_agent_state()
        except Exception:
            try:
                state = sim.get_agent_state(0)
            except Exception:
                return None
        pos = getattr(state, "position", None)
        if pos is None:
            return None
        try:
            arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if arr.size < 3:
            return None
        return float(arr[0]), float(arr[1]), float(arr[2])

    @staticmethod
    def _write_trace(path: Path, rows: List[Dict[str, Any]]) -> None:
        cols = ["trace_idx", "decision_idx", "step", "prim_idx",
                "option", "action_name", "primitive", "collided",
                "x", "y", "z"]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            if rows:
                w.writerows(rows)

    # ── main loop ──────────────────────────────────────────────────────

    def run_episode(
        self,
        env,
        max_steps: int,
        output_dir: Path,
        episode_uid: str,
        reset_env: bool = True,
        initial_obs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{episode_uid}.jsonl"
        trace_path = output_dir / f"{episode_uid}_trace.csv"
        overlay_dir = output_dir / f"{episode_uid}_overlays"
        prob_dir = output_dir / f"{episode_uid}_prob_vis"
        frames_dir = output_dir / f"{episode_uid}_frames"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        obs = env.reset() if reset_env or initial_obs is None else initial_obs

        def _save_frame(curr_step: int, curr_obs: Dict[str, np.ndarray]) -> None:
            if "rgb" in curr_obs:
                Image.fromarray(curr_obs["rgb"]).save(frames_dir / f"s{curr_step:04d}.png")

        step = 0
        decisions = 0
        vlm_calls = 0
        mock_calls = 0
        safety_checks = 0
        fallbacks = 0
        collisions = 0
        consec_fb = 0
        trace: List[Dict[str, Any]] = []
        id_to_name = {a.option_id: a.action_name for a in self._actions}
        last_collided: Optional[str] = None

        ent_thr = float(self.cfg.navigation.entropy_threshold)
        conf_thr = float(self.cfg.navigation.confidence_threshold)
        max_fb = int(self.cfg.navigation.max_consecutive_fallbacks)

        xyz = self._agent_xyz(env)
        if xyz:
            trace.append(dict(trace_idx=0, decision_idx=-1, step=0,
                              prim_idx=-1, option="", action_name="start",
                              primitive="start", collided=0,
                              x=xyz[0], y=xyz[1], z=xyz[2]))

        _save_frame(step, obs)

        with log_path.open("w", encoding="utf-8") as lf:
            while step < max_steps and not env.episode_over:
                rgb = obs["rgb"]

                # ── Stage 1: safety ──
                safety = self.scorer.check_safety(rgb)
                safety_checks += 1
                vlm_calls += 1
                self._log(lf, {
                    "event": "safety_check", "decision": decisions,
                    "step": step, "safe": safety.is_safe,
                    "p_yes": safety.prob_yes, "p_no": safety.prob_no,
                    "consec_fb": consec_fb,
                })

                if not safety.is_safe and consec_fb < max_fb:
                    self._log(lf, {"event": "fallback", "step": step,
                                   "reason": "stage1_unsafe",
                                   "consec_fb": consec_fb + 1})
                    if step < max_steps and not env.episode_over:
                        obs = env.step("turn_left")
                        step += 1; fallbacks += 1; consec_fb += 1
                        _save_frame(step, obs)
                        self._append_trace(trace, decisions, step,
                                           "fallback", "turn_left_fallback",
                                           "turn_left", env)
                    decisions += 1
                    continue

                # ── Stage 2: direction ──
                dist = self.scorer.score_direction(
                    rgb, self._actions, self._turn_deg,
                    float(self.cfg.navigation.forward_step_m),
                    float(self.cfg.navigation.camera_height_m),
                    return_annotated=True,
                )
                vlm_calls += 1
                if dist.used_mock:
                    mock_calls += 1

                max_p = max(dist.prob_by_option.values()) if dist.prob_by_option else 0.0

                # ── Entropy guard ──
                if (dist.entropy > ent_thr or max_p < conf_thr) and consec_fb < max_fb:
                    self._log(lf, {"event": "fallback", "step": step,
                                   "reason": "entropy_guard",
                                   "entropy": dist.entropy, "max_p": max_p,
                                   "consec_fb": consec_fb + 1})
                    if step < max_steps and not env.episode_over:
                        obs = env.step("turn_left")
                        step += 1; fallbacks += 1; consec_fb += 1
                        _save_frame(step, obs)
                        self._append_trace(trace, decisions, step,
                                           "fallback", "entropy_fallback",
                                           "turn_left", env)
                    decisions += 1
                    continue

                consec_fb = 0

                # ── Collision override ──
                raw_opt = dist.selected_option
                opt = raw_opt
                overridden = False
                if last_collided and opt == last_collided:
                    alt = {k: v for k, v in dist.prob_by_option.items()}
                    alt[opt] = 0.0
                    if sum(alt.values()) > 0:
                        opt = max(alt.items(), key=lambda kv: kv[1])[0]
                    if opt != raw_opt:
                        overridden = True

                action_name = id_to_name.get(opt, dist.selected_action_name)

                # Save images.
                tag = f"d{decisions:04d}_s{step:04d}"
                if dist.annotated_rgb is not None:
                    Image.fromarray(dist.annotated_rgb).save(
                        overlay_dir / f"{tag}.png")
                    vis = render_prob_summary(
                        dist.annotated_rgb, self._actions,
                        dist.prob_by_option, opt)
                    Image.fromarray(vis).save(prob_dir / f"{tag}.png")

                self._log(lf, {
                    "event": "decision", "decision": decisions, "step": step,
                    "option": opt, "action": action_name,
                    "raw_option": raw_opt, "overridden": overridden,
                    "entropy": dist.entropy, "max_p": max_p,
                    "prob": dist.prob_by_option,
                    "mock": dist.used_mock, "safe": safety.is_safe,
                })
                decisions += 1

                # ── Execute primitives ──
                macro_seq = option_to_habitat(opt)
                prim_used = 0
                collided = False
                for prim in macro_seq:
                    if step >= max_steps or env.episode_over:
                        break
                    before = collision_count(env.get_metrics())
                    obs = env.step(prim)
                    step += 1; prim_used += 1
                    _save_frame(step, obs)
                    after = collision_count(env.get_metrics())
                    self._append_trace(trace, decisions - 1, step,
                                       opt, action_name, prim, env,
                                       int(after > before))
                    if after > before:
                        collisions += 1; collided = True; break

                self._log(lf, {
                    "event": "execute", "step": step, "option": opt,
                    "action": action_name, "primitives": list(macro_seq),
                    "used": prim_used, "collided": collided,
                })
                last_collided = opt if collided else None

        self._write_trace(trace_path, trace)
        return {
            "steps": float(step), "decisions": float(decisions),
            "vlm_calls": float(vlm_calls), "mock_calls": float(mock_calls),
            "safety_checks": float(safety_checks),
            "fallbacks": float(fallbacks),
            "collisions": float(collisions),
            "episode_over": float(env.episode_over),
            "trace_csv": str(trace_path),
        }

    def _append_trace(self, trace, decision, step, option, action_name,
                      primitive, env, collided=0):
        xyz = self._agent_xyz(env)
        if xyz:
            trace.append(dict(
                trace_idx=len(trace), decision_idx=decision, step=step,
                prim_idx=len([r for r in trace
                              if r.get("decision_idx") == decision]) - 1,
                option=option, action_name=action_name,
                primitive=primitive, collided=collided,
                x=xyz[0], y=xyz[1], z=xyz[2],
            ))
