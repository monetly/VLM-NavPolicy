"""Two-stage VLM scorer for ground-projected A-E direction selection.

Stage 1 (Safety): Grammar-constrained Y/N — is there safe passage?
Stage 2 (Direction): Grammar-constrained A-E — which direction is safest?

Uses ``post_sampling_probs: true`` so llama.cpp returns grammar-constrained
probability distributions instead of raw pre-mask logprobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import base64
import math
import random
import re

import numpy as np
from PIL import Image
import requests

from .actions import MacroAction, MACRO_ACTIONS, recompute_tips
from .config import VLMConfig
from .entropy import normalize_distribution_generic, shannon_entropy_bits_generic
from .ground_overlay import DEFAULT_ACTIONS, render_ground_overlay


_IMG_TAG = "<__media__>"

# ---------------------------------------------------------------------------
# Stage 1:  Safety assessment (Y / N)
# ---------------------------------------------------------------------------
_SAFETY_SYSTEM = (
    "You are the visual navigation core of an autonomous mobile robot. "
    "Your task is to determine whether there is any safe passable direction ahead."
)
_SAFETY_USER = (
    "Look at the current camera image. "
    "Is there at least one safe, unobstructed direction the robot can move toward?\n"
    "Answer only Y (yes, safe passage exists) or N (all directions blocked)."
)

# ---------------------------------------------------------------------------
# Stage 2:  Direction selection (A-E)
# ---------------------------------------------------------------------------
_DIRECTION_SYSTEM = (
    "You are the visual navigation core of an autonomous mobile robot. "
    "Your task is to choose the safest travel direction from the current camera view "
    "to avoid obstacles and reach the next area."
)
_DIRECTION_USER = (
    "The 5 yellow trajectory lines on the ground (from leftmost to rightmost) represent the exact physical paths the robot can take for its next move. "
    "The solid dot at the end of each line marks the precise location the robot will physically reach and stand on after taking that action.\n\n"
    "You must choose the safest path and output exactly one corresponding symbol:\n"
    "- Output '°' for the 1st line (Leftmost turn)\n"
    "- Output '®' for the 2nd line (Slight left turn)\n"
    "- Output '¬' for the 3rd line (Straight Forward)\n"
    "- Output '¦' for the 4th line (Slight right turn)\n"
    "- Output '¯' for the 5th line (Rightmost turn)\n\n"
    "Crucial Collision Rules:\n"
    "1. Check the ENTIRE line: If the trajectory line crosses or touches ANY obstacle (furniture legs, walls, debris) on the ground, the robot will collide while moving.\n"
    "2. Check the END POINT: If the solid dot at the tip of the line lands on or inside an obstacle, the robot will crash into it at the end of the step.\n"
    "3. Only select a path where BOTH the entire line and its end point lie completely on clear, open, walkable floor.\n\n"
    "Output exactly one assigned symbol (°, ®, ¬, ¦, or ¯) corresponding to the safest trajectory."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActionDistribution:
    """Result of Stage 2 direction scoring."""
    prob_by_option: Dict[str, float]
    prob_by_action_name: Dict[str, float]
    entropy: float
    selected_option: str
    selected_action_name: str
    annotated_rgb: Optional[np.ndarray] = None
    used_mock: bool = False
    error: Optional[str] = None
    stage1_safe: Optional[bool] = None


@dataclass
class SafetyResult:
    """Result of Stage 1 safety check."""
    is_safe: bool
    prob_yes: float
    prob_no: float


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class VLMScorer:
    """Two-stage VLM scorer: safety (Y/N) → direction (A-E)."""

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self._url = cfg.base_url.rstrip("/") + "/completion"
        self._counter = 0

    # ── Stage 1 ───────────────────────────────────────────────────────

    def check_safety(self, rgb: np.ndarray) -> SafetyResult:
        """Ask the VLM whether safe passage exists (Y/N)."""
        encoded = self._encode_png(rgb)
        payload = self._payload_safety(encoded)
        try:
            rj = self._post(payload)
            probs = self._parse_constrained_probs(rj, {"Y", "N"})
            py = probs.get("Y", 0.5)
            pn = probs.get("N", 0.5)
            return SafetyResult(is_safe=(py >= pn), prob_yes=py, prob_no=pn)
        except Exception:
            return SafetyResult(is_safe=True, prob_yes=0.5, prob_no=0.5)

    # ── Stage 2 ───────────────────────────────────────────────────────

    def score_direction(
        self,
        rgb: np.ndarray,
        actions: Sequence[MacroAction] = DEFAULT_ACTIONS,
        turn_angle_deg: float = 30.0,
        forward_step_m: float = 0.2,
        camera_height_m: float = 0.6,
        hfov_deg: float = 79.0,
        return_annotated: bool = False,
    ) -> ActionDistribution:
        """Score A-E directions from the ground-projected overlay image."""
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Expected HxWx3, got shape={rgb.shape}")

        actions = recompute_tips(actions, turn_angle_deg, forward_step_m, camera_height_m, hfov_deg)
        option_ids = [a.option_id for a in actions]
        option_set = set(option_ids)
        id_to_name = {a.option_id: a.action_name for a in actions}

        annotated = render_ground_overlay(
            rgb, actions, turn_angle_deg, forward_step_m, camera_height_m, hfov_deg,
        )
        encoded = self._encode_png(annotated)

        try:
            payload = self._payload_direction(encoded, option_ids)
            rj = self._post(payload)

            prob_by_option = self._parse_constrained_probs(rj, option_set)
            prob_by_option = normalize_distribution_generic(prob_by_option)
            entropy = shannon_entropy_bits_generic(prob_by_option)
            selected = max(prob_by_option.items(), key=lambda kv: kv[1])[0]

            prob_by_name: Dict[str, float] = {}
            for oid, p in prob_by_option.items():
                name = id_to_name[oid]
                prob_by_name[name] = prob_by_name.get(name, 0.0) + p
            prob_by_name = normalize_distribution_generic(prob_by_name)

            return ActionDistribution(
                prob_by_option=prob_by_option,
                prob_by_action_name=prob_by_name,
                entropy=entropy,
                selected_option=selected,
                selected_action_name=id_to_name[selected],
                annotated_rgb=annotated if return_annotated else None,
            )
        except Exception as exc:
            if not self.cfg.use_mock_when_unreachable:
                raise
            return self._mock_score(rgb, actions, str(exc),
                                    return_annotated, annotated)

    # ── Payload builders ──────────────────────────────────────────────

    def _payload_safety(self, encoded_rgb: str) -> Dict[str, Any]:
        return {
            "model": self.cfg.model,
            "prompt": {
                "prompt_string": (
                    f"{_SAFETY_SYSTEM}\n\n"
                    f"Current camera image: {_IMG_TAG}\n\n"
                    f"{_SAFETY_USER}"
                ),
                "multimodal_data": [encoded_rgb],
            },
            "n_predict": 1,
            "temperature": 0.1,
            "top_k": 2,
            "top_p": 1.0,
            "grammar": 'root ::= "Y" | "N"',
            "n_probs": self.cfg.n_probs,
            "post_sampling_probs": True,   # ← grammar-constrained probs
            "cache_prompt": False,
            "seed": self._next_seed(),
        }

    def _payload_direction(
        self,
        encoded_rgb: str,
        option_ids: Sequence[str],
    ) -> Dict[str, Any]:
        grammar = self._grammar(option_ids)
        temp = 1.0 if self.cfg.temperature < 0 else max(0.1, self.cfg.temperature)
        top_k = 10 if self.cfg.top_k <= 0 else self.cfg.top_k
        top_p = 0.95 if self.cfg.top_p >= 1.0 else self.cfg.top_p
        return {
            "model": self.cfg.model,
            "prompt": {
                "prompt_string": (
                    f"{_DIRECTION_SYSTEM}\n\n"
                    f"Current camera image: {_IMG_TAG}\n\n"
                    f"{_DIRECTION_USER}"
                ),
                "multimodal_data": [encoded_rgb],
            },
            "n_predict": 1,
            "temperature": temp,
            "top_k": 0,
            "top_p": 1.0,
            "grammar": grammar,
            "n_probs": 50,
            "post_sampling_probs": True,   # ← grammar-constrained probs
            "cache_prompt": False,
            "seed": self._next_seed(),
        }

    # ── Grammar ───────────────────────────────────────────────────────

    @staticmethod
    def _grammar(ids: Sequence[str]) -> str:
        parts = [f'"{s.strip()}"' for s in ids if s.strip()]
        if not parts:
            parts = ['"C"']
        return f"root ::= {' | '.join(parts)}"

    # ── Response parsing ──────────────────────────────────────────────

    def _parse_constrained_probs(
        self,
        rj: Dict[str, Any],
        valid_ids: Set[str],
    ) -> Dict[str, float]:
        """Extract probability distribution from grammar-constrained response.

        With ``post_sampling_probs: true`` the ``top_logprobs`` contain only
        tokens that survived grammar masking + sampling, so we can directly
        read the A-E (or Y/N) probabilities.
        """
        probs = {oid: 0.0 for oid in valid_ids}

        items = self._top_prob_items(rj)
        for item in items:
            tok = str(item.get("token", "")).strip()
            if tok in valid_ids:
                probs[tok] += self._item_prob(item)

        # If top_probs yielded nothing, use the content token as last resort.
        if sum(probs.values()) <= 1e-8:
            content = str(rj.get("content", "")).strip()
            if content in valid_ids:
                # Uniform floor + small boost for selected.
                for oid in valid_ids:
                    probs[oid] = 1.0
                probs[content] += 4.0
            else:
                # Complete fallback: uniform.
                for oid in valid_ids:
                    probs[oid] = 1.0

        return probs

    @staticmethod
    def _top_prob_items(rj: Dict[str, Any]) -> List[Dict[str, Any]]:
        for field in ("probs", "completion_probabilities"):
            val = rj.get(field)
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, dict):
                    for key in ("top_probs", "top_logprobs"):
                        items = first.get(key)
                        if isinstance(items, list):
                            return items
        return []

    @staticmethod
    def _item_prob(item: Dict[str, Any]) -> float:
        if "prob" in item:
            try:
                return max(0.0, float(item["prob"]))
            except Exception:
                return 0.0
        if "logprob" in item:
            try:
                return max(0.0, math.exp(float(item["logprob"])))
            except Exception:
                return 0.0
        return 0.0

    # ── Utilities ─────────────────────────────────────────────────────

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self._url, json=payload,
                          timeout=self.cfg.request_timeout_sec)
        r.raise_for_status()
        return r.json()

    def _next_seed(self) -> int:
        self._counter += 1
        if self.cfg.seed is None:
            return random.randint(1, 2_147_483_647)
        return max(1, (int(self.cfg.seed) + self._counter * 1009) % 2_147_483_647)

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> str:
        img = Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ── Mock fallback ─────────────────────────────────────────────────

    def _mock_score(
        self,
        rgb: np.ndarray,
        actions: Sequence[MacroAction],
        error: str,
        return_annotated: bool,
        annotated: Optional[np.ndarray],
    ) -> ActionDistribution:
        h, w = rgb.shape[:2]
        gray = rgb[..., :3].astype(np.float32).mean(axis=2) / 255.0
        edge = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1])) + \
               np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))

        scores: Dict[str, float] = {}
        for a in actions:
            cx = int(np.clip(round(a.tip_xy_norm[0] * w), 0, w - 1))
            cy = int(np.clip(round(a.tip_xy_norm[1] * h), 0, h - 1))
            rx, ry = max(8, int(0.06 * w)), max(8, int(0.06 * h))
            patch = edge[max(0, cy-ry):min(h, cy+ry), max(0, cx-rx):min(w, cx+rx)]
            scores[a.option_id] = 1.0 / (float(patch.mean()) + 1e-3)

        prob = normalize_distribution_generic(scores)
        ent = shannon_entropy_bits_generic(prob)
        sel = max(prob.items(), key=lambda kv: kv[1])[0]
        id_to_name = {a.option_id: a.action_name for a in actions}

        return ActionDistribution(
            prob_by_option=prob,
            prob_by_action_name=normalize_distribution_generic(
                {id_to_name[k]: v for k, v in prob.items()}),
            entropy=ent,
            selected_option=sel,
            selected_action_name=id_to_name[sel],
            annotated_rgb=annotated if return_annotated else None,
            used_mock=True, error=error,
        )
