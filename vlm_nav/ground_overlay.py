"""Ground-projected fan trajectory overlay for A-E direction selection."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .actions import MACRO_ACTIONS, MacroAction, recompute_tips

# Public re-exports for backward compat.
DEFAULT_ACTIONS: Tuple[MacroAction, ...] = MACRO_ACTIONS

# --- Visual constants ---
_TRAJ_COLOR = (255, 215, 0)         # Gold #FFD700
_ANCHOR_BG = (20, 20, 20)           # Near-black
_ANCHOR_FG = (255, 255, 255)        # White letter
_TRAJ_WIDTH_FRAC = 0.007            # Line width as fraction of image size
_ANCHOR_R_FRAC = 0.040              # Anchor radius as fraction of image size
_TRAJ_LENGTH_FRAC = 0.20            # Line length as fraction of image height


def _load_font(size: int = 20) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _yaw_deg(action: MacroAction, _turn_angle_deg: float) -> float:
    """Fixed semantic yaw delta in degrees for ground visualization.
    A: -60, B: -30, C: 0, D: +30, E: +60
    """
    mapping = {
        "A": -60.0,
        "B": -30.0,
        "C": 0.0,
        "D": 30.0,
        "E": 60.0,
    }
    return mapping.get(action.option_id, 0.0)


# ---------------------------------------------------------------------------
# Main overlay renderer
# ---------------------------------------------------------------------------

def render_ground_overlay(
    rgb: np.ndarray,
    actions: Sequence[MacroAction] = DEFAULT_ACTIONS,
    turn_angle_deg: float = 30.0,
    forward_step_m: float = 0.2,
    hfov_deg: float = 79.0,
) -> np.ndarray:
    """Draw fan-shaped ground trajectory lines with A-E anchor labels.

    Origin: bottom-center of image (camera position).
    Each line extends upward at the yaw angle of the corresponding action.
    Endpoint: high-contrast circle with letter label.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB, got shape={rgb.shape}")

    actions = recompute_tips(actions, turn_angle_deg, forward_step_m, hfov_deg)

    base = Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size

    origin_x = w / 2.0
    origin_y = float(h) - 1.0
    traj_len = _TRAJ_LENGTH_FRAC * h
    anchor_r = max(14, int(round(min(w, h) * _ANCHOR_R_FRAC)))
    line_w = max(2, int(round(min(w, h) * _TRAJ_WIDTH_FRAC)))
    font = _load_font(max(12, int(round(anchor_r * 1.3))))

    for action in actions:
        yaw = _yaw_deg(action, turn_angle_deg)
        angle_rad = math.radians(yaw)
        tip_x = origin_x + traj_len * math.sin(angle_rad)
        tip_y = origin_y - traj_len * math.cos(angle_rad)

        margin = anchor_r + 4
        tip_x = max(margin, min(w - margin, tip_x))
        tip_y = max(margin, min(h - margin, tip_y))

        # Trajectory line.
        draw.line(
            [(origin_x, origin_y), (tip_x, tip_y)],
            fill=(*_TRAJ_COLOR, 220), width=line_w,
        )
        # Anchor circle.
        draw.ellipse(
            [(tip_x - anchor_r, tip_y - anchor_r),
             (tip_x + anchor_r, tip_y + anchor_r)],
            fill=(*_ANCHOR_BG, 230),
            outline=(*_ANCHOR_FG, 255), width=2,
        )
        # Letter label.
        label = action.option_id
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (tip_x - tw / 2.0, tip_y - th / 2.0 - bbox[1]),
            label, fill=(*_ANCHOR_FG, 255), font=font,
        )

    composited = Image.alpha_composite(base.convert("RGBA"), overlay)
    return np.array(composited.convert("RGB"))


# ---------------------------------------------------------------------------
# Probability visualization panel
# ---------------------------------------------------------------------------

_PALETTE = [
    (255, 186, 8),    # A – amber
    (251, 133, 0),    # B – orange
    (33, 158, 188),   # C – teal
    (2, 48, 71),      # D – dark blue
    (142, 202, 230),  # E – light blue
]


def render_prob_summary(
    annotated_rgb: np.ndarray,
    actions: Sequence[MacroAction],
    prob_by_option: Dict[str, float],
    selected_option: str,
) -> np.ndarray:
    """Append a probability bar panel below the annotated image."""
    if annotated_rgb.ndim != 3 or annotated_rgb.shape[2] < 3:
        raise ValueError(f"Expected HxWx3, got shape={annotated_rgb.shape}")

    base = Image.fromarray(annotated_rgb.astype(np.uint8), mode="RGB")
    w, h = base.size
    panel_h = max(140, int(round(h * 0.28)))

    canvas = Image.new("RGB", (w, h + panel_h), color=(22, 22, 24))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    font = _load_font(16)
    font_sm = _load_font(13)

    draw.text((16, h + 12), f"Selected: {selected_option}",
              fill=(245, 245, 245, 255), font=font)

    slot_w = w / max(1, len(actions))
    y0 = h + int(round(panel_h * 0.30))
    y1 = h + int(round(panel_h * 0.85))

    for i, action in enumerate(actions):
        oid = action.option_id
        p = float(prob_by_option.get(oid, 0.0))
        color = _PALETTE[i % len(_PALETTE)]
        x0 = int(round(i * slot_w))
        x1 = int(round((i + 1) * slot_w)) - 1
        is_sel = (oid == selected_option)

        draw.rectangle(
            [(x0, y0), (x1, y1)],
            fill=(*color, 90 if is_sel else 50),
            outline=(*color, 255), width=3 if is_sel else 2,
        )
        draw.text((x0 + 8, y0 + 6), oid,
                  fill=(245, 245, 245, 255), font=font)
        draw.text((x0 + 8, y1 - 22), f"P={p:.3f}",
                  fill=(235, 235, 235, 255), font=font_sm)

    return np.array(canvas)


def actions_to_prompt_lines(
    actions: Sequence[MacroAction],
    turn_angle_deg: Optional[float] = None,
) -> List[str]:
    """Generate prompt description lines for each action."""
    lines: List[str] = []
    for a in actions:
        seq = " ".join(a.primitive_actions)
        if turn_angle_deg is None:
            lines.append(f"{a.option_id}: {seq}")
        else:
            yaw = _yaw_deg(a, float(turn_angle_deg))
            lines.append(f"{a.option_id}: {seq} (yaw {yaw:+.0f}°)")
    return lines
