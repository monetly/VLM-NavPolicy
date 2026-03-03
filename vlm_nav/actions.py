"""Action constants shared across VLM scoring, visualization, and control.

Primary action space: 5 lettered macro options A..E (forward-yaw directions).
Special actions (e.g. turn-around) are decoupled at the agent level.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Tuple, Union


@dataclass(frozen=True)
class MacroAction:
    """One candidate direction with its primitive sequence and overlay tip."""
    option_id: str  # "A", "B", "C", "D", "E"
    action_name: str
    primitive_actions: Tuple[str, ...]
    tip_xy_norm: Tuple[float, float]  # Normalized (x,y) for overlay endpoint


MACRO_ACTIONS: Tuple[MacroAction, ...] = (
    MacroAction(
        option_id="A",
        action_name="turn_left_twice_then_forward",
        primitive_actions=("turn_left", "turn_left", "move_forward"),
        tip_xy_norm=(0.12, 0.82),
    ),
    MacroAction(
        option_id="B",
        action_name="turn_left_then_forward",
        primitive_actions=("turn_left", "move_forward"),
        tip_xy_norm=(0.30, 0.58),
    ),
    MacroAction(
        option_id="C",
        action_name="forward",
        primitive_actions=("move_forward",),
        tip_xy_norm=(0.50, 0.52),
    ),
    MacroAction(
        option_id="D",
        action_name="turn_right_then_forward",
        primitive_actions=("turn_right", "move_forward"),
        tip_xy_norm=(0.70, 0.58),
    ),
    MacroAction(
        option_id="E",
        action_name="turn_right_twice_then_forward",
        primitive_actions=("turn_right", "turn_right", "move_forward"),
        tip_xy_norm=(0.88, 0.82),
    ),
)

OPTION_IDS: Tuple[str, ...] = tuple(a.option_id for a in MACRO_ACTIONS)

OPTION_TO_MACRO: Dict[str, MacroAction] = {
    a.option_id: a for a in MACRO_ACTIONS
}

NAME_TO_MACRO: Dict[str, MacroAction] = {
    a.action_name: a for a in MACRO_ACTIONS
}


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

def _turn_steps(primitives: Tuple[str, ...]) -> int:
    left = sum(1 for a in primitives if a.strip().lower() in {"turn_left", "l"})
    right = sum(1 for a in primitives if a.strip().lower() in {"turn_right", "r"})
    return right - left


def _forward_steps(primitives: Tuple[str, ...]) -> int:
    return sum(1 for a in primitives if a.strip().lower() in {"move_forward", "forward", "f"})


def _tip_xy_from_motion(
    option_id: str,
    forward_steps: int,
    forward_step_m: float,
    hfov_deg: float = 79.0,
) -> Tuple[float, float]:
    """Project macro-action endpoint into normalized image coordinates using fixed visual angles."""
    mapping = {
        "A": -60.0,
        "B": -30.0,
        "C": 0.0,
        "D": 30.0,
        "E": 60.0,
    }
    yaw_deg = mapping.get(option_id, 0.0)
    hfov_deg = float(min(max(hfov_deg, 1.0), 179.0))
    yaw_rad = math.radians(yaw_deg)

    half_hfov_tan = max(1e-6, math.tan(math.radians(hfov_deg * 0.5)))
    tip_x = 0.5 + 0.5 * (math.tan(yaw_rad) / half_hfov_tan)

    travel_m = max(0.0, float(forward_steps) * float(forward_step_m))
    if travel_m > 0.0:
        motion_scale = max(0.0, travel_m / max(1e-6, 0.30))
        base_dy = max(0.10, min(0.45, 0.28 * max(motion_scale, 0.35)))
        len_scale = min(max(math.cos(abs(yaw_rad)), 0.2), 1.0)
        tip_y = 0.999 - base_dy * len_scale
    else:
        tip_y = 0.98

    return float(min(max(tip_x, 0.02), 0.98)), float(tip_y)


def recompute_tips(
    actions: Iterable[MacroAction],
    turn_angle_deg: float,  # kept for signature compatibility, but unused for yaw
    forward_step_m: float,
    hfov_deg: float = 79.0,
) -> Tuple[MacroAction, ...]:
    """Rebuild actions with tip points aligned to fixed visual angles."""
    return tuple(
        MacroAction(
            option_id=a.option_id,
            action_name=a.action_name,
            primitive_actions=a.primitive_actions,
            tip_xy_norm=_tip_xy_from_motion(
                a.option_id,
                _forward_steps(a.primitive_actions),
                forward_step_m, hfov_deg,
            ),
        )
        for a in actions
    )


# ---------------------------------------------------------------------------
# Conversion to Habitat primitives
# ---------------------------------------------------------------------------

def to_habitat_primitives(primitive_actions: Tuple[str, ...]) -> Tuple[str, ...]:
    """Convert primitive action names to Habitat action names."""
    valid = {"move_forward", "turn_left", "turn_right"}
    result = []
    for a in primitive_actions:
        token = a.strip()
        if token not in valid:
            raise KeyError(f"Unsupported primitive action: {a!r}")
        result.append(token)
    return tuple(result)


def option_to_primitives(option_id: str) -> Tuple[str, ...]:
    """Resolve lettered option id → primitive action sequence."""
    macro = OPTION_TO_MACRO.get(str(option_id).strip().upper())
    if macro is None:
        raise KeyError(f"Unknown option id: {option_id!r}")
    return macro.primitive_actions


def option_to_habitat(option_id: str) -> Tuple[str, ...]:
    """Resolve lettered option id → Habitat action name sequence."""
    return to_habitat_primitives(option_to_primitives(option_id))
