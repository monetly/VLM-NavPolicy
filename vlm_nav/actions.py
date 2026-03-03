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
    turn_steps: int,
    forward_steps: int,
    turn_angle_deg: float,
    forward_step_m: float,
    hfov_deg: float = 79.0,
) -> Tuple[float, float]:
    """Project macro-action endpoint into perspective horizontal coordinate."""
    hfov_deg = float(min(max(hfov_deg, 1.0), 179.0))
    yaw_deg = min(max(float(turn_steps) * float(turn_angle_deg), -89.0), 89.0)
    yaw_rad = math.radians(yaw_deg)

    half_hfov_tan = max(1e-6, math.tan(math.radians(hfov_deg * 0.5)))
    tip_x = 0.5 + 0.5 * (math.tan(yaw_rad) / half_hfov_tan)

    # We only care about x here; ground_overlay determines dynamic tip_y (length).
    return float(min(max(tip_x, 0.02), 0.98)), 0.0


def recompute_tips(
    actions: Iterable[MacroAction],
    turn_angle_deg: float,
    forward_step_m: float,
    hfov_deg: float = 79.0,
) -> Tuple[MacroAction, ...]:
    """Rebuild actions with x-tip points conforming strictly to physical projection."""
    return tuple(
        MacroAction(
            option_id=a.option_id,
            action_name=a.action_name,
            primitive_actions=a.primitive_actions,
            tip_xy_norm=_tip_xy_from_motion(
                _turn_steps(a.primitive_actions),
                _forward_steps(a.primitive_actions),
                turn_angle_deg, forward_step_m, hfov_deg,
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
