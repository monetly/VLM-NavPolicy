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
    origin_xy_norm: Tuple[float, float] = (0.5, 1.0) # Normalized (x,y) for starting point


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
        origin_xy_norm=(0.5, 1.0),
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
    camera_height_m: float = 0.6,
    hfov_deg: float = 79.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Project macro-action origin and endpoint into perspective horizontal coordinate."""
    hfov_deg = float(min(max(hfov_deg, 1.0), 179.0))
    yaw_deg = min(max(float(turn_steps) * float(turn_angle_deg), -89.0), 89.0)
    yaw_rad = math.radians(yaw_deg)

    half_hfov_tan = max(1e-6, math.tan(math.radians(hfov_deg * 0.5)))
    tip_x = 0.5 + 0.5 * (math.tan(yaw_rad) / half_hfov_tan)

    # Calculate origin footprint based on camera height and FOV
    # If the camera points straight ahead, distance=0 means origin is straight down.
    # On the projection plane (y=0.5 relates to horizon, y=1.0 relates to bottom edge ray).
    # The bottom ray has declination = vfov / 2.
    # An object at ground distance Z has declination angle theta = atan(camera_height / Z).
    # y_norm = 0.5 + 0.5 * (tan(theta) / tan(vfov/2))
    # For Z=0 (the origin), tan(theta) -> infinity, so origin_y -> infinity.
    # To cap it visually, we place the origin slightly off-screen (e.g. y=1.1 to 1.3).
    # We calibrate it against an assumed minimum visible distance.
    origin_y = 1.15
    origin_x = 0.5

    # A 0.3m or 0.4m step at camera height 0.6m is entirely below the bottom edge of the image
    # (The nearest visible ground is ~0.97m away).
    # To provide the VLM with meaningful visual trajectories, we project a "conceptual" ray 
    # that is longer than the physical step, e.g., 2.5 meters, into the image.
    travel_m = max(0.0, float(forward_steps) * float(forward_step_m))
    # visual_ray_length_m = max(0.2, travel_m * 3.0)
    visual_ray_length_m = max(0.2, travel_m * 1.0)

    if visual_ray_length_m > 0.0:
        # Distance to tip in depth Z
        z_tip = visual_ray_length_m * math.cos(yaw_rad)
        if z_tip > 1e-4:
            theta_tip_tan = camera_height_m / z_tip
            # Assuming square pixels, vfov relates to hfov by aspect ratio.
            # But normally we can approximate half_vfov_tan ~ half_hfov_tan * 0.75 (for 4:3)
            half_vfov_tan = half_hfov_tan * 0.75
            tip_y_proj = 0.5 + 0.5 * (theta_tip_tan / half_vfov_tan)
            tip_y = float(tip_y_proj)
        else:
            tip_y = origin_y
    else:
        tip_y = origin_y

    return (
        # origin (x, y)
        (float(origin_x), float(origin_y)),
        # tip (x, y)
        (float(min(max(tip_x, 0.02), 0.98)), float(max(0.05, tip_y)))
    )


def recompute_tips(
    actions: Iterable[MacroAction],
    turn_angle_deg: float,
    forward_step_m: float,
    camera_height_m: float = 0.6,
    hfov_deg: float = 79.0,
) -> Tuple[MacroAction, ...]:
    """Rebuild actions with x-tip points conforming strictly to physical projection."""
    result = []
    for a in actions:
        origin_xy, tip_xy = _tip_xy_from_motion(
            _turn_steps(a.primitive_actions),
            _forward_steps(a.primitive_actions),
            turn_angle_deg, forward_step_m, camera_height_m, hfov_deg,
        )
        result.append(
            MacroAction(
                option_id=a.option_id,
                action_name=a.action_name,
                primitive_actions=a.primitive_actions,
                tip_xy_norm=tip_xy,
                origin_xy_norm=origin_xy,
            )
        )
    return tuple(result)


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
    macro = OPTION_TO_MACRO.get(str(option_id).strip())
    if macro is None:
        raise KeyError(f"Unknown option id: {option_id!r}")
    return macro.primitive_actions


def option_to_habitat(option_id: str) -> Tuple[str, ...]:
    """Resolve lettered option id → Habitat action name sequence."""
    return to_habitat_primitives(option_to_primitives(option_id))
