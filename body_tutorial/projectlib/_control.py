from fnmatch import fnmatch
from functools import cache

import mujoco as mj
import numpy as np
from mujoco import MjModel, MjSpec  # type: ignore

__all__ = [
    "add_actuator_filtering",
    "disable_actuators",
    "get_walking_actuator_indices",
    "pack_controller_input",
    "to_control_range",
]


def disable_actuators(model_spec: MjSpec, pattern: str) -> None:
    for actuator in model_spec.actuators:
        if any(fnmatch(actuator.name, subpat) for subpat in pattern.split("|")):
            actuator.delete()


def add_actuator_filtering(model_spec: MjSpec) -> None:
    for actuator in model_spec.actuators:
        actuator.dyntype = mj.mjtDyn.mjDYN_FILTER  # type: ignore
        actuator.dynprm[0] = 0.007 if actuator.name.startswith("adhere") else 0.01


def get_walking_actuator_indices(model: MjModel) -> np.ndarray:
    return np.concat([
        model.actuator(segment_name).actadr
        for segment_name in _controller_output_layout()
    ])


def pack_controller_input(
    sensor_readings: dict[str, np.ndarray],
    target_trajectory: dict[str, np.ndarray],
) -> np.ndarray:
    buffers_by_name = {**sensor_readings, **target_trajectory}
    return np.concat([
        buffers_by_name[segment_name].flat
        for segment_name in _controller_input_layout()
    ])


def to_control_range(
    model: MjModel,
    actuator_indices: np.ndarray,
    control_signals: np.ndarray,
) -> np.ndarray:
    minima, maxima = model.actuator_ctrlrange[actuator_indices, :].T  # type: ignore
    control_signals = 0.5 + 0.5 * control_signals.clip(-1.0, 1.0)
    control_signals *= maxima - minima
    control_signals += minima
    return control_signals


@cache
def _controller_input_layout() -> list[str]:
    return [
        "walker/accelerometer",
        "walker/actuator_activation",
        "walker/appendages_pos",
        "walker/force",
        "walker/gyro",
        "walker/joints_pos",
        "walker/joints_vel",
        "walker/ref_displacement",
        "walker/ref_root_quat",
        "walker/touch",
        "walker/velocimeter",
        "walker/world_zaxis",
    ]


@cache
def _controller_output_layout() -> list[str]:
    return [
        "adhere_claw_T1_left",
        "adhere_claw_T1_right",
        "adhere_claw_T2_left",
        "adhere_claw_T2_right",
        "adhere_claw_T3_left",
        "adhere_claw_T3_right",
        "head_abduct",
        "head_twist",
        "head",
        "abdomen_abduct",
        "abdomen",
        "coxa_abduct_T1_left",
        "coxa_twist_T1_left",
        "coxa_T1_left",
        "femur_twist_T1_left",
        "femur_T1_left",
        "tibia_T1_left",
        "tarsus_T1_left",
        "tarsus2_T1_left",
        "coxa_abduct_T1_right",
        "coxa_twist_T1_right",
        "coxa_T1_right",
        "femur_twist_T1_right",
        "femur_T1_right",
        "tibia_T1_right",
        "tarsus_T1_right",
        "tarsus2_T1_right",
        "coxa_abduct_T2_left",
        "coxa_twist_T2_left",
        "coxa_T2_left",
        "femur_twist_T2_left",
        "femur_T2_left",
        "tibia_T2_left",
        "tarsus_T2_left",
        "tarsus2_T2_left",
        "coxa_abduct_T2_right",
        "coxa_twist_T2_right",
        "coxa_T2_right",
        "femur_twist_T2_right",
        "femur_T2_right",
        "tibia_T2_right",
        "tarsus_T2_right",
        "tarsus2_T2_right",
        "coxa_abduct_T3_left",
        "coxa_twist_T3_left",
        "coxa_T3_left",
        "femur_twist_T3_left",
        "femur_T3_left",
        "tibia_T3_left",
        "tarsus_T3_left",
        "tarsus2_T3_left",
        "coxa_abduct_T3_right",
        "coxa_twist_T3_right",
        "coxa_T3_right",
        "femur_twist_T3_right",
        "femur_T3_right",
        "tibia_T3_right",
        "tarsus_T3_right",
        "tarsus2_T3_right",
    ]
