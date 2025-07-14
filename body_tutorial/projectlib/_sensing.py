"""
Definitions related to acquiring data from a Mujoco simulation to send to a
controller.
"""

from collections import deque
from fnmatch import fnmatch
from typing import Literal, Protocol, Sequence

import mujoco as mj
import numpy as np
from mujoco import MjData, MjModel  # type: ignore

__all__ = ["SensorSuite"]


class SensorSuite:
    """
    A collection of sensors operating simultaneously.
    """

    def __init__(self, model: MjModel) -> None:
        self._sensors = {
            "walker/accelerometer": _WeightedSumSensor(
                field_name="sensordata",
                indices=_sensor_data_indices(model, "accelerometer"),
                kernel="mean-sub-second-for-first",
            ),
            "walker/actuator_activation": _LastValueSensor(
                field_name="act",
                indices=np.arange(model.na),
            ),
            "walker/force": _WeightedSumSensor(
                field_name="sensordata",
                indices=_sensor_data_indices(model, "force_*"),
                kernel="mean-sub-second-for-first",
            ),
            "walker/gyro": _WeightedSumSensor(
                field_name="sensordata",
                indices=_sensor_data_indices(model, "gyro"),
                kernel="mean",
            ),
            "walker/joints_pos": _LastValueSensor(
                field_name="qpos",
                indices=np.concatenate(
                    [model.joint(name).qposadr for name in _observed_joint_names()]
                ),
            ),
            "walker/joints_vel": _LastValueSensor(
                field_name="qvel",
                indices=np.concatenate(
                    [model.joint(name).dofadr for name in _observed_joint_names()]
                ),
            ),
            "walker/touch": _WeightedSumSensor(
                field_name="sensordata",
                indices=_sensor_data_indices(model, "touch_*"),
                kernel="mean-sub-second-for-first",
            ),
            "walker/velocimeter": _WeightedSumSensor(
                field_name="sensordata",
                indices=_sensor_data_indices(model, "velocimeter"),
                kernel="mean",
            ),
            "walker/world_zaxis": _LastValueSensor(
                field_name="xmat",
                indices=_xmat_indices(model, "thorax")[6:],
            ),
            "walker/appendages_pos": _AppendagePositionSensor(
                root_body_id=model.body("thorax").id,
                appendage_ids=[model.site(name).id for name in _appendage_names()],
            ),
        }

    def update_state(self, sim_state: MjData) -> None:
        """
        Allow the sensors in the suite to capture data from the simulation.

        This should be called once per simulation step.
        """
        for sensor in self._sensors.values():
            sensor.update_state(sim_state)

    def read(self) -> dict[str, np.ndarray]:
        """
        Read data from the sensors in the suite.

        The returned dictionary will have an entry for each sensor.
        """
        return {name: sensor.read() for name, sensor in self._sensors.items()}


class _Sensor(Protocol):
    def update_state(self, sim_state: MjData) -> None: ...
    def read(self) -> np.ndarray: ...


class _LastValueSensor(_Sensor):
    def __init__(self, field_name: str, indices: np.ndarray) -> None:
        self._field_name = field_name
        self._indices = indices
        self._value: np.ndarray | None = None

    def update_state(self, sim_state: MjData) -> None:
        field = getattr(sim_state, self._field_name)
        self._value = field.flat[self._indices].astype(np.float32)

    def read(self) -> np.ndarray:
        assert self._value is not None
        return self._value


class _WeightedSumSensor(_Sensor):
    def __init__(
        self,
        field_name: str,
        indices: np.ndarray | Sequence[int],
        kernel: Literal["mean", "mean-sub-second-for-first"],
    ) -> None:
        self._field_name = field_name
        self._indices = np.array(indices)
        self._kernel = kernel
        self._snapshots = deque[np.ndarray]()
        self._snapshot_timestamps = deque[float]()

    def update_state(self, sim_state: MjData) -> None:
        while (
            len(self._snapshot_timestamps) > 0
            and self._snapshot_timestamps[0] < sim_state.time - _averaging_interval()
        ):
            self._snapshots.popleft()
            self._snapshot_timestamps.popleft()

        field = getattr(sim_state, self._field_name)
        self._snapshots.append(field.flat[self._indices])
        self._snapshot_timestamps.append(sim_state.time)

    def read(self) -> np.ndarray:
        assert len(self._snapshots) >= 1

        if self._kernel == "mean" or len(self._snapshots) == 1:
            return np.mean(self._snapshots, axis=0).astype(np.float32)
        else:
            mean_inputs = [self._snapshots[1], *list(self._snapshots)[1:]]
            return np.mean(mean_inputs, axis=0).astype(np.float32)


class _AppendagePositionSensor(_Sensor):
    def __init__(self, root_body_id: int, appendage_ids: Sequence[int]) -> None:
        self._root_body_id = root_body_id
        self._appendage_ids = appendage_ids
        self._value: np.ndarray | None = None

    def update_state(self, sim_state: MjData) -> None:
        appendage_pos = sim_state.site_xpos[self._appendage_ids, :]
        root_pos = sim_state.xpos[self._root_body_id, :]
        root_mat = sim_state.xmat[self._root_body_id, :].reshape(3, 3)
        self._value = ((appendage_pos - root_pos) @ root_mat).flatten()

    def read(self) -> np.ndarray:
        assert self._value is not None
        return self._value.astype(np.float32)


def _sensor_data_indices(model: MjModel, pattern: str) -> np.ndarray:
    sensor_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)  # type: ignore
        for i in range(model.nsensor)
    ]
    matching_sensors = [
        model.sensor(name)
        for name in sensor_names
        if isinstance(name, str) and fnmatch(name, pattern)
    ]
    return np.concatenate([
        np.arange(sensor.adr[0], sensor.adr[0] + sensor.dim[0])
        for sensor in matching_sensors
    ])


def _xmat_indices(model: MjModel, pattern: str) -> np.ndarray:
    body_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)  # type: ignore
        for i in range(model.nbody)
    ]
    matching_bodies = [
        model.body(name)
        for name in body_names
        if isinstance(name, str) and fnmatch(name, pattern)
    ]
    return np.concatenate(
        [np.arange(9 * body.id, 9 * body.id + 9) for body in matching_bodies]
    )


def _observed_joint_names() -> list[str]:
    return [
        "head_abduct",
        "head_twist",
        "head",
        "abdomen_abduct",
        "abdomen",
        "abdomen_abduct_2",
        "abdomen_2",
        "abdomen_abduct_3",
        "abdomen_3",
        "abdomen_abduct_4",
        "abdomen_4",
        "abdomen_abduct_5",
        "abdomen_5",
        "abdomen_abduct_6",
        "abdomen_6",
        "abdomen_abduct_7",
        "abdomen_7",
        "haltere_left",
        "haltere_right",
        "coxa_abduct_T1_left",
        "coxa_twist_T1_left",
        "coxa_T1_left",
        "femur_twist_T1_left",
        "femur_T1_left",
        "tibia_T1_left",
        "tarsus_T1_left",
        "tarsus2_T1_left",
        "tarsus3_T1_left",
        "tarsus4_T1_left",
        "tarsus5_T1_left",
        "coxa_abduct_T1_right",
        "coxa_twist_T1_right",
        "coxa_T1_right",
        "femur_twist_T1_right",
        "femur_T1_right",
        "tibia_T1_right",
        "tarsus_T1_right",
        "tarsus2_T1_right",
        "tarsus3_T1_right",
        "tarsus4_T1_right",
        "tarsus5_T1_right",
        "coxa_abduct_T2_left",
        "coxa_twist_T2_left",
        "coxa_T2_left",
        "femur_twist_T2_left",
        "femur_T2_left",
        "tibia_T2_left",
        "tarsus_T2_left",
        "tarsus2_T2_left",
        "tarsus3_T2_left",
        "tarsus4_T2_left",
        "tarsus5_T2_left",
        "coxa_abduct_T2_right",
        "coxa_twist_T2_right",
        "coxa_T2_right",
        "femur_twist_T2_right",
        "femur_T2_right",
        "tibia_T2_right",
        "tarsus_T2_right",
        "tarsus2_T2_right",
        "tarsus3_T2_right",
        "tarsus4_T2_right",
        "tarsus5_T2_right",
        "coxa_abduct_T3_left",
        "coxa_twist_T3_left",
        "coxa_T3_left",
        "femur_twist_T3_left",
        "femur_T3_left",
        "tibia_T3_left",
        "tarsus_T3_left",
        "tarsus2_T3_left",
        "tarsus3_T3_left",
        "tarsus4_T3_left",
        "tarsus5_T3_left",
        "coxa_abduct_T3_right",
        "coxa_twist_T3_right",
        "coxa_T3_right",
        "femur_twist_T3_right",
        "femur_T3_right",
        "tibia_T3_right",
        "tarsus_T3_right",
        "tarsus2_T3_right",
        "tarsus3_T3_right",
        "tarsus4_T3_right",
        "tarsus5_T3_right",
    ]


def _appendage_names() -> list[str]:
    return [
        "claw_T1_left",
        "claw_T1_right",
        "claw_T2_left",
        "claw_T2_right",
        "claw_T3_left",
        "claw_T3_right",
        "head",
    ]


def _averaging_interval() -> float:
    return 0.002
