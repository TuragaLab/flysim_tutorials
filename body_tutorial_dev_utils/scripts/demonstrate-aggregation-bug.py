#!/usr/bin/env python3
"""
Provide evidence of a bug in the `dm_control` library's sensor aggregation
logic.
"""
import numpy as np
from flybody.fly_envs import walk_imitation


class MeanAggregatingSensor:
    def __init__(self, sensor_name: str) -> None:
        self._sensor_name = sensor_name
        self._snapshots = list[np.ndarray]()

    def reset(self) -> None:
        self._snapshots.clear()

    def update_state(self, sim_state: object) -> None:
        mj_sensor = sim_state.sensor(self._sensor_name)  # type: ignore
        self._snapshots.append(mj_sensor.data.copy())

    def read(self) -> np.ndarray:
        assert len(self._snapshots) > 0
        return np.mean(self._snapshots, axis=0)


def main() -> None:
    env = walk_imitation()
    sensor_name = "walker/accelerometer"
    sensor = MeanAggregatingSensor(sensor_name)

    env.reset()

    while env.physics.data.time < 0.01:  # type: ignore
        print(f"\nTimestep starting at t={env.physics.data.time:.04f}:")  # type: ignore

        sensor.reset()
        env.task.after_substep = lambda physics, _: sensor.update_state(physics.data)
        timestep = env.step(action=np.zeros(env.action_spec().shape[0]))

        obs_buf = env._observation_updater._enabled_structure[sensor_name].buffer  # type: ignore
        step_counter = env._observation_updater._step_counter
        dm_control_substep_measurements = obs_buf.read(step_counter)

        print("  dm_control substep measurements:")
        for obs in dm_control_substep_measurements:
            print(f"    {obs}")

        print("  Reimplementation substep measurements:")
        for obs in sensor._snapshots:
            print(f"    {obs}")

        assert np.array_equal(
            timestep.observation[sensor_name],
            np.mean(dm_control_substep_measurements, axis=0),
        )


if __name__ == "__main__":
    main()
