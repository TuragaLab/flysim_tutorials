#!/usr/bin/env python3
"""
Test the sensing logic in `sensing.py`.
"""
import sys
from typing import cast, Any

import numpy as np
from flybody.fly_envs import walk_imitation

sys.path.append(".")
from sensing import SensorSuite


def main() -> None:
    env = cast(Any, walk_imitation())
    sensor_suite = SensorSuite(env.physics.model._model)

    env.reset()

    env.task.after_substep = lambda physics, _: sensor_suite.update_state(physics.data._data)
    timestep = env.step(action=np.random.randn(env.physics.model.na))
    dm_control_observation = timestep.observation
    mj_observation = sensor_suite.read()

    print(
        "Comparing the `walk_imitation` observations to observations gathered "
        "using the `sensing` module..."
    )

    for buffer_name, dmc_buffer in dm_control_observation.items():
        if buffer_name in mj_observation:
            mj_buffer = mj_observation[buffer_name]
            if np.allclose(dmc_buffer, mj_buffer, rtol=1e-5, atol=1e-7):
                print(f"`{buffer_name}` matches")
            else:
                print(f"`{buffer_name}` does not match")
                print(f"  Walk imitation value: {dmc_buffer}")
                print(f"  Sensing module value: {mj_buffer}")
        else:
            print(f"`{buffer_name}` (shape {dmc_buffer.shape}) is not present")


if __name__ == "__main__":
    main()
