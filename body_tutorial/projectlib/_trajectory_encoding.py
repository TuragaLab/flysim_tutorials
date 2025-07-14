"""
Definitions related to encoding a target trajectory to send to a controller.
"""

from typing import Callable

import numpy as np

__all__ = ["OrientationFn", "PositionFn", "encode_trajectory"]


PositionFn = Callable[
    [np.ndarray],
    tuple[
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
    ],
]
""" A callable object that accepts an array of time points and returns
    corresponding target *x*, *y*, and *z* coordinate specifiers. Each specifier
    can either be an array, if the coordinate is time-dependent, or a single
    number, if it is not. """


OrientationFn = Callable[
    [np.ndarray],
    tuple[
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
    ],
]
""" A callable object that accepts an array of time points and returns
    corresponding target *w*, *x*, *y*, and *z* quaternion component specifiers.
    Each specifier can either be an array, if the quaternion component is
    time-dependent, or a single number, if it is not. """


def encode_trajectory(
    pos_fn: PositionFn | None = None,
    ori_fn: OrientationFn | None = None,
) -> dict[str, np.ndarray]:
    """
    Return an encoding of a target trajectory to send to a controller.
    """
    t = _sampling_period() * np.arange(_n_samples()).astype(np.float32)
    pos = np.repeat(np.array([[0.0, 0.0, 0.0]], np.float32), _n_samples(), axis=0)
    ori = np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), _n_samples(), axis=0)

    if pos_fn is not None:
        x, y, z = pos_fn(t)
        pos[:, 0] = x
        pos[:, 1] = y
        pos[:, 2] = z

    if ori_fn is not None:
        w, x, y, z = ori_fn(t)
        ori[:, 0] = w
        ori[:, 1] = x
        ori[:, 2] = y
        ori[:, 3] = z

    return {"walker/ref_displacement": pos, "walker/ref_root_quat": ori}


def _n_samples() -> int:
    return 65


def _sampling_period() -> float:
    return 0.002
