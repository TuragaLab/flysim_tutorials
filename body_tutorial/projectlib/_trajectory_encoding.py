from functools import cache
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

OrientationFn = Callable[
    [np.ndarray],
    tuple[
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
    ],
]


def encode_trajectory(
    pos_fn: PositionFn | None = None,
    ori_fn: OrientationFn | None = None,
) -> dict[str, np.ndarray]:
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
