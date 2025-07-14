"""
Inverse kinematics functionality, adapted from
https://github.com/TuragaLab/flybody/blob/e1a6135c310c39291f4fb68d682f2fd0b05e0555/flybody/inverse_kinematics.py.
"""

from copy import deepcopy
from itertools import product
from typing import Sequence

import mujoco as mj
import numpy as np
from mujoco import MjData, MjModel, MjSpec  # type: ignore

__all__ = ["PoseOptimizer", "add_keypoint_sites", "add_target_position_sites"]


class PoseOptimizer:
    """
    An object that can be used to iteratively move joints in a Mujoco simulation
    bring sites of interest closer to corresponding targets.
    """

    def __init__(
        self,
        model: MjModel,
        sim_state: MjData,
        joint_names: Sequence[str],
        site_names: Sequence[str],
        target_pos: np.ndarray,
        reg_coef: float,
    ) -> None:
        self.model = model
        self.sim_state = deepcopy(sim_state)
        self._joint_qvel_indices = np.concat(
            [self._get_qvel_indices(jn) for jn in joint_names]
        )
        self._hinge_joint_qpos_indices = np.concat([
            self._get_qpos_indices(jn)
            for jn in joint_names
            if model.joint(jn).type == mj.mjtJoint.mjJNT_HINGE  # type: ignore
        ])
        self._hinge_joint_qvel_indices = np.concat([
            self._get_qvel_indices(jn)
            for jn in joint_names
            if model.joint(jn).type == mj.mjtJoint.mjJNT_HINGE  # type: ignore
        ])
        self._site_ids = np.array(
            [model.site(sn).id for sn in site_names],
        )
        self._target_pos = np.array(target_pos, sim_state.qpos.dtype)
        self._reg_coef = reg_coef
        self._grad_ema = np.zeros(model.nv, sim_state.qpos.dtype)

    def loss(self) -> float:
        """
        Return the value of the loss function.
        """
        hjqpi = self._hinge_joint_qpos_indices
        site_pos = self.sim_state.site_xpos[self._site_ids]
        error_loss = np.sum(np.square(site_pos - self._target_pos))
        ext_loss = self._reg_coef * np.sum(np.square(self.sim_state.qpos[hjqpi]))
        return error_loss + ext_loss

    def step(self, learning_rate: float, momentum_coef: float = 0.0) -> None:
        """
        Take an optimization step.
        """
        # Define shorthands.
        mj_dtype: np.dtype = self.sim_state.qpos.dtype
        nv: int = self.model.nv
        jqvi = self._joint_qvel_indices
        hjqpi = self._hinge_joint_qpos_indices
        hjqvi = self._hinge_joint_qvel_indices
        mc = momentum_coef

        # Compute the full translational Jacobian, for all degrees of freedom.
        jacobian = np.empty((3 * self._target_pos.shape[0], nv), mj_dtype)
        for i, site_id in enumerate(self._site_ids):
            jacobian_slice = jacobian[3 * i : 3 * i + 3, :]
            mj.mj_jacSite(self.model, self.sim_state, jacobian_slice, None, site_id)  # type: ignore

        # Compute the gradient of the error loss.
        site_pos = self.sim_state.site_xpos[self._site_ids]
        error_loss_grad = 2.0 * ((site_pos - self._target_pos).flatten() @ jacobian)

        # Compute the gradient of the joint extension loss.
        ext_loss_grad = np.zeros(nv, mj_dtype)
        ext_loss_grad[hjqvi] = 2.0 * (self._reg_coef * self.sim_state.qpos[hjqpi])

        # Update the gradient exponential moving average.
        total_loss_grad = error_loss_grad + ext_loss_grad
        self._grad_ema = mc * self._grad_ema + (1.0 - mc) * total_loss_grad

        # Move joints.
        grad_ema_norm = np.linalg.norm(self._grad_ema[jqvi])
        clipped_norm = grad_ema_norm.clip(min=np.finfo(mj_dtype).eps)
        update = np.zeros_like(self._grad_ema)
        update[jqvi] = -learning_rate * self._grad_ema[jqvi] / clipped_norm
        mj.mj_integratePos(self.model, self.sim_state.qpos, update, 1.0)  # type: ignore
        mj.mj_fwdPosition(self.model, self.sim_state)  # type: ignore

    def _get_qpos_indices(self, joint_name: str) -> np.ndarray:
        offset = self.model.joint(joint_name).qposadr[0]
        size = len(self.sim_state.joint(joint_name).qpos)
        return np.arange(offset, offset + size)

    def _get_qvel_indices(self, joint_name: str) -> np.ndarray:
        offset = self.model.joint(joint_name).dofadr[0]
        size = len(self.sim_state.joint(joint_name).qvel)
        return np.arange(offset, offset + size)


def add_keypoint_sites(model_spec: MjSpec) -> None:
    """
    Add sites corresponding to tracked keypoints to a model.

    The sites are named "site_0", "site_1", ..., "site_35". Sites 0 through 29
    are on the legs, and sites 30 through 35 are on the body.
    """
    leg_names = ["T1_left", "T2_left", "T3_left", "T3_right", "T2_right", "T1_right"]
    leg_part_names = ["coxa", "femur", "tibia", "tarsus", "claw"]

    body_site_specs = [
        ("head", "site_30", [0.0, 0.04375, 0.00875]),  # Head
        ("abdomen_7", "site_31", [0.0, 0.039375, -0.004375]),  # Tail
        ("thorax", "site_32", [-0.0455, 0.02625, -0.00875]),  # Left haltere
        ("thorax", "site_33", [-0.0455, -0.02625, -0.00875]),  # Right haltere
        ("head", "site_34", [-0.04375, 0.016625, 0.0035]),  # Left eye
        ("head", "site_35", [0.04375, 0.016625, 0.0035]),  # Right eye
    ]

    size = (0.006, 0.006, 0.006)
    color = (0, 1, 0, 0.8)

    for i, (leg_name, part_name) in enumerate(product(leg_names, leg_part_names)):
        body = model_spec.body(f"{part_name}_{leg_name}")
        is_claw = part_name == "claw"
        pos = model_spec.site(f"claw_{leg_name}").fromto[-3:] if is_claw else [0, 0, 0]
        body.add_site(name=f"site_{i}", pos=pos, size=size, rgba=color, group=0)

    for body_name, site_name, pos in body_site_specs:
        body = model_spec.body(body_name)
        body.add_site(name=site_name, pos=pos, size=size, rgba=color, group=0)


def add_target_position_sites(model_spec: MjSpec, target_positions: np.ndarray) -> None:
    """
    Add sites to a model spec to illustrate a set of target positions.
    """
    for pos in target_positions:
        size = (0.006, 0.006, 0.006)
        color = (1.0, 0.0, 0.0, 0.8)
        model_spec.worldbody.add_site(pos=pos, size=size, rgba=color)
