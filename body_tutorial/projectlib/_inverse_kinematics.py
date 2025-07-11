"""
Inverse kinematics functionality, adapted from
https://github.com/TuragaLab/flybody/blob/e1a6135c310c39291f4fb68d682f2fd0b05e0555/flybody/inverse_kinematics.py.
"""

from copy import deepcopy
from typing import Sequence

import mujoco as mj
import numpy as np

__all__ = ["PoseOptimizer"]


class PoseOptimizer:
    def __init__(
        self,
        model: object,
        sim_state: object,
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
            [model.site(sn).id for sn in site_names],  # type: ignore
        )
        self._target_pos = np.array(target_pos, sim_state.qpos.dtype)  # type: ignore
        self._reg_coef = reg_coef
        self._grad_ema = np.zeros(model.nv, sim_state.qpos.dtype)  # type: ignore

    def loss(self) -> float:
        """
        Return the value of the loss function.
        """
        hjqpi = self._hinge_joint_qpos_indices
        site_pos = self.sim_state.site_xpos[self._site_ids]  # type: ignore
        error_loss = np.sum(np.square(site_pos - self._target_pos))
        ext_loss = self._reg_coef * np.sum(np.square(self.sim_state.qpos[hjqpi]))  # type: ignore
        return error_loss + ext_loss

    def step(self, learning_rate: float, momentum_coef: float = 0.0) -> None:
        """
        Take an optimization step.
        """
        # Define shorthands.
        mj_dtype: np.dtype = self.sim_state.qpos.dtype  # type: ignore
        nv: int = self.model.nv  # type: ignore
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
        site_pos = self.sim_state.site_xpos[self._site_ids]  # type: ignore
        error_loss_grad = 2.0 * ((site_pos - self._target_pos).flatten() @ jacobian)

        # Compute the gradient of the joint extension loss.
        ext_loss_grad = np.zeros(nv, mj_dtype)
        ext_loss_grad[hjqvi] = 2.0 * (self._reg_coef * self.sim_state.qpos[hjqpi])  # type: ignore

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
        offset: int = self.model.joint(joint_name).qposadr[0]  # type: ignore
        size = len(self.sim_state.joint(joint_name).qpos)  # type: ignore
        return np.arange(offset, offset + size)

    def _get_qvel_indices(self, joint_name: str) -> np.ndarray:
        offset: int = self.model.joint(joint_name).dofadr[0]  # type: ignore
        size = len(self.sim_state.joint(joint_name).qvel)  # type: ignore
        return np.arange(offset, offset + size)
