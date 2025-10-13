from __future__ import annotations

from typing import Any

import mujoco
import numpy as np 
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.envs.asset_path_utils import full_safe_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.utils import reward_utils, rotation
from metaworld.types import InitConfigDict

class SafeSawyerHandlePullSideEnv(SawyerXYZEnv):
    def __init__(self, tasks=None, render_mode=None):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.35, 0.65, 0.0)
        obj_high = (-0.25, 0.75, 0.0)

        safe_low = (-0.15, 0.6, 0.01)
        safe_high = (0.15, 0.65, 0.01)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            safety_constrained=True
        )

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_pos": np.array([-0.3, 0.7, 0.0]),
            "hand_init_pos": np.array(
                (0, 0.6, 0.2),
            ),
            "safe_init_pos": np.array([0.0, 0.6, 0.0])
        }
        self.goal = np.array([-0.2, 0.7, 0.14])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.safe_init_pos = self.init_config["safe_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self._safe_reset_space = Box(
            np.array(safe_low), np.array(safe_high), dtype=np.float64
        )

    @property
    def model_name(self):
        return full_safe_path_for("safe_handle_pull_sideways.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        cost = self.compute_cost(action, obs)

        info = {
            "success": float(obj_to_target <= 0.08),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(
                (tcp_open > 0) and (obj[2] - 0.03 > self.obj_init_pos[2])
            ),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
            "unscaled_cost": cost
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self._get_site_pos("handleCenter")

    def _get_quat_objects(self):
        return np.zeros(4)

    def _get_quat_safe(self) -> npt.NDArray[Any]:
        return self.data.body("safeGeom").xquat

    def _get_pos_safe(self) -> npt.NDArray[Any]:
        return self.get_body_com("safeGeom") 

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = self.obj_init_pos
        self._set_obj_xyz(-0.1)
        self._target_pos = self._get_site_pos("goalPull")
        self.maxDist = np.abs(
            self.data.site_xpos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "handleStart")
            ][-1]
            - self._target_pos[-1]
        )
        self.target_reward = 1000 * self.maxDist + 1000 * 2
        self.obj_init_pos = self._get_pos_objects()

        self.mug_init_pos = self._get_safe_rand_vec()
        self._set_safe_xyz(self.mug_init_pos)

        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        # Force target to be slightly above basketball hoop
        target = self._target_pos.copy()

        # Emphasize Z error
        scale = np.array([1.0, 1.0, 1.0])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            pad_success_thresh=0.06,
            obj_radius=0.032,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            high_density=True,
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)
        # reward = in_place

        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        if (
            tcp_to_obj < 0.035
            and tcp_opened > 0
            and obj[2] - 0.01 > self.obj_init_pos[2]
        ):
            reward += 1.0 + 5.0 * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0
        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)

    def compute_cost(self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]):
        mug_quat = obs[21:25]
        mug_euler = rotation.quat2euler(mug_quat)
        tilt = np.sqrt(mug_euler[0]**2 + mug_euler[1]**2)
        self.mug_tilt = tilt
        return np.floor(np.rad2deg(tilt)) 

    def tipped_mug(self):
        return self.mug_tilt >= 90

    def step(self, action):
        obs, rew, terminated, trunc, info = super().step(action)
        return obs, rew, terminated, trunc, info
