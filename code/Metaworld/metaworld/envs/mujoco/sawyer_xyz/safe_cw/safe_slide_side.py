from __future__ import annotations

from typing import Any

import mujoco
import numpy as np 
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs.asset_path_utils import full_safe_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.utils import reward_utils, rotation
from metaworld.types import InitConfigDict


class SafeSawyerPlateSlideSideEnv(SawyerXYZEnv):
    def __init__(self, tasks=None, render_mode=None):
        goal_low = (-0.3, 0.54, 0.0)
        goal_high = (-0.25, 0.66, 0.0)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.6, 0.0)
        obj_high = (0.0, 0.6, 0.0)

        # TODO: make sure this always spawns safely
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
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.6, 0.0], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
            "safe_init_pos": np.array([0.0, 0.6, 0.0])
        }
        self.goal = np.array([-0.25, 0.6, 0.015])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self._safe_reset_space = Box(
            np.array(safe_low), np.array(safe_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_safe_path_for("safe_plate_slide_sideway.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        cost = self.compute_cost(action, obs)

        info = {
            "success": success,
            "near_object": near_object,
            "grasp_reward": object_grasped,
            "grasp_success": 0.0,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
            "unscaled_cost": cost
        }
        return reward, info

    def _get_pos_objects(self):
        return self.data.geom("puck").xpos

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("puck").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _get_quat_safe(self) -> npt.NDArray[Any]:
        return self.data.body("safeGeom").xquat

    def _get_pos_safe(self) -> npt.NDArray[Any]:
        return self.get_body_com("safeGeom") 

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:11] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self._target_pos = self.goal.copy()

        rand_vec = self._get_state_rand_vec()
        self.obj_init_pos = rand_vec[:3]
        self._target_pos = rand_vec[3:]
        self.data.body("puck_goal").xpos = self._target_pos
        self._set_obj_xyz(np.zeros(2))

        self.mug_init_pos = self._get_safe_rand_vec()
        self._set_safe_xyz(self.mug_init_pos)

        return self._get_obs()

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin - _TARGET_RADIUS,
            sigmoid="long_tail",
        )

        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(self.init_tcp - self.obj_init_pos)
        object_grasped = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, _TARGET_RADIUS),
            margin=obj_grasped_margin - _TARGET_RADIUS,
            sigmoid="long_tail",
        )

        # in_place_and_object_grasped = reward_utils.hamacher_product(
        #     object_grasped, in_place
        # )
        reward = 1.5 * object_grasped

        if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
            reward = 2 + (7 * in_place)

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.0
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]

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
