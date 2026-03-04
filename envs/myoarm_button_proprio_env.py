"""
MyoArm Phase 1: Proprioceptive Button-Press Environment
=========================================================
Full arm (32 muscles, ~38 joints) learns to press a randomly
placed button on a table using state vector observations.

Observation vector:
  - fingertip positions (5 tips × 3 = 15)
  - button position xyz (3)
  - distance fingertip-to-button (1, min of 5 tips)
  - joint angles for key joints (elbow + wrist = 3)
  - button displacement (1)
  - button touch force (1)
  Total: 24

Action: 32 muscle activations [0, 1]
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


# Fingertip site names in the MyoArm model
FINGERTIP_SITES = ["THtip", "IFtip", "MFtip", "RFtip", "LFtip"]

# Hand geom names for proximity-based press detection
HAND_GEOMS = [
    "thumbdist", "2distph", "3distph", "4distph", "5distph",
    "thumbprox", "2midph", "3midph", "2proxph", "3proxph",
    "2mc", "3mc",
]


class MyoArmButtonPressProprioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        max_steps: int = 300,
        reward_type: str = "staged",
        randomize_button: bool = True,
        button_range_x: tuple = (-0.31, -0.15),
        button_range_y: tuple = (-0.03, 0.08),
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"
        self.xml_path = xml_path

        # --- MuJoCo ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- Spaces ---
        self.obs_dim = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float64,
        )
        n_muscles = self.model.nu
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_muscles,), dtype=np.float32,
        )

        # --- Episode ---
        self.max_steps = max_steps
        self.step_count = 0
        self.n_substeps = 5
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.randomize_button = randomize_button
        self.button_range_x = button_range_x
        self.button_range_y = button_range_y

        # --- Cache IDs ---
        self._tip_site_ids = []
        for name in FINGERTIP_SITES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            self._tip_site_ids.append(sid)

        self._hand_geom_ids = []
        for name in HAND_GEOMS:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._hand_geom_ids.append(gid)

        self._btn_target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "btn_target")
        self._btn_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "button_panel")
        self._btn_pos_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "btn_pos")
        self._btn_touch_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "btn_touch")

        # Key joint IDs
        self._elbow_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
        self._deviation_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "deviation")
        self._flexion_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "flexion")

        # --- Store defaults ---
        self._original_btn_pos = self.model.body_pos[self._btn_body_id].copy()
        self._np_random = np.random.default_rng()

        # --- Renderer ---
        self._renderer = None

        # --- Stats ---
        self.episode_count = 0
        self.success_count = 0

    def _get_fingertip_positions(self):
        """Get all 5 fingertip positions (15-dim)."""
        positions = []
        for sid in self._tip_site_ids:
            positions.append(self.data.site_xpos[sid].copy())
        return np.concatenate(positions)

    def _get_min_hand_distance(self):
        """Get minimum distance from any hand geom to button."""
        btn_pos = self.data.site_xpos[self._btn_target_id]
        min_dist = float("inf")
        for gid in self._hand_geom_ids:
            geom_pos = self.data.geom_xpos[gid]
            dist = np.linalg.norm(geom_pos - btn_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _check_press(self):
        """Proximity-based press detection."""
        min_dist = self._get_min_hand_distance()
        pressed = min_dist < 0.018
        return pressed, min_dist

    def _get_obs(self):
        tips = self._get_fingertip_positions()  # 15
        btn_pos = self.data.site_xpos[self._btn_target_id].copy()  # 3
        min_dist = self._get_min_hand_distance()  # 1

        # Key joint angles
        elbow_q = self.data.qpos[self.model.jnt_qposadr[self._elbow_jnt]] if self._elbow_jnt >= 0 else 0.0
        dev_q = self.data.qpos[self.model.jnt_qposadr[self._deviation_jnt]] if self._deviation_jnt >= 0 else 0.0
        flex_q = self.data.qpos[self.model.jnt_qposadr[self._flexion_jnt]] if self._flexion_jnt >= 0 else 0.0

        btn_disp = self.data.sensordata[self._btn_pos_sensor]
        btn_force = self.data.sensordata[self._btn_touch_sensor]

        return np.concatenate([
            tips,                   # 15
            btn_pos,                # 3
            [min_dist],             # 1
            [elbow_q, dev_q, flex_q],  # 3
            [btn_disp],             # 1
            [btn_force],            # 1
        ])                          # total: 24

    def _randomize_button(self):
        """Place button at random position on table."""
        if not self.randomize_button:
            return

        x = self._np_random.uniform(*self.button_range_x)
        y = self._np_random.uniform(*self.button_range_y)
        z = self._original_btn_pos[2]  # keep height fixed (table surface)

        self.model.body_pos[self._btn_body_id] = [x, y, z]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._randomize_button()
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.episode_count += 1

        obs = self._get_obs()
        info = {"btn_pos": self.data.site_xpos[self._btn_target_id].copy()}
        return obs, info

    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        pressed, min_dist = self._check_press()

        # --- Reward ---
        if self.reward_type == "staged":
            r_reach = -5.0 * min_dist
            r_prox = 10.0 if min_dist < 0.05 else (5.0 if min_dist < 0.10 else (2.0 if min_dist < 0.15 else 0.0))
            r_press = 100.0 if pressed else 0.0
            r_effort = -0.002 * np.sum(action ** 2)
            reward = r_reach + r_prox + r_press + r_effort
        else:
            reward = 100.0 if pressed else -0.01

        # --- Termination ---
        terminated = False
        truncated = False
        info = {
            "pressed": pressed,
            "distance": min_dist,
            "success": False,
            "btn_pos": self.data.site_xpos[self._btn_target_id].copy(),
        }

        if pressed:
            terminated = True
            info["success"] = True
            self.success_count += 1
        elif self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 480)
            self._renderer.update_scene(self.data, camera="static_cam")
            return self._renderer.render().copy()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        super().close()
