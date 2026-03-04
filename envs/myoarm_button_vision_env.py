"""
MyoArm Phase 2: Vision-Only Button-Press Environment
======================================================
Same task as Phase 1, but observation = stacked RGB camera frames.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional

from envs.myoarm_button_proprio_env import HAND_GEOMS


class MyoArmButtonPressVisionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        camera_name: str = "static_cam",
        image_size: tuple = (64, 64),
        frame_stack: int = 3,
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

        # --- Renderer ---
        self.image_size = image_size
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(self.model, image_size[0], image_size[1])
        self.render_mode = render_mode

        # --- Spaces ---
        self.frame_stack = frame_stack
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, image_size[0], image_size[1], 3),
            dtype=np.uint8,
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
        self.frames = deque(maxlen=frame_stack)
        self.randomize_button = randomize_button
        self.button_range_x = button_range_x
        self.button_range_y = button_range_y

        # --- Cache IDs ---
        self._hand_geom_ids = []
        for name in HAND_GEOMS:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._hand_geom_ids.append(gid)

        self._btn_target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "btn_target")
        self._btn_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "button_panel")
        self._original_btn_pos = self.model.body_pos[self._btn_body_id].copy()
        self._np_random = np.random.default_rng()

        # --- Hires renderer for visualization ---
        self._hires_renderer = None

    def _get_min_hand_distance(self):
        btn_pos = self.data.site_xpos[self._btn_target_id]
        min_dist = float("inf")
        for gid in self._hand_geom_ids:
            dist = np.linalg.norm(self.data.geom_xpos[gid] - btn_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _check_press(self):
        min_dist = self._get_min_hand_distance()
        pressed = min_dist < 0.018
        return pressed, min_dist

    def _render_frame(self):
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return self.renderer.render().copy()

    def _randomize_button(self):
        if not self.randomize_button:
            return
        x = self._np_random.uniform(*self.button_range_x)
        y = self._np_random.uniform(*self.button_range_y)
        z = self._original_btn_pos[2]
        self.model.body_pos[self._btn_body_id] = [x, y, z]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._randomize_button()
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.frames.clear()

        frame = self._render_frame()
        for _ in range(self.frame_stack):
            self.frames.append(frame)

        obs = np.stack(list(self.frames), axis=0)
        info = {"btn_pos": self.data.site_xpos[self._btn_target_id].copy()}
        return obs, info

    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        mujoco.mj_forward(self.model, self.data)

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

        terminated = False
        truncated = False
        info = {"pressed": pressed, "distance": min_dist, "success": False}

        if pressed:
            terminated = True
            info["success"] = True
        elif self.step_count >= self.max_steps:
            truncated = True

        frame = self._render_frame()
        self.frames.append(frame)
        obs = np.stack(list(self.frames), axis=0)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._hires_renderer is None:
                self._hires_renderer = mujoco.Renderer(self.model, 480, 480)
            self._hires_renderer.update_scene(self.data, camera=self.camera_name)
            return self._hires_renderer.render().copy()
        return None

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()
        if self._hires_renderer is not None:
            self._hires_renderer.close()
        super().close()
