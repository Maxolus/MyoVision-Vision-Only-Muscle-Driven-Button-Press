"""
Phase 1: Proprioceptive Button-Press Environment
==================================================
Same task, but observation = state vector instead of pixels.
The agent learns muscle coordination first, then we transfer to vision.

Observation vector (11-dim):
  - r_elbow_flex angle        (1)
  - r_elbow_flex velocity     (1)
  - wrist position xyz        (3)
  - button position xyz       (3)
  - distance wrist-to-button  (1)
  - button displacement       (1)
  - button touch force        (1)

Action: 6 muscle activations [0, 1]
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, List


class ButtonPressProprioEnv(gym.Env):
    """
    Proprioceptive button-press environment.
    Observation: compact state vector (no vision).
    Action: muscle activations [0, 1]^6
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        max_steps: int = 500,
        reward_type: str = "staged",
        randomization_level: str = "none",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # --- Paths ---
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "assets", "elbow", "myoelbow_buttonpress.xml")
        self.xml_path = xml_path

        # --- MuJoCo model ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- Spaces ---
        # Obs: elbow angle, elbow vel, wrist xyz, button xyz, distance, btn_disp, btn_force
        self.obs_dim = 11
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float64,
        )
        n_muscles = self.model.nu
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_muscles,), dtype=np.float32,
        )

        # --- Episode state ---
        self.max_steps = max_steps
        self.step_count = 0
        self.dt = self.model.opt.timestep
        self.n_substeps = 5
        self.reward_type = reward_type
        self.render_mode = render_mode

        # --- Cache IDs ---
        self._elbow_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "r_elbow_flex")
        self._elbow_qpos_addr = self.model.jnt_qposadr[self._elbow_jnt_id]
        self._elbow_qvel_addr = self.model.jnt_dofadr[self._elbow_jnt_id]
        self._wrist_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "wrist")
        self._btn_target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "btn_target")
        self._btn_pos_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "btn_pos")
        self._btn_touch_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "btn_touch")
        self._btn_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "button_panel")

        # --- Randomization ---
        self._original_btn_pos = self.model.body_pos[self._btn_body_id].copy()
        self._original_muscle_force = self.model.actuator_gainprm[:, 0].copy()
        self.randomization_level = randomization_level
        self._np_random = np.random.default_rng()

        # --- Renderer (for visualization only) ---
        self._renderer = None

        # --- Logging ---
        self.episode_reward = 0.0
        self.success_count = 0
        self.episode_count = 0

    def _get_obs(self):
        """Build compact state vector."""
        elbow_angle = self.data.qpos[self._elbow_qpos_addr]
        elbow_vel = self.data.qvel[self._elbow_qvel_addr]
        wrist_pos = self.data.site_xpos[self._wrist_site_id].copy()
        btn_pos = self.data.site_xpos[self._btn_target_site_id].copy()
        distance = np.linalg.norm(wrist_pos - btn_pos)
        btn_disp = self.data.sensordata[self._btn_pos_sensor]
        btn_force = self.data.sensordata[self._btn_touch_sensor]

        return np.concatenate([
            [elbow_angle],          # 1
            [elbow_vel],            # 1
            wrist_pos,              # 3
            btn_pos,                # 3
            [distance],             # 1
            [btn_disp],             # 1
            [btn_force],            # 1
        ])                          # total: 11

    def _check_press(self):
        """Check if button is pressed via proximity of nearest hand geom."""
        btn_pos = self.data.site_xpos[self._btn_target_site_id]
        
        # Check distance of hand geoms to button
        min_dist = float("inf")
        hand_geoms = ["arm_r_2mc", "arm_r_3mc", "arm_r_1mc", 
                      "arm_r_thumbdist", "arm_r_capitate", "arm_r_hamate"]
        
        for name in hand_geoms:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                geom_pos = self.data.geom_xpos[gid]
                dist = np.linalg.norm(geom_pos - btn_pos)
                if dist < min_dist:
                    min_dist = dist
        
        # Press = hand geom within 1.5 cm of button center
        pressed = min_dist < 0.025
        return pressed, min_dist, 1.0 if pressed else 0.0

    def _randomize(self):
        """Apply domain randomization."""
        if self.randomization_level == "none":
            return

        if self.randomization_level in ("medium", "high"):
            offset = self._np_random.uniform([-0.03, -0.03, -0.02],
                                              [0.03,  0.03,  0.02])
            self.model.body_pos[self._btn_body_id] = self._original_btn_pos + offset

        if self.randomization_level == "high":
            scale = self._np_random.uniform(0.8, 1.2, size=self._original_muscle_force.shape)
            self.model.actuator_gainprm[:, 0] = self._original_muscle_force * scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._randomize()
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_count += 1

        obs = self._get_obs()
        info = {
            "wrist_pos": self.data.site_xpos[self._wrist_site_id].copy(),
            "btn_pos": self.data.site_xpos[self._btn_target_site_id].copy(),
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        pressed, btn_disp, btn_force = self._check_press()

        # --- Reward ---
        wrist_pos = self.data.site_xpos[self._wrist_site_id]
        btn_pos = self.data.site_xpos[self._btn_target_site_id]
        dist = np.linalg.norm(wrist_pos - btn_pos)

        if self.reward_type == "staged":
            r_reach = -1.0 * dist
            r_prox = 3.0 if dist < 0.03 else (1.0 if dist < 0.08 else 0.0)
            r_press = 50.0 if pressed else 0.0
            r_effort = -0.005 * np.sum(action ** 2)
            r_overshoot = -0.1 * max(0.0, btn_force - 1.0) if pressed else 0.0
            reward = r_reach + r_prox + r_press + r_effort + r_overshoot
        elif self.reward_type == "dense":
            # Even more shaped: reward for reducing distance
            reward = -dist + (10.0 if dist < 0.03 else 0.0) + (50.0 if pressed else 0.0)
        else:
            reward = 50.0 if pressed else -0.01

        self.episode_reward += reward

        # --- Termination ---
        terminated = False
        truncated = False
        info = {
            "pressed": pressed,
            "distance": dist,
            "btn_displacement": btn_disp,
            "btn_force": btn_force,
            "success": False,
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
