"""
Vision-Only Button-Press Environment for MyoSuite/MuJoCo
=========================================================
A muscle-driven elbow model must press a button using only RGB camera input.
No proprioception, no target coordinates in the observation.

Usage:
    import gymnasium as gym
    from envs.button_press_env import ButtonPressEnv

    env = ButtonPressEnv(camera_name="static_cam", image_size=(64, 64))
    obs, info = env.reset()
    # obs shape: (3, 64, 64, 3) — 3 stacked RGB frames
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List
import time


# ============================================================
# Trial Record — logging schema for human comparison
# ============================================================
@dataclass
class TrialRecord:
    """One trial's data, compatible with both RL agent and human participants."""
    participant_id: str = ""
    trial_id: int = 0
    condition: str = ""

    # Fitts' law parameters
    target_distance: float = 0.0        # D: meters
    target_width: float = 0.0           # W: meters (button diameter)
    index_of_difficulty: float = 0.0    # log2(2D/W)

    # Timing (seconds)
    stimulus_onset_time: float = 0.0
    movement_onset_time: float = 0.0
    contact_time: float = 0.0
    reaction_time: float = 0.0         # movement_onset - stimulus_onset
    movement_time: float = 0.0         # contact - movement_onset

    # Spatial accuracy
    endpoint_pos: Optional[np.ndarray] = None
    target_center: Optional[np.ndarray] = None
    endpoint_error: float = 0.0
    hit: bool = False

    # Trajectory (lists populated each step)
    timestamps: List[float] = field(default_factory=list)
    wrist_xyz: List[np.ndarray] = field(default_factory=list)
    wrist_velocity: List[np.ndarray] = field(default_factory=list)
    muscle_activations: List[np.ndarray] = field(default_factory=list)
    joint_angles: List[float] = field(default_factory=list)


# ============================================================
# Press Detector
# ============================================================
class PressDetector:
    """Detects button press via joint displacement + touch sensor."""

    def __init__(self, model, data,
                 displacement_threshold=0.005,
                 force_threshold=0.2):
        self.model = model
        self.data = data
        self.disp_thresh = displacement_threshold
        self.force_thresh = force_threshold

        # Cache sensor indices
        self.btn_pos_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "btn_pos")
        self.btn_touch_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "btn_touch")

    def check(self):
        """Returns (is_pressed, min_distance, force)."""
        btn_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "btn_target")
        btn_pos = self.data.site_xpos[btn_site_id]
        
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
        
        pressed = min_dist < 0.025
        return pressed, min_dist, 1.0 if pressed else 0.0


# ============================================================
# Domain Randomizer
# ============================================================
class DomainRandomizer:
    """Randomizes visual and dynamics parameters each episode."""

    def __init__(self, model, np_random, level="medium"):
        self.model = model
        self.np_random = np_random
        self.level = level

        # Store originals for resetting
        self._original_btn_pos = None
        self._original_btn_rgba = None
        self._original_light_pos = None
        self._original_muscle_force = None

    def store_defaults(self):
        """Call once after model is loaded."""
        btn_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "button_panel")
        self._original_btn_pos = self.model.body_pos[btn_body_id].copy()

        btn_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "btn_top")
        self._original_btn_rgba = self.model.geom_rgba[btn_geom_id].copy()

        # Store muscle forces (actuator gear/force)
        self._original_muscle_force = self.model.actuator_gainprm[:, 0].copy()

    def randomize(self):
        """Apply randomization. Call at the start of each episode."""
        if self.level == "none":
            return {}

        params = {}

        # --- Button position ---
        btn_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "button_panel")
        if self.level in ("medium", "high"):
            offset = self.np_random.uniform([-0.03, -0.03, -0.02],
                                            [0.03,  0.03,  0.02])
            self.model.body_pos[btn_body_id] = self._original_btn_pos + offset
            params["btn_offset"] = offset

        # --- Button color ---
        btn_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "btn_top")
        if self.level in ("medium", "high"):
            hue_r = self.np_random.uniform(0.6, 1.0)
            hue_g = self.np_random.uniform(0.0, 0.3)
            hue_b = self.np_random.uniform(0.0, 0.3)
            self.model.geom_rgba[btn_geom_id] = [hue_r, hue_g, hue_b, 1.0]
            params["btn_color"] = [hue_r, hue_g, hue_b]

        # --- Lighting ---
        if self.level == "high":
            light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "task_light")
            if light_id >= 0:
                light_offset = self.np_random.uniform([-0.1, -0.1, -0.05],
                                                      [0.1,  0.1,  0.05])
                self.model.light_pos[light_id] += light_offset
                params["light_offset"] = light_offset

        # --- Muscle strength ---
        if self.level in ("medium", "high"):
            scale = self.np_random.uniform(0.8, 1.2, size=self._original_muscle_force.shape)
            self.model.actuator_gainprm[:, 0] = self._original_muscle_force * scale
            params["muscle_scale"] = scale

        return params


# ============================================================
# Main Environment
# ============================================================
class ButtonPressEnv(gym.Env):
    """
    Vision-only button-press environment.

    Observation: stacked RGB frames (frame_stack, H, W, 3)
    Action: muscle activations [0, 1]^6
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        camera_name: str = "static_cam",
        image_size: tuple = (64, 64),
        frame_stack: int = 3,
        max_steps: int = 500,
        reward_type: str = "staged",
        randomization_level: str = "medium",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # --- Paths ---
        if xml_path is None:
            # Default: look in assets/ relative to this file's parent
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "assets", "elbow", "myoelbow_buttonpress.xml")
        self.xml_path = xml_path

        # --- MuJoCo model ---
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
        n_muscles = self.model.nu  # should be 6
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_muscles,), dtype=np.float32
        )

        # --- Episode state ---
        self.max_steps = max_steps
        self.step_count = 0
        self.dt = self.model.opt.timestep
        self.frames = deque(maxlen=frame_stack)
        self.reward_type = reward_type

        # --- Press detection ---
        self.press_detector = PressDetector(self.model, self.data)

        # --- Domain randomization ---
        self.randomizer = DomainRandomizer(
            self.model, np.random.default_rng(), level=randomization_level
        )
        self.randomizer.store_defaults()

        # --- Logging ---
        self.trial_record = None
        self.trial_count = 0
        self._prev_wrist_pos = None
        self._movement_started = False

        # --- Cache body/site IDs ---
        self._wrist_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "wrist"
        )
        self._btn_target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "btn_target"
        )
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )

    # ----------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.randomizer.np_random = np.random.default_rng(seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Domain randomization
        rand_params = self.randomizer.randomize()

        # Forward to update positions
        mujoco.mj_forward(self.model, self.data)

        # Reset episode state
        self.step_count = 0
        self.frames.clear()
        self._movement_started = False

        # Initialize trial record
        self.trial_count += 1
        wrist_pos = self.data.site_xpos[self._wrist_site_id].copy()
        btn_pos = self.data.site_xpos[self._btn_target_site_id].copy()
        distance = np.linalg.norm(wrist_pos - btn_pos)
        button_width = 0.030  # diameter = 2 * radius (0.015)

        self.trial_record = TrialRecord(
            participant_id="RL_agent",
            trial_id=self.trial_count,
            target_distance=distance,
            target_width=button_width,
            index_of_difficulty=np.log2(2 * distance / button_width),
            stimulus_onset_time=0.0,
            target_center=btn_pos.copy(),
        )
        self._prev_wrist_pos = wrist_pos.copy()

        # Stack initial frames
        frame = self._render_frame()
        for _ in range(self.frame_stack):
            self.frames.append(frame)

        obs = np.stack(list(self.frames), axis=0)
        info = {"rand_params": rand_params}
        return obs, info

    def step(self, action):
        # Clip action to [0, 1]
        action = np.clip(action, 0.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        # Step simulation (multiple substeps for stability)
        n_substeps = 5
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # Update forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get positions
        wrist_pos = self.data.site_xpos[self._wrist_site_id].copy()
        btn_pos = self.data.site_xpos[self._btn_target_site_id].copy()

        # Check press
        pressed, btn_disp, btn_force = self.press_detector.check()

        # Compute reward
        reward = self._compute_reward(wrist_pos, btn_pos, pressed, btn_force, action)

        # Log trajectory
        current_time = self.step_count * self.dt * n_substeps
        self._log_step(wrist_pos, action, current_time)

        # Check movement onset (velocity threshold)
        velocity = np.linalg.norm(wrist_pos - self._prev_wrist_pos) / (self.dt * n_substeps)
        if not self._movement_started and velocity > 0.01:
            self._movement_started = True
            self.trial_record.movement_onset_time = current_time
        self._prev_wrist_pos = wrist_pos.copy()

        # Termination
        terminated = False
        truncated = False
        info = {
            "pressed": pressed,
            "btn_displacement": btn_disp,
            "btn_force": btn_force,
            "wrist_pos": wrist_pos,
            "btn_pos": btn_pos,
            "distance": np.linalg.norm(wrist_pos - btn_pos),
        }

        if pressed:
            terminated = True
            info["success"] = True
            self.trial_record.contact_time = current_time
            self.trial_record.hit = True
            self.trial_record.endpoint_pos = wrist_pos.copy()
            self.trial_record.endpoint_error = np.linalg.norm(wrist_pos - btn_pos)
            if self._movement_started:
                self.trial_record.reaction_time = (
                    self.trial_record.movement_onset_time - self.trial_record.stimulus_onset_time
                )
                self.trial_record.movement_time = (
                    self.trial_record.contact_time - self.trial_record.movement_onset_time
                )
            info["trial_record"] = self.trial_record

        elif self.step_count >= self.max_steps:
            truncated = True
            info["success"] = False
            self.trial_record.hit = False
            info["trial_record"] = self.trial_record

        # Render observation
        frame = self._render_frame()
        self.frames.append(frame)
        obs = np.stack(list(self.frames), axis=0)

        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------
    # Reward
    # ----------------------------------------------------------
    def _compute_reward(self, wrist_pos, btn_pos, pressed, btn_force, action):
        dist = np.linalg.norm(wrist_pos - btn_pos)

        if self.reward_type == "staged":
            # Stage 1: reaching (shaped, small weight)
            r_reach = -1.0 * dist

            # Stage 2: proximity bonus
            r_prox = 3.0 if dist < 0.03 else 0.0

            # Stage 3: press event (dominant)
            r_press = 50.0 if pressed else 0.0

            # Regularization: penalize excessive muscle activation
            r_effort = -0.01 * np.sum(action ** 2)

            # Penalize overshoot force
            r_overshoot = -0.1 * max(0.0, btn_force - 1.0) if pressed else 0.0

            return r_reach + r_prox + r_press + r_effort + r_overshoot

        elif self.reward_type == "sparse":
            return 50.0 if pressed else -0.01

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    # ----------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------
    def _render_frame(self):
        """Render an RGB frame from the specified camera."""
        self.renderer.update_scene(self.data, camera=self.camera_name)
        img = self.renderer.render()
        return img.copy()

    def render(self):
        """Gym render method."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------
    def _log_step(self, wrist_pos, action, current_time):
        """Append one timestep to the trial record."""
        tr = self.trial_record
        tr.timestamps.append(current_time)
        tr.wrist_xyz.append(wrist_pos.copy())
        tr.muscle_activations.append(action.copy())

        # Joint angle (elbow flexion)
        elbow_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "r_elbow_flex"
        )
        if elbow_jnt_id >= 0:
            qpos_addr = self.model.jnt_qposadr[elbow_jnt_id]
            tr.joint_angles.append(float(self.data.qpos[qpos_addr]))

        # Velocity (finite difference)
        if len(tr.wrist_xyz) >= 2:
            dt = tr.timestamps[-1] - tr.timestamps[-2]
            if dt > 0:
                vel = (tr.wrist_xyz[-1] - tr.wrist_xyz[-2]) / dt
                tr.wrist_velocity.append(vel)
            else:
                tr.wrist_velocity.append(np.zeros(3))
        else:
            tr.wrist_velocity.append(np.zeros(3))

    def close(self):
        """Clean up renderer."""
        if hasattr(self, "renderer"):
            self.renderer.close()
        super().close()
