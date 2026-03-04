"""
Visualize the trained agent pressing the button.
Saves a video to results/replay.mp4 and shows live window.
"""
import os, sys
import numpy as np
import mujoco
import mujoco.viewer
import imageio
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.button_press_env import ButtonPressEnv
from stable_baselines3 import PPO

XML_PATH = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
MODEL_PATH = r"results\phase2_vision\best_model\best_model.zip"

# --- Save video ---
print("Recording video...")

# Low-res env for policy input
env = ButtonPressEnv(
    xml_path=XML_PATH,
    camera_name="static_cam",
    image_size=(64, 64),
    frame_stack=3,
    max_steps=500,
    randomization_level="none",
)

# High-res renderer for video only
hires_renderer = mujoco.Renderer(env.model, 480, 480)

model = PPO.load(MODEL_PATH, device="cpu")
frames = []

obs, info = env.reset(seed=42)
done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1

    # Render high-res frame for video
    hires_renderer.update_scene(env.data, camera="static_cam")
    frame = hires_renderer.render()
    frames.append(frame.copy())

status = "PRESSED!" if info.get("success", False) else "missed"
print(f"  Result: {status} after {step} steps")
print(f"  Distance: {info['distance']:.4f} m")

hires_renderer.close()
env.close()

if frames:
    video_path = os.path.join("results", "replay.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"  Video saved: {video_path}")

# --- Option 2: Live 3D viewer ---
print("\nLaunching live viewer (close window to exit)...")
env2 = ButtonPressEnv(
    xml_path=XML_PATH,
    image_size=(64, 64),
    frame_stack=3,
    max_steps=500,
    randomization_level="none",
)

obs, info = env2.reset(seed=42)

with mujoco.viewer.launch_passive(env2.model, env2.data) as viewer:
    for episode in range(5):
        obs, info = env2.reset(seed=episode)
        done = False

        while not done and viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env2.step(action)
            done = terminated or truncated

            viewer.sync()
            time.sleep(0.01)

        if not viewer.is_running():
            break

        result = "PRESSED" if info.get("success", False) else "missed"
        print(f"  Episode {episode+1}: {result}")
        time.sleep(0.5)

env2.close()
print("Done!")