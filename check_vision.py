"""
Saves what the agent sees at different resolutions.
Creates a grid image: results/agent_vision.png
"""
import os, sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.button_press_env import ButtonPressEnv

XML_PATH = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"

resolutions = [32, 64, 128, 256]
cameras = ["static_cam", "button_cam"]

fig, axes = plt.subplots(len(cameras), len(resolutions), figsize=(16, 8))

for row, cam in enumerate(cameras):
    for col, res in enumerate(resolutions):
        env = ButtonPressEnv(
            xml_path=XML_PATH,
            camera_name=cam,
            image_size=(res, res),
            frame_stack=3,
            randomization_level="none",
        )
        obs, _ = env.reset(seed=42)

        # Take a few steps so arm is in a non-default pose
        for _ in range(20):
            env.step(env.action_space.sample())

        # Render high quality for display
        renderer = mujoco.Renderer(env.model, res, res)
        renderer.update_scene(env.data, camera=cam)
        img = renderer.render()

        ax = axes[row][col]
        ax.imshow(img)
        ax.set_title(f"{cam}\n{res}x{res}", fontsize=10)
        ax.axis("off")

        renderer.close()
        env.close()

plt.suptitle("What the agent sees", fontsize=14, fontweight="bold")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", "agent_vision.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# Also save the actual 64x64 observation (3 stacked frames)
env = ButtonPressEnv(
    xml_path=XML_PATH,
    camera_name="static_cam",
    image_size=(64, 64),
    frame_stack=3,
    randomization_level="none",
)
obs, _ = env.reset(seed=42)
for _ in range(20):
    obs, _, _, _, _ = env.step(env.action_space.sample())

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    axes2[i].imshow(obs[i])
    axes2[i].set_title(f"Frame t-{2-i}", fontsize=11)
    axes2[i].axis("off")

plt.suptitle("3 stacked frames (actual policy input at 64x64)", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path2 = os.path.join("results", "stacked_frames.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path2}")

env.close()
print("\nOpen results\\agent_vision.png and results\\stacked_frames.png to inspect.")