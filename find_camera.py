"""
Opens MuJoCo interactive viewer.
Move camera with mouse until you have the perfect first-person view.
Press Tab to print current camera position/orientation.
"""
import sys, mujoco, mujoco.viewer, time, numpy as np
sys.path.insert(0, ".")
from envs.button_press_proprio_env import ButtonPressProprioEnv

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
env = ButtonPressProprioEnv(xml_path=XML)
env.reset()

# Set arm to a mid-flexion pose so we can see it
env.data.qpos[env._elbow_qpos_addr] = 1.5
env.data.ctrl[:] = [0.2, 0.2, 0.2, 0.5, 0.5, 0.3]
for _ in range(500):
    mujoco.mj_step(env.model, env.data)
mujoco.mj_forward(env.model, env.data)

print("=== Interactive Camera Finder ===")
print("Use mouse to orbit/pan/zoom:")
print("  Left drag   = orbit")
print("  Right drag  = pan")
print("  Scroll      = zoom")
print("")
print("When you have a good first-person view,")
print("look at the terminal — camera info prints every 2 seconds.")
print("Copy the lookat/distance/azimuth/elevation values.")
print("Close window to exit.")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        cam = viewer.cam
        print(f"\r  lookat=[{cam.lookat[0]:.3f}, {cam.lookat[1]:.3f}, {cam.lookat[2]:.3f}]  "
              f"dist={cam.distance:.3f}  azimuth={cam.azimuth:.1f}  elevation={cam.elevation:.1f}",
              end="", flush=True)
        time.sleep(2)

print("\n\nDone! Use these values to set the camera in the XML.")
env.close()