import sys, os, numpy as np, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Take random steps so arm is in a pose
data.ctrl[:] = np.random.uniform(0, 0.5, model.nu)
for _ in range(500):
    mujoco.mj_step(model, data)
mujoco.mj_forward(model, data)

os.makedirs("results", exist_ok=True)
cameras = ["static_cam", "button_cam"]
resolutions = [64, 128, 256]

fig, axes = plt.subplots(len(cameras), len(resolutions), figsize=(12, 8))
for row, cam in enumerate(cameras):
    for col, res in enumerate(resolutions):
        r = mujoco.Renderer(model, res, res)
        r.update_scene(data, camera=cam)
        img = r.render()
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"{cam} {res}x{res}")
        axes[row][col].axis("off")
        r.close()

plt.suptitle("MyoArm: What the agent sees")
plt.tight_layout()
plt.savefig("results/myoarm_vision.png", dpi=150)
print("Saved: results/myoarm_vision.png")
print("Open: start results\\myoarm_vision.png")