import sys
sys.path.insert(0, ".")
from envs.button_press_proprio_env import ButtonPressProprioEnv

XML = r"C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
env = ButtonPressProprioEnv(xml_path=XML)

hits = 0
min_d = 999
for ep in range(500):
    env.reset(seed=ep)
    for step in range(100):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        if info["distance"] < min_d:
            min_d = info["distance"]
        if info.get("success"):
            hits += 1
            break
    if ep % 100 == 0:
        print(f"  ep {ep}: hits={hits}, min_dist={min_d:.4f}")

print(f"\nHits: {hits}/500, Min distance: {min_d:.4f} m ({min_d*100:.1f} cm)")
env.close()