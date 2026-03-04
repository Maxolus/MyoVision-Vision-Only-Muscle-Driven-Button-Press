# MyoVision — Vision-Only Muscle-Driven Button Press

A reinforcement learning system that trains a musculoskeletal arm to press a randomly placed button using **only first-person camera images**. No proprioception, no coordinates — just pixels and muscles.

Built on [MyoSuite](https://sites.google.com/view/myosuite) and [MuJoCo](https://mujoco.org/), designed for comparison with human motor performance via Fitts' law.

---

## How It Works

A virtual first-person camera renders 64×64 RGB frames from the model's head position, looking down at its own hand and a table with a button. Three consecutive frames are stacked and fed through a CNN, which outputs activation levels for all muscles. The agent learns entirely from pixel input — it must discover where the button is, plan a reach, and coordinate muscle contractions to press it.

Training uses a **two-phase curriculum**:

**Phase 1 (Proprioception):** The agent receives a compact state vector (fingertip positions, button position, joint angles, distance). It learns muscle coordination — how to move the arm and hand to reach and press the button. This converges fast because the agent knows exactly where everything is.

**Phase 2 (Vision):** The MLP weights from Phase 1 are transferred into a new policy with a CNN encoder. The MLP layers (muscle control) are optionally frozen. The CNN learns to extract from pixels the same spatial information the state vector provided. Because the agent already knows how to move, it only needs to learn to see.

```
Phase 1:  State Vector (24-dim) → MLP → 32 muscle activations
Phase 2:  Camera Frames (3×64×64×3) → CNN → MLP → 32 muscle activations
                                        ↑new    ↑transferred from Phase 1
```

---

## Two Models

The project contains two model variants. The **Elbow** model was used for initial prototyping and proof-of-concept. The **MyoArm** model is the full implementation for human comparison.

**Elbow (prototype):** 1 DOF (elbow flexion), 6 muscles. Simple but limited — the hand moves on a fixed arc and cannot reach varied button positions. Good for testing the pipeline.

**MyoArm (full):** Shoulder (4 joints) + elbow + pronation/supination + wrist (2 DOF) + thumb (3 DOF) + 4 fingers (3-4 DOF each). 63 muscles, full workspace. Button spawns at random positions on a table each episode, forcing the agent to actually use vision.

---

## Prerequisites

- Windows 10/11 with NVIDIA GPU (recommended)
- Anaconda/Miniconda installed

---

## Installation

```cmd
conda create -n myovision python=3.10 -y
conda activate myovision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install mujoco myosuite stable-baselines3 gymnasium
pip install opencv-python-headless matplotlib pandas scipy tensorboard
pip install imageio imageio-ffmpeg tqdm wandb
```

If numpy errors appear after installing:

```cmd
pip install numpy==1.26.4 --force-reinstall
```

---

## Important: XML Files Must Be Copied to MyoSuite

The MuJoCo XML files use `<include>` tags that reference assets (meshes, textures, muscle definitions) inside the MyoSuite installation directory via relative paths. They will **not work** from your project folder — MuJoCo cannot resolve the `../../../../simhive/...` paths from `D:\MyoVision`. You must copy them into the MyoSuite package where the relative paths are correct:

**Elbow model:**
```cmd
copy "D:\MyoVision\assets\elbow\myoelbow_buttonpress.xml" "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml"
```

**MyoArm model:**
```cmd
copy "D:\MyoVision\assets\arm\myoarm_buttonpress.xml" "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"
```

**You must re-copy after every XML edit.** If you change the camera position, button position, or any other parameter in the XML, the copy in the MyoSuite directory is what gets loaded — not the one in your project folder.

The full paths used in all scripts:

```
Elbow: C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\elbow\myoelbow_buttonpress.xml
MyoArm: C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml
```

---

## Project Structure

```
D:\MyoVision\
├── assets\
│   ├── elbow\myoelbow_buttonpress.xml    ← Elbow model (prototype)
│   └── arm\myoarm_buttonpress.xml        ← MyoArm model (full)
├── envs\
│   ├── __init__.py
│   ├── button_press_env.py               ← Elbow vision env
│   ├── button_press_proprio_env.py       ← Elbow proprioceptive env
│   ├── myoarm_button_proprio_env.py      ← MyoArm proprioceptive env (Phase 1)
│   └── myoarm_button_vision_env.py       ← MyoArm vision env (Phase 2)
├── training\
│   ├── __init__.py
│   ├── train_phase1_proprio.py           ← Elbow Phase 1
│   ├── train_phase2_vision.py            ← Elbow Phase 2
│   ├── train_myoarm_phase1.py            ← MyoArm Phase 1
│   ├── train_myoarm_phase2.py            ← MyoArm Phase 2
│   └── train_ppo_vision.py              ← Elbow direct vision (deprecated)
├── scripts\
│   └── fitts_analysis.py                 ← Fitts' law evaluation
├── results\                               ← Models, logs, videos, plots
│
│   === Test & Utility Scripts ===
├── test_setup.py                          ← Verify Python/MuJoCo/PyTorch install
├── test_env.py                            ← Test Elbow button press env
├── test_myoarm.py                         ← Test MyoArm env + reachability
├── test_reachability.py                   ← Elbow: can arm reach button?
├── test_contact.py                        ← Debug collision detection
├── test_contact2.py                       ← Debug geom-to-button distances
├── quick_test.py                          ← Fast Elbow brute-force hit test
├── quick_test2.py                         ← Fast Elbow geom distance check
├── quick_arm_test.py                      ← Fast MyoArm brute-force hit test
├── find_workspace.py                      ← Map fingertip workspace + camera check
├── find_camera.py                         ← Interactive MuJoCo viewer for camera tuning
├── check_vision.py                        ← Render what Elbow agent sees
├── check_myoarm_vision.py                 ← Render what MyoArm agent sees
├── visualize.py                           ← Watch trained agent + save video
└── README.md
```

---

## Test Scripts Explained

**`test_setup.py`** — Run first after installation. Verifies MuJoCo, MyoSuite, PyTorch (with CUDA), Stable-Baselines3, and OpenCV all work. Also tests offscreen rendering. If this fails, fix your environment before proceeding.

**`test_myoarm.py`** — Comprehensive test for the MyoArm environment. Loads the model, checks obs/action shapes, runs random steps, verifies button randomization produces different positions, runs a 500-episode brute-force reachability test, and checks vision rendering. Run after any XML change.

**`test_reachability.py`** — Elbow-specific. Sweeps the elbow joint through its full range to find the minimum geometric distance to the button. Also runs brute-force random episodes to check if random actions can produce a hit.

**`quick_arm_test.py`** — Fast 500-episode brute-force test for MyoArm. Reports hit count and minimum distance. Use after changing button position or press threshold. Target: 1-10% random hits. 0% means the button is unreachable or the threshold is too tight. 100% means the task is too easy.

**`find_workspace.py`** — Runs 1000 random episodes and records all 5 fingertip positions at every timestep. Outputs the XYZ range and mean for each fingertip. Use the workspace center as the starting point for button placement. Also renders and saves camera views.

**`find_camera.py`** — Opens MuJoCo's interactive 3D viewer. Drag with mouse to orbit, right-drag to pan, scroll to zoom. Camera parameters (pos, orientation) print to console every 2 seconds. When you find a good first-person view, copy the values into the XML camera definition.

**`check_vision.py`** / **`check_myoarm_vision.py`** — Renders what the agent sees at multiple resolutions (32, 64, 128, 256) from each camera. Saves grid image to `results/`. Use to verify the button and hand are visible. If you only see walls or floor, the camera needs repositioning.

**`visualize.py`** — Loads a trained model, runs episodes, saves video to `results/replay.mp4`, and opens a live 3D viewer. Edit `MODEL_PATH` and `XML_PATH` at the top of the file to point to the correct model and XML.

---

## Quick Start: MyoArm (Recommended)

### 1. Verify installation

```cmd
conda activate myovision
cd D:\MyoVision
python test_setup.py
```

### 2. Copy XML to MyoSuite

```cmd
copy "D:\MyoVision\assets\arm\myoarm_buttonpress.xml" "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"
```

### 3. Test the environment

```cmd
python test_myoarm.py
```

Check that the reachability test shows some hits (1-10% is ideal). If 0%, move the button closer to the workspace center. If 100%, reduce the press threshold or move the button further away. After any XML edit, re-copy to the MyoSuite directory.

### 4. Phase 1 — Learn muscle coordination

```cmd
python training\train_myoarm_phase1.py --total-timesteps 2000000 --n-envs 8 --wandb
```

Monitor `success_rate` in Weights & Biases. Wait until it reaches 50%+ consistently before proceeding to Phase 2. With 63 muscles and randomized button, expect this to take 1-2M steps.

### 5. Phase 2 — Learn to see

```cmd
python training\train_myoarm_phase2.py --phase1-model results\myoarm_phase1\best_model\best_model.zip --total-timesteps 2000000 --n-envs 4 --wandb
```

Add `--freeze-mlp` to freeze the muscle control layers and only train the CNN encoder. Use lower `--n-envs` than Phase 1 because rendering is expensive.

### 6. Evaluate with Fitts' Law

```cmd
python scripts\fitts_analysis.py --model-path results\myoarm_phase2\best_model\best_model.zip --xml-path "C:\Users\max-r\anaconda3\envs\myovision\lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_buttonpress.xml"
```

---

## Training Options

### MyoArm Phase 1

| Flag | Default | Description |
|---|---|---|
| `--n-envs` | 8 | Parallel environments |
| `--total-timesteps` | 2000000 | Total training steps |
| `--max-steps` | 300 | Max steps per episode |
| `--randomize-button` | True | Random button position each episode |
| `--no-randomize-button` | — | Fixed button position |
| `--wandb` | off | Enable Weights & Biases logging |
| `--device` | auto | cpu or cuda |

### MyoArm Phase 2

| Flag | Default | Description |
|---|---|---|
| `--phase1-model` | required | Path to Phase 1 .zip model |
| `--image-size` | 64 | Camera resolution |
| `--frame-stack` | 3 | Stacked frames |
| `--lr` | 1e-4 | Learning rate (lower than Phase 1) |
| `--freeze-mlp` | off | Freeze MLP, train only CNN |
| `--n-envs` | 4 | Fewer envs due to rendering cost |
| `--wandb` | off | Enable W&B logging |

---

## Key Design Decisions

### Press Detection

Button presses are detected via **proximity** rather than physics contact. MuJoCo contact detection requires matching `contype`/`conaffinity` bits, but MyoSuite arm geoms have `conaffinity=0` by default. Rather than modifying the base model, we check Euclidean distance from hand geoms to the button center. A press is registered when any hand geom comes within 1.8 cm.

### Button Randomization

Each episode, the button spawns at a random (x, y) position within the arm's reachable workspace on the table. Height stays fixed. This forces the agent to use visual information to locate the button rather than memorizing a fixed motor sequence. The randomization range was determined empirically using `find_workspace.py`.

### Reward Design

Staged reward: strong distance shaping (`-5 × distance`), proximity bonuses at 15cm/10cm/5cm thresholds, a large press bonus (+100), and a small effort penalty. The strong shaping ensures the agent gets gradient signal even when far from the button.

### Camera

First-person perspective calibrated using `find_camera.py` in the MuJoCo interactive viewer. The camera sits at the model's head position looking down at its own hand and the table. FOV 60°, matching approximate human vertical field of view.

### Why Curriculum Learning

Training vision-only from scratch (without Phase 1) failed after 2M steps on the Elbow model. The agent must simultaneously learn to see and to coordinate muscles — too hard to solve jointly. The curriculum approach (Phase 1 proprio → Phase 2 vision) solved the Elbow task in 26k Phase 2 steps with 96% success rate.

---

## Fitts' Law Metrics for Human Comparison

| Metric | Human (typical) | Description |
|---|---|---|
| Throughput | 4-10 bits/s | Information processing rate |
| Slope b | 100-200 ms/bit | Sensitivity to task difficulty |
| R² | 0.90-0.99 | Linearity of speed-accuracy tradeoff |
| Velocity symmetry | 0.4-0.5 | Bell-shaped velocity profile |
| Path efficiency | 0.8-0.95 | Straightness of reach trajectory |
| Submovements | 1-3 | Corrective adjustments during reach |

---

## Troubleshooting

**"XML Error: file not found"** — The XML is not in the MyoSuite directory. Re-run the `copy` command. Remember to re-copy after every XML edit.

**0% success in reachability test** — Button is out of reach. Run `find_workspace.py`, move button to workspace center in XML, re-copy.

**100% success with random actions** — Task too easy. Reduce press threshold in env files (currently 0.018 m) or move button further from workspace center.

**"DLL load failed" for OpenCV** — Use `pip install opencv-python-headless`.

**CUDA: False** — Reinstall PyTorch with CUDA: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`

**Black images** — Don't set `MUJOCO_GL` on Windows. MuJoCo uses WGL by default.

**"cannot import name 'deepcopy' from 'copy'"** — You have a file named `copy.py` in your project. Delete it. Never name files after Python standard library modules (`copy`, `logging`, `test`, `email`, `io`, `json`, `typing`).

---

## Architecture

- **Simulation**: MuJoCo 3.3, 2ms timestep, 5 substeps per action
- **Musculoskeletal model**: MyoSuite MyoArm — 39 joints, 63 Hill-type muscles
- **Button**: Spring-loaded slide joint, press via hand-geom proximity (1.8 cm)
- **Vision**: 64×64 RGB, 3-frame stack, NatureCNN encoder (3 conv → 256-d)
- **RL**: PPO via Stable-Baselines3, curriculum (proprio → vision)
- **Reward**: staged (-5×dist + proximity bonuses + 100×press - effort)

---

## License

MyoSuite model files are under Apache License 2.0 (Vikash Kumar, Vittorio Caggiano, Huawei Wang). Environment wrappers, training scripts, and analysis code are provided as-is for research purposes.
