# Windows Setup Guide: Vision-Only MyoSuite Button-Press Controller

## Conda-Native auf Windows 10/11

Alle Befehle sind für **PowerShell**. Conda übernimmt das Management der nativen C/C++-Abhängigkeiten — du brauchst dadurch **keine Visual Studio Build Tools** separat zu installieren.

---

## Schritt 0: Voraussetzungen prüfen

Öffne PowerShell:

```powershell
# Windows-Version prüfen (mind. Windows 10 Build 19041 oder Windows 11)
winver

# GPU prüfen (NVIDIA empfohlen, aber nicht zwingend)
nvidia-smi
```

Wenn `nvidia-smi` funktioniert, notiere die **CUDA Version** oben rechts in der Ausgabe (z. B. 12.4). Falls der Befehl nicht erkannt wird, hast du entweder keine NVIDIA-GPU oder der Treiber fehlt — dann installierst du später die CPU-Variante von PyTorch.

---

## Schritt 1: Miniconda installieren

Miniconda ist die schlanke Variante von Anaconda — nur der Paketmanager, ohne vorinstallierte Pakete.

```powershell
# Option A: über winget (am schnellsten)
winget install Anaconda.Miniconda3

# Option B: manuell
# Gehe zu https://docs.conda.io/en/latest/miniconda.html
# Lade "Miniconda3 Windows 64-bit" herunter
# Im Installer:
#   - "Just Me" auswählen
#   - Installationspfad: Standard belassen (C:\Users\DEIN_NAME\miniconda3)
#   - "Add Miniconda3 to my PATH" NICHT ankreuzen (Conda aktiviert sich über eigene Shell)
#   - "Register Miniconda3 as my default Python" ankreuzen
```

**Nach der Installation: PowerShell komplett schließen und neu öffnen.**

Dann Conda initialisieren (einmalig):

```powershell
# Conda in PowerShell verfügbar machen
conda init powershell
```

**PowerShell nochmal schließen und neu öffnen.** Du solltest jetzt `(base)` am Anfang deines Prompts sehen.

Verifizieren:

```powershell
conda --version
# Erwartet: conda 24.x.x oder neuer
```

---

## Schritt 2: Git installieren

```powershell
winget install Git.Git
```

PowerShell schließen und neu öffnen, dann:

```powershell
git --version
# Erwartet: git version 2.x.x
```

---

## Schritt 3: Conda-Environment erstellen

Hier erstellen wir eine isolierte Umgebung mit Python 3.10. Alles, was du installierst, bleibt in dieser Umgebung — dein System-Python wird nicht berührt.

```powershell
# Environment erstellen
conda create -n myovision python=3.10 -y

# Environment aktivieren (musst du jedes Mal machen wenn du ein neues Terminal öffnest)
conda activate myovision
```

Dein Prompt sollte jetzt `(myovision)` zeigen statt `(base)`.

**Wichtig:** Alle folgenden Befehle setzen voraus, dass `(myovision)` aktiv ist.

---

## Schritt 4: PyTorch installieren (über Conda)

Conda holt sich automatisch die passenden CUDA-Bibliotheken — du brauchst das CUDA Toolkit **nicht** separat zu installieren.

```powershell
# MIT NVIDIA GPU (empfohlen — Conda lädt die CUDA-Runtime automatisch):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# OHNE NVIDIA GPU (CPU-only):
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

Verifizieren:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA verfuegbar:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'keine')"
```

Erwartete Ausgabe (GPU-Variante):

```
PyTorch: 2.x.x
CUDA verfuegbar: True
GPU: NVIDIA GeForce RTX xxxx
```

---

## Schritt 5: MuJoCo installieren (über Conda)

```powershell
conda install -c conda-forge mujoco -y
```

Verifizieren:

```powershell
python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
# Erwartet: MuJoCo: 3.x.x
```

---

## Schritt 6: Wissenschaftliche Pakete (über Conda)

Diese Pakete kommen als vorkompilierte Binaries über conda-forge — kein C-Compiler nötig.

```powershell
conda install -c conda-forge opencv matplotlib pandas scipy h5py pyyaml tqdm -y
```

---

## Schritt 7: MyoSuite und RL-Pakete (über pip)

MyoSuite und Stable-Baselines3 sind nicht in conda-forge verfügbar. pip innerhalb einer Conda-Umgebung funktioniert problemlos — die Regel ist nur: **erst Conda-Pakete, dann pip-Pakete** (genau diese Reihenfolge halten wir ein).

```powershell
pip install myosuite
pip install stable-baselines3[extra]
pip install gymnasium
pip install tensorboard
pip install imageio
pip install imageio-ffmpeg
```

Verifizieren:

```powershell
python -c "import myosuite; print('MyoSuite OK')"
python -c "from stable_baselines3 import PPO; print('SB3 OK')"
python -c "import gymnasium as gym; env = gym.make('myoElbowPose1D6MRandom-v0'); obs, info = env.reset(); print('Env OK — Obs shape:', obs.shape); env.close()"
```

---

## Schritt 8: Offscreen-Rendering konfigurieren

Deine Vision-Policy braucht gerenderte Kamerabilder bei jedem Zeitschritt. MuJoCo rendert diese offscreen (ohne sichtbares Fenster). Auf Windows gibt es drei Backends — wir probieren sie der Reihe nach:

```powershell
# Variante 1: EGL (bevorzugt, braucht NVIDIA GPU)
$env:MUJOCO_GL = "egl"

# Teste es:
python -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"sphere\" size=\"0.1\"/></worldbody></mujoco>'); d = mujoco.MjData(m); r = mujoco.Renderer(m, 64, 64); mujoco.mj_forward(m, d); r.update_scene(d); img = r.render(); print('Shape:', img.shape, '| Mean:', round(img.mean(), 1))"
```

Erwartete Ausgabe: `Shape: (64, 64, 3) | Mean: XX.X` (Mean muss > 0 sein, sonst ist das Bild schwarz).

Falls es fehlschlägt:

```powershell
# Variante 2: GLFW
$env:MUJOCO_GL = "glfw"
# Gleichen Test nochmal ausführen

# Variante 3: OSMesa (Software-Rendering, langsam aber zuverlässig)
conda install -c conda-forge mesalib -y
$env:MUJOCO_GL = "osmesa"
# Gleichen Test nochmal ausführen
```

Sobald ein Backend funktioniert, mach die Einstellung permanent:

```powershell
# Ersetze "egl" durch das Backend, das bei dir funktioniert hat
[System.Environment]::SetEnvironmentVariable("MUJOCO_GL", "egl", "User")
```

---

## Schritt 9: Projektstruktur anlegen

```powershell
# Projektordner erstellen
mkdir C:\Projects\myosuite-vision
cd C:\Projects\myosuite-vision

# Unterordner erstellen
mkdir assets, envs, training, logging, configs, scripts, results

# Python-Package-Marker erstellen
New-Item -Path envs\__init__.py -ItemType File -Force
New-Item -Path training\__init__.py -ItemType File -Force
New-Item -Path logging\__init__.py -ItemType File -Force
```

Ergebnis:

```
C:\Projects\myosuite-vision\
├── assets\          MuJoCo XMLs, Meshes, Texturen
├── envs\            Custom Gym-Environments
├── training\        PPO-Trainingsscripts
├── logging\         TrialRecord-Logger
├── configs\         YAML-Konfigurationen pro Experiment
├── scripts\         Utility-Scripts (Evaluation, Plots, Fitts-Analyse)
└── results\         Gespeicherte Modelle, Logs, Plots
```

---

## Schritt 10: MyoSuite-Arm-Modell kopieren

```powershell
# Finde den Asset-Ordner von MyoSuite
python -c "import myosuite, os; base = os.path.join(os.path.dirname(myosuite.__file__), 'envs', 'myo', 'assets'); [print(os.path.join(r, f)) for r, d, files in os.walk(base) for f in files if f.endswith('.xml')]"
```

Das gibt dir alle XML-Dateien aus. Suche nach der Elbow-Datei (typischerweise etwas mit `elbow` oder `1dof6muscles` im Namen) und kopiere sie samt dem gesamten Unterordner:

```powershell
# Passe den Pfad an deine Ausgabe an, z. B.:
python -c "
import myosuite, shutil, os
src = os.path.join(os.path.dirname(myosuite.__file__), 'envs', 'myo', 'assets', 'arm')
dst = r'C:\Projects\myosuite-vision\assets\arm'
if os.path.exists(src):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f'Kopiert: {src} -> {dst}')
    print('Dateien:', os.listdir(dst))
else:
    print(f'Ordner nicht gefunden: {src}')
    print('Suche manuell im Output von oben.')
"
```

---

## Schritt 11: Vollständiger Smoke-Test

Erstelle die Testdatei:

```powershell
@"
import sys
print(f"Python: {sys.version}")
print(f"Prefix: {sys.prefix}")
print()

errors = []

# --- Test 1: PyTorch ---
print("=== Test 1: PyTorch ===")
try:
    import torch
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        x = torch.randn(100, 100, device='cuda')
        print(f"  GPU-Tensor OK: {x.shape}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append("PyTorch")
print()

# --- Test 2: MuJoCo ---
print("=== Test 2: MuJoCo ===")
try:
    import mujoco
    print(f"  Version: {mujoco.__version__}")
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="sphere" size="0.1"/></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    print("  Simulation step OK")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append("MuJoCo")
print()

# --- Test 3: Offscreen Rendering ---
print("=== Test 3: Offscreen Rendering ===")
try:
    import mujoco
    import os
    print(f"  MUJOCO_GL = {os.environ.get('MUJOCO_GL', 'nicht gesetzt')}")
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<body pos="0.2 0 0.1"><geom type="sphere" size="0.05" rgba="1 0 0 1"/></body>'
        '</worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 64, 64)
    mujoco.mj_forward(m, d)
    renderer.update_scene(d)
    img = renderer.render()
    mean_val = img.mean()
    print(f"  Bild: {img.shape}, dtype={img.dtype}, Mean={mean_val:.1f}")
    assert img.shape == (64, 64, 3), "Falsche Bild-Groesse"
    assert mean_val > 0, "Schwarzes Bild — Rendering-Backend pruefen"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append("Rendering")
print()

# --- Test 4: MyoSuite Environment ---
print("=== Test 4: MyoSuite ===")
try:
    import myosuite
    import gymnasium as gym
    env = gym.make('myoElbowPose1D6MRandom-v0')
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Action shape: {env.action_space.shape}")
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
    print(f"  10 Steps OK, Reward: {reward:.4f}")
    env.close()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append("MyoSuite")
print()

# --- Test 5: MyoSuite + Rendering kombiniert ---
print("=== Test 5: MyoSuite Kamera-Rendering ===")
try:
    import myosuite
    import gymnasium as gym
    import mujoco
    env = gym.make('myoElbowPose1D6MRandom-v0')
    obs, info = env.reset()

    # Zugriff auf MuJoCo model/data (probiere beide API-Varianten)
    try:
        model = env.unwrapped.sim.model._model
        data = env.unwrapped.sim.data._data
        api = "sim.model._model"
    except AttributeError:
        model = env.unwrapped.model
        data = env.unwrapped.data
        api = "unwrapped.model"

    print(f"  API-Zugriff ueber: {api}")

    renderer = mujoco.Renderer(model, 64, 64)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    print(f"  Kamerabild: {img.shape}, Mean={img.mean():.1f}")
    assert img.mean() > 0, "Schwarzes Bild"
    env.close()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append("MyoSuite+Rendering")
print()

# --- Test 6: CNN Forward Pass ---
print("=== Test 6: PyTorch CNN ===")
try:
    import torch
    import torch.nn as nn
    dummy = torch.randn(1, 3, 64, 64)
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        nn.Flatten(),
    )
    out = cnn(dummy)
    print(f"  CNN Output: {out.shape}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append("CNN")
print()

# --- Test 7: Stable-Baselines3 ---
print("=== Test 7: Stable-Baselines3 ===")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    print("  PPO, DummyVecEnv, SubprocVecEnv importiert")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append("SB3")
print()

# --- Zusammenfassung ---
print("=" * 50)
if not errors:
    print("ALLE 7 TESTS BESTANDEN")
    print("Dein Setup ist bereit fuer die Entwicklung!")
else:
    print(f"FEHLGESCHLAGEN: {', '.join(errors)}")
    print("Behebe die obigen Fehler bevor du weitermachst.")
print("=" * 50)
"@ | Out-File -Encoding utf8 test_setup.py

python test_setup.py
```

---

## Kurzreferenz: Alle Befehle auf einen Blick

```powershell
# ---- EINMALIG ----
winget install Anaconda.Miniconda3
# PowerShell neu starten
conda init powershell
# PowerShell neu starten
winget install Git.Git

# ---- ENVIRONMENT ----
conda create -n myovision python=3.10 -y
conda activate myovision

# ---- CONDA-PAKETE (zuerst!) ----
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge mujoco -y
conda install -c conda-forge opencv matplotlib pandas scipy h5py pyyaml tqdm -y

# ---- PIP-PAKETE (danach) ----
pip install myosuite
pip install stable-baselines3[extra] gymnasium tensorboard imageio imageio-ffmpeg

# ---- RENDERING ----
$env:MUJOCO_GL = "egl"
[System.Environment]::SetEnvironmentVariable("MUJOCO_GL", "egl", "User")

# ---- PROJEKT ----
mkdir C:\Projects\myosuite-vision
cd C:\Projects\myosuite-vision
mkdir assets, envs, training, logging, configs, scripts, results

# ---- TEST ----
python test_setup.py
```

---

## Täglicher Workflow

Jedes Mal wenn du ein neues PowerShell-Fenster öffnest:

```powershell
conda activate myovision
cd C:\Projects\myosuite-vision
```

Das ist alles — conda kümmert sich um den Rest.

---

## Fehlerbehebung

### `conda` wird nicht erkannt
```powershell
conda init powershell
# PowerShell schließen und neu öffnen
```

### Execution Policy blockiert conda
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# "Y" bestätigen, dann PowerShell neu starten
```

### Schwarzes Bild beim Rendering (Mean = 0.0)
Rendering-Backend wechseln:
```powershell
$env:MUJOCO_GL = "glfw"     # Alternative 1
$env:MUJOCO_GL = "osmesa"   # Alternative 2 (Software, langsam)
# Dann test_setup.py nochmal laufen lassen
```

### `nvidia-smi` funktioniert, aber `torch.cuda.is_available()` ist False
PyTorch-CUDA-Version und Treiber passen nicht zusammen:
```powershell
# Prüfe die CUDA-Version deines Treibers:
nvidia-smi
# Oben rechts steht z. B. "CUDA Version: 12.4"
# Deine pytorch-cuda Version muss kleiner oder gleich sein
# Falls nötig, neu installieren:
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### `pip install myosuite` schlägt fehl mit C++-Compiler-Fehler
Conda hat die meisten C-Dependencies schon gelöst. Falls trotzdem ein Build fehlschlägt:
```powershell
# Build Tools nachinstallieren (nur als Fallback)
winget install Microsoft.VisualStudio.2022.BuildTools
# Im Installer: "Desktop development with C++" ankreuzen
# Danach pip install myosuite nochmal versuchen
```

### `env.unwrapped.sim` AttributeError
MyoSuite-API hat sich zwischen Versionen geändert:
```python
# Alte API:
model = env.unwrapped.sim.model._model
data  = env.unwrapped.sim.data._data

# Neue API:
model = env.unwrapped.model
data  = env.unwrapped.data
```
Der Smoke-Test probiert automatisch beide Varianten.

### CUDA Out of Memory beim Training
```python
# In deinem Trainings-Config:
n_envs = 16           # statt 64
image_res = (48, 48)  # statt (64, 64)
batch_size = 256      # statt 512
```

---

## Nächste Schritte nach erfolgreichem Test

Sobald alle 7 Tests bestanden sind, geht es an die Implementierung:

1. **Arm-XML modifizieren** — Button-Panel und Kamera in `assets\myoarm_button.xml` einfügen
2. **Custom Environment** — `envs\button_press_env.py` mit Kontakt-Erkennung, Vision-Wrapper, Reward
3. **Trainings-Script** — `training\train_ppo_vision.py` mit SB3 und CNN-Policy
4. **Trial-Logger** — `logging\trial_logger.py` mit TrialRecord-Dataclass
5. **Phase 1** — Propriozeptive Baseline um Reward/Kontakt-Logik zu validieren
6. **Phase 2** — Vision-Observations, CNN-Training
