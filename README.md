# Robot Get-Up Using Action Priors

A MuJoCo research project for training the 29-DoF Unitree G1 to stand up from
fallen poses. The ultimate goal is to use action priors to
guide exploration and train standing up end to end, while producing a final
policy that runs from robot state alone without a reference trajectory.

The repository currently includes:

- A deterministic G1 MuJoCo environment with normalized joint-position actions
- Joint-space PD control and contact/state diagnostics
- BONES-SEED loading, preprocessing, and replay
- A fixed-fallen-state task and vanilla PPO baseline

APEX action priors, style rewards, multiple critics, randomized falls, and
domain randomization are planned for later milestones.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
```

The interactive viewer requires a desktop/OpenGL session. Use `--headless` on
servers or in environments without a display.

## Running the project

View or smoke-test the environment:

```bash
python scripts/view_env.py --pose supine
python scripts/view_env.py --pose prone --headless --duration 2
```

Place BONES-SEED under `datasets/bones-seed`, then inspect and preprocess the
selected get-up motion:

```bash
python scripts/inspect_bones_dataset.py --samples 2
python scripts/find_getup_demos.py --limit 20
python scripts/preprocess_demo.py --motion stand_up_lying_R_002__A473
```

Replay or calibrate the motion:

```bash
python scripts/replay_demo.py --motion stand_up_lying_R_002__A473 --mode kinematic
python scripts/replay_demo.py --motion stand_up_lying_R_002__A473 --mode dynamic --headless
python scripts/sweep_pd_gains.py --motion stand_up_lying_R_002__A473
```

Run the task-only PPO baseline:

```bash
python scripts/evaluate_baselines.py
python scripts/train_ppo.py --total-steps 100000
python scripts/evaluate_ppo.py \
  --checkpoint artifacts/ppo/milestone3/best.npz \
  --episodes 5 \
  --reward-plot artifacts/ppo/milestone3/reward_components.png
```

Record a headless rollout when EGL is available:

```bash
MUJOCO_GL=egl python scripts/evaluate_ppo.py \
  --checkpoint artifacts/ppo/milestone3/best.npz \
  --episodes 1 \
  --record-gif artifacts/ppo/milestone3/representative.gif
```

## Core conventions

Actions are normalized 29-vectors in `[-1, 1]`. Each action spans the full
finite MuJoCo range of its corresponding joint:

```text
q_center = (q_min + q_max) / 2
q_scale  = (q_max - q_min) / 2
q_target = q_center + q_scale * action
```

The controller applies
`tau = kp * (q_target - q) - kd * qdot`, with default gains `kp=60` and `kd=3`.
Defaults are a 0.002 s simulation timestep and a 50 Hz control rate.

The PPO actor receives a 97-value proprioceptive observation containing
projected gravity, normalized joint state, torso-frame root velocity, pelvis
height, and the previous action. It receives no demonstration or reference
information. BONES-SEED is used only to define the fixed fallen reset pose in
the current baseline.

## Model and outputs

The vendored `g1_29dof_rev_1_0` model comes from MuJoCo Menagerie under its
included BSD-3-Clause license. Its position actuators were replaced with
unit-gear torque motors so PD control can be implemented explicitly in Python.

Generated demonstrations, calibration results, checkpoints, plots, metrics,
and rollout recordings are written beneath `datasets/processed/` and
`artifacts/`.

## Reference

This project builds on the action-prior approach introduced by Shivam Sood et
al. in [APEX: Action Priors Enable Efficient Exploration for Robust Motion
Tracking on Legged Robots](https://arxiv.org/pdf/2505.10022) (arXiv:2505.10022).

## Disclaimer

OpenAI Codex was used to assist with the development and documentation of this
project.
