# APEX Robot Get-Up

MuJoCo research code for training the 29-actuator Unitree G1 to stand from a
fallen pose. The long-term goal is a reference-free get-up policy trained with
APEX-style action priors.

## Current status

The project has a successful rigid-ground residual-PPO baseline from the fixed
frame-zero supine pose.

In its recorded deterministic evaluation, it reached sustained standing in
7.60 s, finished at 0.795 m pelvis height and 0.985 uprightness, and remained
stable for 4.62 s. This policy is still reference-conditioned: it corrects the
prior trajectory and is not yet the final reference-free APEX policy.

Full APEX task/style critics, randomized fallen
poses, multiple demonstrations, and domain randomization remain future work.

## Training framework

- MuJoCo G1 with 29 actuated joints on rigid ground.
- Native position actuators with the official joint-group PD gains.
- 0.002 s physics timestep and 8 substeps per action (62.5 Hz control).
- Native 120 Hz reference queried by simulation time, with configurable
  quaternion interpolation.
- Physical residual action
  `q_cmd = clip_to_limits(q_ref + 0.25 * policy_action)`.
- 255-Dimensional reference-conditioned observation
- MuJoCo rollout collection with batched policy inference.
- Stable-Baselines3 PPO is used by the successful configuration; the native
  PyTorch PPO backend remains available for experiments.
- Deterministic frame-zero evaluation selects the best checkpoint independently
  of periodic checkpoint snapshots.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
```

The live MuJoCo viewer requires a desktop/OpenGL session.

## Data and validation

Place the A475 (get up from supine) motion at:

```text
datasets/stand_up_lying_R_002__A475_new.csv
```

## Train

The main residual-policy experiment uses 20 environments, 2,048 rollout steps,
SB3 PPO, and frame-zero resets:

```bash
python scripts/train_apex_prior.py \
  --config configs/residual_policy_milestone_1.yaml
```

YAML contains the complete simulator, controller, reward, PPO, normalization,
and output configuration. Explicit CLI options override supported YAML values.

## Evaluate and render

Evaluate the preserved successful policy with the configuration from its
original run:

```bash
python scripts/evaluate_residual_ppo.py \
  --checkpoint artifacts/baselines/stage-b-frame0-residual-success-seed1-57M/best.zip \
  --config configs/stage_b_frame0_paper_timing.yaml \
  --residual-scale 0.25 \
  --mode residual \
  --episodes 5
```

Render one deterministic episode by adding `--render` and using one episode:

```bash
python scripts/evaluate_residual_ppo.py \
  --checkpoint artifacts/baselines/stage-b-frame0-residual-success-seed1-57M/best.zip \
  --config configs/stage_b_frame0_paper_timing.yaml \
  --residual-scale 0.25 \
  --mode residual \
  --episodes 1 \
  --render
```

Training outputs, checkpoints, normalization statistics, metrics, and plots are
stored under `artifacts/`. See [COMMANDS.md](COMMANDS.md) for additional replay,
ablation, plotting, and testing commands.

## References

- [APEX: Action Priors Enable Efficient Exploration for Robust Motion Tracking
  on Legged Robots](https://arxiv.org/pdf/2505.10022)
- [Demonstration-Guided Humanoid Stand-Up on an Emulated Deformable
  Surface](https://arxiv.org/pdf/2608.20852)

The vendored MuJoCo Menagerie Unitree G1 model retains its included
BSD-3-Clause license.

## Disclaimer

OpenAI Codex was used to assist with development and documentation.
