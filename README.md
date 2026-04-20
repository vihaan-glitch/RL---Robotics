# RL Robotics — Robotic Arm Reach with PPO

A reinforcement learning project that trains a simulated robotic arm to reach random target positions using Proximal Policy Optimization (PPO).

## Overview

This project uses a custom Gymnasium environment built on MuJoCo to simulate a **KUKA iiwa robotic arm**. The agent learns to control the arm's joints to reach randomly placed 3D target points, trained entirely from scratch using PPO from Stable-Baselines3.

## Demo

The trained agent controls a 7-DOF robotic arm in a MuJoCo physics simulation, learning to minimize the distance between the end-effector and a randomly placed target.

## Project Structure

```
RL---Robotics/
├── arm_reach_env.py       # Custom Gymnasium environment (PyBullet simulation)
├── train.py               # PPO training script
├── evaluate.py            # Evaluate a trained model
├── requirements.txt       # Python dependencies
├── models/                # Saved model checkpoints
│   └── ppo_arm_reach.zip  # Pre-trained model
└── logs/                  # Training logs (TensorBoard)
```

## Environment Details

**`ArmReachEnv`** — a custom `gym.Env` wrapping PyBullet:

| Property | Value |
|---|---|
| Robot | KUKA iiwa (7 DOF) |
| Action Space | Continuous joint position targets, normalized to `[-1, 1]` |
| Observation Space | Joint positions + velocities + target position + end-effector position + distance |
| Episode Length | Max 500 steps |
| Success Condition | End-effector within 5cm of target |

**Reward shaping:**
- Distance penalty: `-distance` each step
- Progress bonus: reward for getting closer to the target
- Success bonus: `+100` when within 5cm
- Time penalty: `-0.01` per step to encourage efficiency

## Getting Started

### Prerequisites

- Python 3.8+
- (Optional) CUDA GPU for faster training

### Installation

```bash
git clone https://github.com/vihaan-glitch/RL---Robotics.git
cd RL---Robotics
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

Training runs for 500,000 timesteps. Checkpoints are saved to `models/` every 10,000 steps, and the best model is saved to `models/best_model/`.

Optional: Monitor training with TensorBoard:
```bash
pip install tensorboard
tensorboard --logdir logs/tensorboard/
```

### Evaluate

```bash
python evaluate.py
```

This loads the trained model and runs it in the GUI-rendered simulation so you can watch the arm move.

## Dependencies

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
- [MuJoCo](https://mujoco.org/) — Physics simulation
- [Gymnasium](https://gymnasium.farama.org/) — RL environment interface
- [NumPy](https://numpy.org/)

## Training Notes

- Observation and reward normalization is applied via `VecNormalize` for training stability
- Entropy coefficient (`ent_coef=0.01`) encourages exploration early in training
- The `__pycache__` folder can be safely added to `.gitignore`
- MuJoCo requires a valid installation — see [MuJoCo installation guide](https://mujoco.readthedocs.io/en/stable/python.html)

## Future Ideas

- Add obstacle avoidance
- Train on more complex manipulation tasks (grasp, push, stack)
- Deploy to a real robot using ROS
- Try SAC or TD3 for comparison

## License

MIT