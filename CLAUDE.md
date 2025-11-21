# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project for training agents to play the Atari game **Galaxian (ALE/Galaxian-v5)** using Gymnasium and ALE-Py. The codebase implements three different RL algorithms from scratch:

1. **DQN (Deep Q-Network)**: Value-based RL with experience replay
2. **A2C (Advantage Actor-Critic)**: Policy gradient method with n-step rollouts and GAE(λ)
3. **Dueling Double DQN with PER**: Advanced DQN variant with Dueling architecture, Double Q-learning, and Prioritized Experience Replay

All implementations are designed to run in Google Colab with checkpoints saved to Google Drive.

## Architecture

### Environment Preprocessing Pipeline

The codebase uses custom Gymnasium wrappers for Atari preprocessing (defined in cell 2):

1. **`CustomAtariPreprocessing`**: Applies frame skipping (4 frames), grayscale conversion, and resizing to 84x84
2. **`CustomFrameStack`**: Stacks the last 4 frames to capture temporal dynamics
3. **`make_galaxian_env()`**: Factory function that composes both wrappers

**Final observation shape**: `(4, 84, 84)` - 4 grayscale frames, channel-first format

**Layout robustness**: All networks auto-detect and handle both `(C, H, W)` and `(H, W, C)` formats by checking dimensions and permuting as needed.

### Neural Network Architectures

#### DQN Network (cell 3: `DQN` class)
- **Backbone**: 3-layer CNN (Nature DQN architecture):
  - Conv1: 32 filters, 8x8 kernel, stride 4
  - Conv2: 64 filters, 4x4 kernel, stride 2
  - Conv3: 64 filters, 3x3 kernel, stride 1
- **Head**: 2 fully-connected layers (512 → n_actions)
- **Output**: Q-values for all actions

#### A2C Network (cell 5: `A2CNet` class)
- **Shared backbone**: Same 3-layer CNN as DQN
- **Actor head**: FC layers outputting action logits
- **Critic head**: FC layers outputting state value V(s)
- Both heads diverge after the shared convolutional features

#### Dueling DQN Network (cell 7: `DuelingDQN` class)
- **Shared backbone**: Same 3-layer CNN
- **Value stream**: Estimates V(s)
- **Advantage stream**: Estimates A(s,a)
- **Aggregation**: Q(s,a) = V(s) + (A(s,a) - mean(A))

### Training Details

#### DQN (`train_dqn()` in cell 3)
- **Replay buffer**: Standard uniform sampling (100K capacity)
- **Target network**: Updated every 1,000 steps
- **Epsilon decay**: Linear from 1.0 → 0.1 over 300 episodes
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: MSE between Q(s,a) and r + γ max Q'(s',a')

#### A2C (`train_a2c()` in cell 5)
- **Rollout length**: 5 steps (n-step returns)
- **GAE(λ)**: λ=0.95 for advantage estimation
- **Optimizer**: RMSprop (lr=2.5e-4, eps=1e-5)
- **Loss components**:
  - Policy loss: -log π(a|s) * advantage
  - Value loss: MSE on returns
  - Entropy regularization: coefficient 0.01

#### Dueling Double DQN + PER (`train_dueling_double_dqn_per()` in cell 7)
- **Replay buffer**: Prioritized Experience Replay
  - Segment trees for O(log n) sampling/updates
  - α=0.6 (priority exponent), β anneals 0.4 → 1.0
- **Double Q-learning**: Action selection from online net, evaluation from target net
- **Target network**: Updated every 1,000 steps
- **Weighted loss**: TD errors weighted by importance sampling ratios
- **Logging**: Saves reward plots (PNG) and CSV every 200 episodes

### Checkpoint Management

All training functions save periodic checkpoints to Google Drive:
- **Interval**: Every 50 episodes (configurable via `save_interval`)
- **Location**: Specified by `checkpoint_dir` parameter (e.g., `/content/drive/MyDrive/galaxian_rl/dqn`)
- **Checkpoint contents**:
  - Model weights (q_net, target_net, optimizer state)
  - Training metadata (episode, global_step, rewards history)
  - Architecture info (input_shape, n_actions)

**Final weights**: Saved as `{algorithm}_galaxian_final.pth` (state_dict only for inference)

### Policy Interface for Evaluation

Each algorithm provides a `{Algorithm}Policy` class with standardized interface:
```python
policy = DQNPolicy(q_net)  # or A2CPolicy(a2c_net)
action = policy(obs, info)  # obs: (C,H,W) uint8
```
- Takes observation and info dict
- Returns greedy action (argmax for DQN, argmax over softmax for A2C)
- Used for evaluation/competition scenarios

## Development Commands

### Installing Dependencies
```bash
pip install gymnasium[atari] ale-py autorom imageio imageio-ffmpeg torch torchvision
AutoROM --accept-license
```

### Running Training (in Jupyter/Colab)

**Mount Google Drive** (for checkpoints):
```python
from google.colab import drive
drive.mount('/content/drive')
BASE_DIR = "/content/drive/MyDrive/galaxian_rl"
```

**Train DQN**:
```python
dqn_dir = f"{BASE_DIR}/dqn"
q_net = train_dqn(checkpoint_dir=dqn_dir, total_episodes=1000)
```

**Train A2C**:
```python
a2c_dir = f"{BASE_DIR}/a2c"
a2c_net = train_a2c(checkpoint_dir=a2c_dir, total_episodes=500)
```

**Train Dueling DDQN + PER**:
```python
model = train_dueling_double_dqn_per(
    checkpoint_dir=f"{BASE_DIR}/dueling_ddqn_per",
    total_episodes=1000,
    save_interval=50,
    plot_interval=50
)
```

### Loading Trained Models

**Load checkpoint** (full training state):
```python
checkpoint = torch.load("/path/to/checkpoint.pth")
q_net.load_state_dict(checkpoint["q_net"])
episode = checkpoint["episode"]
rewards = checkpoint["rewards"]
```

**Load final weights** (inference only):
```python
q_net = DQN(input_shape, n_actions)
q_net.load_state_dict(torch.load("/path/to/final.pth"))
q_net.eval()
```

## Key Implementation Notes

### Frame Processing
- All networks expect **float tensors normalized to [0,1]** (normalization happens inside `forward()`)
- Input is stored as **uint8 in replay buffers** to save memory
- Conversion to float happens during mini-batch processing

### Observation Format Handling
All networks include automatic layout detection:
```python
# Handles both (B,C,H,W) and (B,H,W,C)
if x.shape[1] != self.expected_c and x.shape[-1] == self.expected_c:
    x = x.permute(0, 3, 1, 2)
```
This ensures compatibility regardless of wrapper output format.

### GPU Usage
- Device selection: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- All tensors and models are moved to device automatically
- Training is significantly faster on GPU (recommended for Dueling DDQN+PER)

### Hyperparameter Tuning
Key hyperparameters to adjust:
- **DQN**: `epsilon_decay_episodes`, `target_update_interval`, `replay_size`
- **A2C**: `rollout_length`, `gae_lambda`, `entropy_coef`
- **Dueling DDQN+PER**: `per_alpha`, `per_beta_start/end`, `buffer_capacity`

### Logging and Visualization
- **Dueling DDQN+PER** automatically generates:
  - Reward plots with moving average (100 episodes)
  - CSV logs of episode rewards
  - Saved to checkpoint directory every `plot_interval` episodes
- Other algorithms log to console only (rewards can be extracted from checkpoints)
