# Wargame RL Project Analysis

## Project Overview
This is a reinforcement learning environment for wargame simulations built with:
- PyTorch for deep learning models
- Gymnasium for environment interface
- Python with modern development tools (UV, Just, Ruff, etc.)

## Key Components

### Environment
- `WargameEnv` in `wargame_rl.wargame.envs.wargame`
- Supports configurable number of wargame models and objectives
- 2D board environment with movement actions
- Action space: Tuple(Discrete(5), Discrete(5)) where 5 actions = [up, down, left, right, stay]

### Models
- `DQN_MLP` - Multi-layer perceptron DQN (Legacy model, support will be dropped in the future)
- `DQN_Transformer` - Transformer-based DQN (This is the model we are currently using)
- Both models support different network architectures

### Configuration
- Environment configs in `examples/env_config/`
- Support for different board sizes and numbers of models/objectives
- Configurable parameters like objective radius, group cohesion, etc.
- Support for opponent deployment zones (recently added feature)

## How to Run

Look at the justfile for available commands.

### Environment Test
You can use pytest
```bash
uv run pytest --verbose tests
```

### Training (with example config)
```bash
# Without wandb (since it requires API key)
python train.py --env-config-path examples/env_config/example.yaml --max-epochs 1 --network-type transformer
```

### Simulation
For now, you cannot really see what happens. So simulating is mostly to see if the codes runs.

## Key Features
1. Configurable wargame environments with different board sizes
2. Support for multiple wargame models and objectives
3. Both MLP and Transformer DQN architectures
4. Proper reinforcement learning interface with Gymnasium
5. Modular design with clear separation of environment and model components
6. Support for opponent deployment zones (recently added feature)
7. Support for per-entity configuration in environment configs

## Recent Changes
- Added support for opponent deployment zones to enable adversarial play
- Added support for optional per-entity configuration in environment configs
- Enhanced wandb logging with video recording capabilities

## Current Status
The environment and models work correctly. The main limitation is wandb configuration, but the core functionality is intact. The DQN_Transformer model is the current focus for training and development.
