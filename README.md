# Aesthetic-Crop-Selection-with-RLHF
explores automatic image cropping and composition by combining a pre-trained NIMA model with reinforcement learning from human feedback (RLHF). The system uses NIMA to score candidate crops and then refines its selection policy through human-aligned reinforcement learning, aiming to produce visually appealing images without manual labeling.

## Project Structure

- `scorer/` - Aesthetic scoring models and training
- `rl_env/` - Reinforcement learning environment for frame selection
- `actor/` - RL agent training and inference
- `utils/` - Utility functions for image processing
- `data/` - Data storage and processing
- `results/` - Output results and visualizations

## Steps

1. **Scorer (Aesthetic Critic)** - Neural network for beauty scoring
2. **Sliding Window + Scorer Test** - Test the scorer on image crops
3. **RL Environment** - Define the frame selection environment
4. **Actor Training** - Train RL agent with PPO/REINFORCE
5. **Combine and Evaluate** - End-to-end system testing
6. **Incremental Improvement** - Continuous learning strategy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See individual step implementations for detailed usage instructions.