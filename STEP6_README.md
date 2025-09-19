# Step 6: Incremental Improvement Strategy (RLHF-inspired)

This implements a practical 3-stage incremental improvement approach inspired by RLHF (Reinforcement Learning from Human Feedback) to create a personalized aesthetic photographer.

## 🎯 Strategy Overview

### Stage 1: Bootstrap (NIMA-style)
- **Goal**: Train initial policy using pre-trained aesthetic scoring
- **Input**: Large image + pre-trained aesthetic scorer
- **Output**: Bootstrap policy that can find reasonable crops
- **Why**: Quick start with general aesthetic sense, no human data needed

### Stage 2: Personalize (Human Preferences)
- **Goal**: Collect human preferences and train reward model
- **Input**: Bootstrap policy + human preference pairs
- **Output**: Human reward model that learns personal taste
- **Why**: Capture subjective aesthetic preferences that NIMA can't

### Stage 3: Fine-tune (Combined Reward)
- **Goal**: Combine NIMA + human reward for final policy
- **Input**: Bootstrap policy + human reward model
- **Output**: Personalized policy with both general and personal aesthetic sense
- **Why**: Best of both worlds - general aesthetics + personal style

## 🚀 Usage

### Stage 1: Bootstrap Training
```bash
python step6_incremental_improvement.py --stage 1 --image image.jpg --timesteps 20000
```
This trains an initial policy using the pre-trained aesthetic scorer.

### Stage 2: Collect Human Preferences
```bash
# Generate preference pairs for annotation
python step6_incremental_improvement.py --stage 2 --image image.jpg --num_pairs 100

# Annotate preferences using GUI tool
python preference_annotator.py --preferences results/step6/preference_pairs.json

# Train human reward model on annotated data
python step6_incremental_improvement.py --stage 2 --preference_data results/step6/preference_pairs.json --epochs 50
```

### Stage 3: Fine-tune with Combined Reward
```bash
python step6_incremental_improvement.py --stage 3 \
  --image image.jpg \
  --human_reward_model results/step6/human_reward_model.pth \
  --timesteps 10000
```

## 🎨 Preference Annotation Tool

The `preference_annotator.py` provides a user-friendly GUI for annotating crop preferences:

### Features:
- Side-by-side crop comparison
- Keyboard shortcuts (1=A better, 2=B better, 0=no preference)
- Progress tracking
- Save/resume functionality
- Navigation between pairs

### Controls:
- **1** - Crop A is better
- **2** - Crop B is better  
- **0** - No preference
- **Left/Right arrows** - Navigate
- **S** - Save progress

## 📊 Expected Results

### Stage 1 (Bootstrap):
- Policy learns to find aesthetically pleasing crops
- Uses general aesthetic principles from pre-trained model
- Good starting point but not personalized

### Stage 2 (Human Preferences):
- Reward model learns your specific aesthetic taste
- Captures subjective preferences (composition, color, style)
- Small model that can be trained with limited data

### Stage 3 (Combined):
- Policy combines general aesthetics + personal taste
- 70% NIMA score + 30% human preference score
- Personalized photographer that reflects your style

## 🔧 Technical Details

### Human Reward Model:
- Small neural network (EfficientNet backbone + preference head)
- Trained on preference pairs using binary classification
- Concatenates features from both crops for comparison
- Frozen backbone for efficiency

### Combined Reward Function:
```python
combined_reward = 0.7 * (aesthetic_score - 1) / 9 + 0.3 * human_score
```

### Data Flow:
1. **Bootstrap**: NIMA scorer → PPO policy
2. **Collect**: Policy → Crop pairs → Human annotation
3. **Train**: Preference pairs → Human reward model
4. **Fine-tune**: Combined reward → Final policy

## 📈 Performance Expectations

### With Limited Human Data (10-50 pairs):
- Start with reward model fine-tuning only
- Keep policy frozen initially
- Focus on high-quality preference annotations

### With More Human Data (100+ pairs):
- Full end-to-end fine-tuning
- Policy can learn complex preference patterns
- Better personalization and style consistency

## 🎯 Best Practices

### Preference Collection:
- Annotate diverse crop types (close-ups, wide shots, different compositions)
- Be consistent in your preferences
- Focus on crops that clearly differ in aesthetic quality
- Skip pairs where you have no clear preference

### Training:
- Start with small datasets and gradually increase
- Monitor reward model accuracy on held-out data
- Use early stopping to prevent overfitting
- Combine with data augmentation if needed

### Evaluation:
- Compare final policy against bootstrap policy
- Test on new images not seen during training
- Measure both aesthetic scores and personal preference alignment
- Visualize crop selection patterns

## 🔄 Iterative Improvement

This pipeline is designed for iterative improvement:

1. **Initial Run**: Complete all 3 stages with basic setup
2. **Collect More Data**: Add more preference pairs based on policy behavior
3. **Retrain**: Re-run Stage 2 and 3 with expanded dataset
4. **Evaluate**: Compare new policy with previous version
5. **Repeat**: Continue until satisfied with personalization

## 🎨 Customization Options

### Reward Weighting:
- Adjust NIMA vs human preference weights
- Experiment with different combinations (e.g., 60/40, 80/20)
- A/B test different reward functions

### Model Architecture:
- Try different backbones for human reward model
- Experiment with policy network architectures
- Use different feature extractors

### Training Strategy:
- Implement curriculum learning
- Use different RL algorithms (SAC, TD3)
- Add regularization techniques

## 🚨 Troubleshooting

### Common Issues:
- **Low preference data**: Start with reward model only, collect more data
- **Overfitting**: Reduce model size, add regularization, early stopping
- **Poor convergence**: Check reward scaling, adjust learning rates
- **Inconsistent preferences**: Review annotation guidelines, retrain annotators

### Performance Tips:
- Use GPU for faster training
- Batch preference collection for efficiency
- Cache precomputed features when possible
- Monitor training curves and adjust hyperparameters

This incremental approach gives you a personalized aesthetic photographer that learns your specific taste while maintaining general aesthetic principles!
