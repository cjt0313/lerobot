# Anybox Inference Diffusion Policy

This folder contains scripts for running asynchronous inference with LeRobot policies, specifically designed for diffusion policies with temporal ensembling capabilities.

## Scripts

- `async_inference_agent.py`: A reusable `AsyncInferenceAgent` class that handles model loading, async inference queue management, and temporal ensembling (weighted averaging of overlapping action chunks).
- `eval_dataset_async.py`: A script to evaluate a trained policy against a recorded dataset episode, visualizing the ground truth vs predicted trajectory.

## Installation

Ensure you have the `lerobot` package installed in your environment. These scripts are part of the `lerobot.anybox` namespace.

## Usage

### 1. `AsyncInferenceAgent` Usage in Python

You can import the agent in your own inference scripts:

```python
from lerobot.anybox.inference.diffusion_policy.async_inference_agent import AsyncInferenceAgent

# Initialize the agent
agent = AsyncInferenceAgent(
    checkpoint_dir="/path/to/your/checkpoint",
    device="cuda",
    use_amp=False
)

# Optional: Set temporal ensembling weights (e.g., blend last 2 predictions)
# Weights are applied to [newest_prediction, older_prediction, oldest_prediction...]
agent.set_aggregation_weights([0.6, 0.4]) 

# Loop for inference
# observation is a dict matching the policy expected input (e.g., {"observation.images": ..., "observation.state": ...})
action = agent.predict_action(observation)

# 'action' is a single step action (Tensor) ready to be executed
print("Action:", action)
```

### 2. Running Evaluation (`eval_dataset_async.py`)

This script runs the policy against a recorded episode from a dataset to check performance (e.g., MSE) and visual smoothness.

**Basic Usage:**

```bash
python eval_dataset_async.py \
  --checkpoint /path/to/checkpoint \
  --repo-id <dataset_repo_id> \
  --root /path/to/dataset/root \
  --episode 0

# example
python ws/lerobot/src/lerobot/anybox/inference/diffusion_policy/eval_dataset_async.py \
  --repo-id /inspire/hdd/project/robot-decision/caijintian-p-caijintian/outputs/train/2025-02-18/14-04-42_autolife_s1_diffusion_default \
  --episodes 0 \
  --ensemble-weights 0.4 0.6
```



**With Temporal Ensembling:**
Use `--ensemble-weights` to specify weights for overlapping action chunks. The weights should sum to (or close to) 1.0, but the code handles normalization if needed mostly implicitly by the user logic.
This example gives 40% weight to the newest inference chunk, 30% to the previous, and 30% to the one before that.

```bash
python eval_dataset_async.py \
  --checkpoint /path/to/checkpoint \
  --repo-id <dataset_repo_id> \
  --root /path/to/dataset/root \
  --episode 0 \
  --ensemble-weights 0.4 0.3 0.3 \
  --output evaluation_result.png
```

**Arguments:**

- `--checkpoint`: Path to the policy checkpoint directory (containing `config.yaml` and `pretrained_model`).
- `--repo-id`: LeRobot dataset repository ID (e.g., `creator/dataset-name`).
- `--root`: Root directory where datasets are stored.
- `--episode`: Index of the episode to evaluate (default: 0).
- `--output`: Filename for the output plot (default: `evaluation_plots.png`).
- `--device`: Device to run inference on (`cuda` or `cpu`).
- `--ensemble-weights`: Space-separated list of floats for temporal ensembling. First number is for the most recent prediction.
- `--disable-async`: (Flag) Force synchronous inference (n_action_steps=1), effectively disabling the receding horizon control.

## How Temporal Ensembling Works

The `AsyncInferenceAgent` maintains a buffer of recent action predictions covering the future horizon.
When `set_aggregation_weights` is used, the final action at time `t` is a weighted average of predictions made at different past time steps that cover time `t`.

Example with weights `[0.6, 0.4]`:
- At step `t`, we have a new prediction chunk `P_t` and a previous prediction chunk `P_{t-1}` from the last step.
- The action executed is `0.6 * P_t[0] + 0.4 * P_{t-1}[1]`.
