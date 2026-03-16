import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed

# Import the new reusable agent
from async_inference_agent import AsyncInferenceAgent


class DatasetEvaluator:
    def __init__(self, agent: AsyncInferenceAgent, dataset_repo: str, dataset_root: str, episode_index: int):
        self.agent = agent
        self.dataset_repo = dataset_repo
        self.dataset_root = dataset_root
        self.episode_index = episode_index
        self.logger = logging.getLogger(__name__)

    def load_dataset(self) -> Tuple[LeRobotDataset, int, int]:
        self.logger.info(f"Loading dataset {self.dataset_repo} from {self.dataset_root}...")
        dataset = LeRobotDataset(repo_id=self.dataset_repo, root=self.dataset_root)

        if self.episode_index >= dataset.num_episodes:
            raise ValueError(f"EPISODE_INDEX={self.episode_index} out of range [0, {dataset.num_episodes - 1}]")

        ep_info = dataset.meta.episodes[self.episode_index]
        from_idx = ep_info["dataset_from_index"]
        to_idx = ep_info["dataset_to_index"]

        # Apply drop_n_last_frames logic
        drop_n = self.agent.drop_n_last_frames
        if drop_n > 0:
            self.logger.info(f"Adjusting evaluation range: Skipping last {drop_n} frames per policy config.")
            to_idx -= drop_n
        
        return dataset, from_idx, to_idx

    def _extract_observation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only model-relevant fields from a dataset item."""
        obs = {}
        for k, v in item.items():
            if k.startswith("observation."):
                obs[k] = v
        return obs

    def run(self, output_plot: str = "evaluation_plots.png"):
        dataset, from_idx, to_idx = self.load_dataset()
        episode_len = to_idx - from_idx

        self.logger.info(f"Evaluating Episode {self.episode_index} (Frames: {episode_len}, Indices: {from_idx}-{to_idx})...")

        # Determine GT key
        gt_key = "action" if "action" in dataset.features else "observation.state.q_pos"
        if gt_key != "action":
            self.logger.warning(f"'action' key missing in dataset, using '{gt_key}' as GT.")

        self.logger.info("Pre-loading GT actions for comparison...")
        gt_actions = torch.stack([dataset[i][gt_key] for i in range(from_idx, to_idx)]).cpu().numpy()

        predicted_actions = np.full_like(gt_actions, np.nan)
        inference_times = []
        
        self.agent.reset()
        
        progress = tqdm.tqdm(range(episode_len), total=episode_len)
        for rel_t in progress:
            abs_t = from_idx + rel_t
            item = dataset[abs_t]
            
            # Extract observation from dataset item
            observation = self._extract_observation(item)
            
            # Run inference via agent
            action, info = self.agent.select_action(observation)
            
            # Ensure action shape compatibility (e.g. handle batch dim if leaked)
            if action.ndim > 1:
                action = action.squeeze()
            
            predicted_actions[rel_t] = action
            
            if info.get("did_inference", False):
                inference_times.append(info["latency"])

        self._print_error_analysis(gt_actions, predicted_actions, episode_len)
        metrics = self._compute_metrics(gt_actions, predicted_actions, inference_times)
        
        self._plot_results(gt_actions, predicted_actions, metrics["valid_mask"], output_plot, metrics["mse_overall"])
        
        return metrics

    def _print_error_analysis(self, gt: np.ndarray, pred: np.ndarray, length: int):
        print("\nError analysis for last 10 steps:")
        for i in range(max(0, length - 10), length):
            err = pred[i] - gt[i]
            mae = np.mean(np.abs(err))
            max_err = np.max(np.abs(err))
            print(f"Step {i}: MAE={mae:.4f}, MaxErr={max_err:.4f}")

    def _compute_metrics(self, gt: np.ndarray, pred: np.ndarray, latencies: list) -> Dict[str, Any]:
        valid_mask = ~np.isnan(pred).any(axis=1)
        if valid_mask.sum() == 0:
            raise RuntimeError("No valid predictions were produced.")

        err = pred[valid_mask] - gt[valid_mask]
        sqerr = err**2
        abserr = np.abs(err)
        
        mse_per_joint = sqerr.mean(axis=0)
        mae_per_joint = abserr.mean(axis=0)

        print("\n" + "=" * 52)
        print(f"ASYNC DATASET EVAL REPORT | episode={self.episode_index}")
        print("=" * 52)
        print(f"Steps: {len(gt)}")
        print(f"Chunk inferences: {len(latencies)}")
        if latencies:
            print(f"Inference time avg: {1000 * np.mean(latencies):.2f} ms")
            print(f"Inference time p95: {1000 * np.percentile(latencies, 95):.2f} ms")
        print(f"MSE (overall): {float(mse_per_joint.mean()):.6f}")
        print(f"MAE (overall): {float(mae_per_joint.mean()):.6f}")
        print("Per-joint MAE:", np.array2string(mae_per_joint, precision=5, separator=", "))
        print("Per-joint MSE:", np.array2string(mse_per_joint, precision=5, separator=", "))
        print("=" * 52)

        return {
            "mse_overall": float(mse_per_joint.mean()),
            "mae_overall": float(mae_per_joint.mean()),
            "valid_mask": valid_mask
        }

    def _plot_results(self, gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray, filename: str, overall_mse: float):
        n_joints = gt.shape[1]
        cols = 3
        rows = (n_joints + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows), squeeze=False)
        axes = axes.flatten()
        x = np.arange(len(gt))

        for j in range(n_joints):
            ax = axes[j]
            ax.plot(x, gt[:, j], color="black", alpha=0.7, linewidth=1.2, label="GT")
            y_pred = pred[:, j].copy()
            y_pred[~valid_mask] = np.nan
            ax.plot(x, y_pred, color="red", linestyle="--", alpha=0.85, linewidth=1.2, label="Pred")
            ax.set_title(f"Joint {j}")
            ax.grid(alpha=0.25)
            if j == 0:
                ax.legend()

        for k in range(n_joints, len(axes)):
            axes[k].axis("off")

        plt.suptitle(f"Async-style rollout vs GT | episode={self.episode_index} | overall MSE={overall_mse:.6f}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=140)
        print(f"Saved plot: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeRobot Policy on Dataset (Async Wrapper)")
    parser.add_argument("--checkpoint", type=str, help="Path to policy checkpoint")
    parser.add_argument("--repo-id", type=str, default="26-01n02", help="Dataset Repo ID")
    parser.add_argument("--root", type=str, default="/inspire/qb-ilm/project/robot-decision/public/datasets/autolife/anybox_data/26-01n02", help="Dataset root directory")
    parser.add_argument("--episode", type=int, default=2, help="Episode index to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_plots.png", help="Output plot filename")
    parser.add_argument("--disable-async", action="store_true", help="Force sync inference (n_action_steps=1)")
    parser.add_argument("--device", type=str, metavar="DEVICE", help="Device (cuda/cpu)")
    parser.add_argument("--ensemble-weights", type=float, nargs="+", help="Weights for temporal ensembling (newest first). e.g. 0.8 0.2")

    args = parser.parse_args()

    # Fallback logic for backward compatibility (if args.checkpoint is not provided)
    DEFAULT_CHECKPOINT = "/inspire/hdd/project/robot-decision/caijintian-p-caijintian/outputs/train/diffusion_box_2_0312_3/checkpoints/last/pretrained_model"
    checkpoint = args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINT
    
    # Check env var for disable async if not provided via args (compatibility)
    disable_async = args.disable_async or (os.environ.get("DISABLE_ASYNC", "0") == "1")

    init_logging()
    set_seed(42)

    agent = AsyncInferenceAgent(
        checkpoint_dir=checkpoint,
        device=args.device,
        disable_async=disable_async
    )

    if args.ensemble_weights:
        agent.set_aggregation_weights(args.ensemble_weights)

    evaluator = DatasetEvaluator(
        agent=agent,
        dataset_repo=args.repo_id,
        dataset_root=args.root,
        episode_index=args.episode
    )

    evaluator.run(output_plot=args.output)


if __name__ == "__main__":
    main()
