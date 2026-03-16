import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


class AsyncInferenceAgent:
    """
    Agent for asynchronous inference with LeRobot policies.
    Handles policy loading, preprocessing, postprocessing, and action queue management.
    Designed for real-world inference integration.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        disable_async: bool = False,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading policy from {checkpoint_dir}...")
        self.policy = DiffusionPolicy.from_pretrained(checkpoint_dir)
        self.policy.eval()

        if device:
            self.policy.to(device)
            # Update config device to match manual override
            self.policy.config.device = device
        
        # Fallback to config device if not overridden
        self.device = getattr(self.policy.config, "device", "cuda" if torch.cuda.is_available() else "cpu")

        if disable_async:
            self.logger.warning("Disabling Async Inference (forcing n_action_steps=1).")
            self.policy.config.n_action_steps = 1
        
        self.drop_n_last_frames = getattr(self.policy.config, "drop_n_last_frames", 0)
        self.logger.info(f"Policy Config: drop_n_last_frames={self.drop_n_last_frames}, n_action_steps={self.policy.config.n_action_steps}")

        self._setup_processors()
        
        # Ensembling params
        self.aggregation_weights = None
        self.past_action_chunks = [] # List of tuples (start_step, chunk_tensor)

        self.reset()
        
    def set_aggregation_weights(self, weights: list[float]):
        """
        Enable Temporal Ensembling with custom weights.
        
        Args:
           weights: List of weights [w_0, w_1, ... w_k] corresponding to the age of the prediction.
                    w_0 is applied to the newest chunk (generated at current step t).
                    w_1 is applied to the chunk generated at t-1.
                    The logic will automatically normalize the weights if history is shorter than len(weights).
        """
        self.aggregation_weights = weights
        self.logger.info(f"Temporal Ensembling Enabled. Weights: {self.aggregation_weights}")

    def _setup_processors(self):
        # Match processor device.
        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.checkpoint_dir,
            preprocessor_overrides=preprocessor_overrides,
        )

        # Patch postprocessor to accept tuple (action, observation) for delta action conversion
        # This is often needed for diffusion policies that output delta actions
        # We wrap the original method to handle the tuple input case
        original_to_transition = self.postprocessor.to_transition

        def simple_action_obs_to_transition(batch):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                action, observation = batch
                return {"action": action, "observation": observation}
            # Fallback for standard usage (usually just action dict or tensor)
            return batch

        self.postprocessor.to_transition = simple_action_obs_to_transition

    def reset(self):
        """Reset policy internal state (action queues, etc.)."""
        self.policy.reset()
        self.past_action_chunks = []
        self._current_step_cnt = 0

    def select_action(self, observation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run inference for a single step.
        
        Args:
            observation: Dictionary of observation keys (e.g. 'observation.state.q_pos') 
                         mapping to values (tensors or arrays). 
                         Expects single-step inputs (no batch dim), will unsqueeze internally.
        
        Returns:
            action: Numpy array of the executed action for this step.
            info: Dictionary containing metadata (e.g. latency).
        """
        # 1. Preprocess: Add batch dimension if needed and run preprocessor
        obs_dict = {}
        for k, v in observation.items():
            if isinstance(v, (np.ndarray, float, int)):
                # Improve robustness: if scalar, make it 1D array first? 
                # LeRobot usually expects at least [D].
                v = np.array(v) if np.isscalar(v) else v
                v = torch.tensor(v, device=self.device)
            elif isinstance(v, torch.Tensor):
                v = v.to(self.device)
            
            # Add batch dimension [B=1, ...] if missing
            # Logic: If dimension matches Expected Shape, it needs batch dim. 
            # But we don't have expected shape handy easily without looking at config. 
            # We assume the user passes a single sample [D] or [C,H,W].
            # So we always unsqueeze(0).
            if v.ndim == 0:
                v = v.unsqueeze(0) # Scalar -> [1]

            obs_dict[k] = v.unsqueeze(0) # [D] -> [1, D]

        # Run preprocessor
        obs_dict = self.preprocessor(obs_dict)

        # 2. Inference
        start_time = time.time()
        
        # Determine if inference will happen (accessing private member _queues for this check)
        # Note: If aggregation is enabled, we FORCE inference every step.
        will_infer = (self.aggregation_weights is not None) or (len(self.policy._queues["action"]) == 0)
        
        if will_infer and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        t0 = time.time()
        with torch.inference_mode():
            if self.aggregation_weights is not None:
                # Temporal Ensembling Mode: Always predict new chunk
                # We need to manually populate the queues logic if we used select_action, 
                # but here we bypass it and use predict_action_chunk directly
                
                # IMPORTANT: predict_action_chunk does NOT update the policy's internal queues.
                # However, the policy usually relies on select_action to update queues (observation history).
                # DiffusionPolicy.select_action calls populate_queues then predict_action_chunk.
                # We must ensure observation history is updated.
                
                # To be safe and reuse LeRobot logic, we can call a method that updates queues but gives us the chunk.
                # But select_action swallows the chunk.
                
                # Workaround: Manually manage queues or force select_action to refill?
                # Actually, simply calling `select_action` updates the queue.
                # But `select_action` returns only 1 action.
                # If we want the FULL CHUNK for ensembling, we need to access it.
                
                # Direct approach: 
                # 1. Update observation queues (populate_queues)
                # 2. Run predict_action_chunk
                
                # Accessing internal method populate_queues
                from lerobot.policies.utils import populate_queues
                
                # IMPORTANT: predict_action_chunk expects 'observation.images' to be stacked if image features exist.
                # Standard select_action handles this. We must replicate it.
                
                # 1. Pre-process dict for queue population
                batch_for_queue = dict(obs_dict)
                if self.policy.config.image_features:
                    # Check if 'observation.images' already exists? No, preprocessor usually gives separated keys.
                    # modeling_diffusion:109 -> batch[OBS_IMAGES] = torch.stack(...)
                    # We need to do this stacking BEFORE populate_queues if we want the queue to have the stacked images?
                    # No, populate_queues takes the raw keys.
                    # WAIT, `select_action` in modeling_diffusion.py:
                    #   if self.config.image_features:
                    #       batch[OBS_IMAGES] = torch.stack(...)
                    #   self._queues = populate_queues(self._queues, batch)
                    # So populate_queues expects the STACKED 'observation.images' key if config.image_features is true!
                    
                    imgs = [batch_for_queue[key] for key in self.policy.config.image_features]
                    # Stack dim=-4 (usually images are [B, C, H, W] -> [B, N_cams, C, H, W])
                    # But here input is [1, C, H, W] or [1, T, C, H, W]??
                    # Preprocessor output usually [1, D] or [1, C, H, W].
                    # Let's check modeling_diffusion.py again.
                    # It stacks dim=-4. If shape is [B, C, H, W], dim -4 is B?? No.
                    # If shape [B, C, H, W] (4 dims). -4 is B.
                    # It expects [B, N_CAM, C, H, W] result.
                    # So inputs must be [B, C, H, W]. Stack adds new dim.
                    
                    batch_for_queue["observation.images"] = torch.stack(imgs, dim=-4)
                    
                self.policy._queues = populate_queues(self.policy._queues, batch_for_queue)
                
                # 2. Run prediction
                # predict_action_chunk ALSO expects 'observation.images' to be present if image_features is set.
                # Because it calls self.diffusion.generate_actions(batch)
                # generate_actions -> _prepare_global_conditioning -> batch[OBS_IMAGES]
                
                # So we pass the batch WITH the stacked images.
                action_chunk = self.policy.predict_action_chunk(batch_for_queue)
                # action_chunk shape: (B, T, D) -> (1, T, D)
                action_tensor = action_chunk # Keep it as chunk for now
                
            else:
                # Standard Receding Horizon Control (LeRobot Default)
                action_tensor = self.policy.select_action(obs_dict)
        
        latency = 0.0
        if will_infer:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.time() - t0

        # 3. Postprocess & Aggregation
        
        if self.aggregation_weights is not None:
            # action_tensor is (1, T, D) - The full chunk
            # Postprocess expects (B, D) usually, wait.
            # Postprocessor usually handles single steps.
            # If we pass (1, T, D), the unnormalizer might handle it if shapes match features.
            # But the DeltaToAbsolute step needs careful handling of state.
            
            # For simplicity, let's post-process the full chunk step-by-step or check if postprocessor handles T dim.
            # LeRobot postprocessors usually handle (B, D).
            # We iterate through the chunk to post-process it effectively.
            
            # However, my simple_action_obs_to_transition patch assumes (action, obs).
            # obs contains "current_state".
            # For a chunk of actions [a_t, a_t+1...], apply current_state to ALL?
            # Yes, standard Delta policy applies current state to the sequence of deltas.
            
            # Postprocess the whole chunk at once (assuming broadcasting works)
            # action_tensor: (1, T, D)
            # obs_dict state: (1, D)
            
            # The DeltaToAbsoluteActionProcessorStep I checked earlier handles broadcasting:
            # if action.ndim == current_state.ndim + 1: state.unsqueeze(-2)
            # So (1, T, D) vs (1, D) works perfectly!
            
            processed_chunk_dict = self.postprocessor((action_tensor, obs_dict))
            
            if isinstance(processed_chunk_dict, dict):
                chunk = processed_chunk_dict["action"]
            else:
                chunk = processed_chunk_dict
            
            # chunk is (1, T, D). Squeeze batch.
            chunk = chunk.squeeze(0).cpu().numpy() # (T, D)
            
            # Add to history
            self.past_action_chunks.append({
                "start_step": self._current_step_cnt,
                "data": chunk
            })
            
            # Prune history based on weight length
            max_hist = len(self.aggregation_weights)
            # We only need chunks that overlap with current step.
            # Actually, `past_action_chunks` grows. prune old ones.
            # A chunk created at `start_step` with length L remains valid until `start_step + L`.
            # We can prune efficiently.
            
            # Aggregation Step
            aggregated_action = np.zeros_like(chunk[0]) # (D,)
            total_weight = 0.0
            
            # Iterate backwards (newest first)
            # weights[0] -> newest chunk (age 0)
            
            kept_chunks = []
            
            # We iterate through past chunks.
            # For a chunk created at S with data C (len L).
            # At current step T (self._current_step_cnt).
            # The relative index is idx = T - S.
            # If 0 <= idx < L: It is a valid prediction for now.
            # The "Age" of this prediction is determined by how long ago it was made.
            # Age = T - S. (0 for current step, 1 for previous step).
            # If Age < len(weights): We use weight[Age].
            
            for item in self.past_action_chunks:
                start_step = item["start_step"]
                chunk_data = item["data"] # (L, D)
                chunk_len = len(chunk_data)
                
                rel_idx = self._current_step_cnt - start_step
                
                # Check if this chunk is still relevant for *future* or *now*
                # If rel_idx >= chunk_len, it's exhausted.
                if rel_idx < chunk_len:
                    kept_chunks.append(item)
                    
                # Check if it covers CURRENT step
                if 0 <= rel_idx < chunk_len:
                    age = rel_idx # How many steps ago was this plan made?
                    # Wait, is age = rel_idx?
                    # If I made a plan at T=100.
                    # At T=100 (Now), rel_idx=0. Age=0.
                    # At T=101, rel_idx=1. Age=1.
                    # Yes.
                    
                    if age < len(self.aggregation_weights):
                        w = self.aggregation_weights[age]
                        pred = chunk_data[rel_idx]
                        aggregated_action += pred * w
                        total_weight += w
            
            self.past_action_chunks = kept_chunks
            
            if total_weight > 1e-6:
                action = aggregated_action / total_weight
            else:
                # Fallback if no weights cover (should not happen if len(weights) > 0 and w0 > 0)
                # Just take newest if available
                if self.past_action_chunks:
                     last = self.past_action_chunks[-1]
                     idx = self._current_step_cnt - last["start_step"]
                     if idx < len(last["data"]):
                         action = last["data"][idx]
                     else:
                         action = np.zeros_like(aggregated_action)
                else:
                     action = np.zeros_like(aggregated_action)
            
            self._current_step_cnt += 1
            
        else:
            # Standard Path
            # action_tensor is (1, D)
            action_dict = self.postprocessor((action_tensor, obs_dict))
            
            # Extract action
            if isinstance(action_dict, dict):
                action = action_dict["action"]
            elif isinstance(action_dict, torch.Tensor):
                action = action_dict
            else:
                raise ValueError(f"Unknown action output type: {type(action_dict)}")

            # Remove batch dimension and convert to numpy
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()
            
            self._current_step_cnt += 1
        
        info = {
            "latency": latency,
            "did_inference": will_infer,
        }
        
        return action, info
