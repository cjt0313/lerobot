#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch
from lerobot.configs.types import FeatureType, PolicyFeature, PipelineFeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    ProcessorStep,
    EnvTransition,
    TransitionKey,
)
from lerobot.processor.delta_processors import AbsoluteToDeltaActionProcessorStep, DeltaToAbsoluteActionProcessorStep
from lerobot.processor.converters import (
    policy_action_to_transition, 
    transition_to_policy_action,
    batch_to_transition,
    transition_to_batch,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from dataclasses import dataclass


@dataclass
class CopyStateToActionStep(ProcessorStep):
    """Copies a state feature to action if action is missing."""
    source_key: str = "observation.state.q_pos"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Debugging
        # print(f"DEBUG: Keys in transition: {list(transition.keys())}")
        # if transition.get(TransitionKey.OBSERVATION):
        #    print(f"DEBUG: Keys in observation: {list(transition[TransitionKey.OBSERVATION].keys())}")
        if transition.get(TransitionKey.ACTION) is None:
            obs = transition.get(TransitionKey.OBSERVATION)
            # Handle recursive search if needed, but for now assuming flat access via helper or direct
            # EnvTransition/RobotObservation allows key access.
            # Observation is usually a dict.
            if obs is not None and self.source_key in obs:
                 # print(f"DEBUG: Copying {self.source_key} to action")
                 transition[TransitionKey.ACTION] = obs[self.source_key]
            # else:
            #     print(f"DEBUG: Source key {self.source_key} not found in observation")
        
        # Ensure action_is_pad is present if action is present
        # This is critical for training compute_loss
        if transition.get(TransitionKey.ACTION) is not None:
             comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
             if "action_is_pad" not in comp_data:
                 # Attempt to synthesize action_is_pad
                 # Check if we have source key pad info?
                 # If not, assume all False (valid data)
                 action_tensor = transition[TransitionKey.ACTION]
                 # action_tensor shape: (B, T, D) or (T, D)
                 # We need (B, T) or (T)
                 if isinstance(action_tensor, torch.Tensor):
                     shape = action_tensor.shape
                     # Typically (B, Horizon, D)
                     if len(shape) >= 2:
                         pad_shape = shape[:-1]
                         comp_data["action_is_pad"] = torch.zeros(pad_shape, dtype=torch.bool, device=action_tensor.device)
                         transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
                         # print("DEBUG: Synthesized action_is_pad")

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if PipelineFeatureType.ACTION not in features:
            features[PipelineFeatureType.ACTION] = {}
        
        # If action features are empty, check if we can source from observation
        if not features[PipelineFeatureType.ACTION]:
            if PipelineFeatureType.OBSERVATION in features and self.source_key in features[PipelineFeatureType.OBSERVATION]:
                source_ft = features[PipelineFeatureType.OBSERVATION][self.source_key]
                features[PipelineFeatureType.ACTION]["action"] = PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=source_ft.shape,
                )
        return features



@dataclass
class SliceObservationStep(ProcessorStep):
    """Slices observation to n_obs_steps."""
    n_obs_steps: int = 2

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION)
        if obs:
            for key, val in obs.items():
                if isinstance(val, torch.Tensor):
                    # Determine time dimension index based on ndim
                    # (T, D) or (T, C, H, W) -> dim 0
                    # (B, T, D) or (B, T, C, H, W) -> dim 1
                    time_dim = 0
                    if val.ndim in (3, 5):
                        time_dim = 1
                    
                    if val.shape[time_dim] > self.n_obs_steps:
                        if time_dim == 0:
                            obs[key] = val[:self.n_obs_steps]
                        elif time_dim == 1:
                            obs[key] = val[:, :self.n_obs_steps]
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class ExtractLastStateStep(ProcessorStep):
    """
    Extracts the last timestep from a state observation to use as current_state 
for delta calculation.
    Assumes inputs are batched (B, T, D) or unbatched (T, D).
    """
    source_key: str = "state"
    dest_key: str = "current_state"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION)
        if obs and self.source_key in obs:
            val = obs[self.source_key]
            if isinstance(val, torch.Tensor):
                # (B, T, D) -> (B, D)
                if val.ndim == 3:
                     obs[self.dest_key] = val[:, -1]
                # (T, D) -> (D)
                elif val.ndim == 2:
                     obs[self.dest_key] = val[-1]
                # (T) -> scalar
                elif val.ndim == 1:
                     obs[self.dest_key] = val[-1]
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We don't strictly need to declare the new feature as it's transient for delta proc
        return features



def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.
    
    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Normalizing the input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the diffusion policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    rename_map = {
        "observation.state.q_pos": "observation.state",
        "observation.images.head_left": "observation.image",
    }
    
    # Remap dataset stats to match renamed observations and synthetic action
    if dataset_stats:
        dataset_stats = dataset_stats.copy()
        for src, dst in rename_map.items():
             if src in dataset_stats:
                 dataset_stats[dst] = dataset_stats[src]
        
        # If action stats are missing but we copy from q_pos, copy stats too
        if "action" not in dataset_stats and "observation.state.q_pos" in dataset_stats:
             dataset_stats["action"] = dataset_stats["observation.state.q_pos"]

    input_steps = [
        # Helper to ensure action exists (aliasing q_pos to action if action missing)
        # We do this BEFORE rename because CopyStateToActionStep looks for "observation.state.q_pos"
        CopyStateToActionStep(source_key="observation.state.q_pos"),
        
        RenameObservationsProcessorStep(rename_map=rename_map),
        
        # After renaming (and copying), q_pos (now observation.state) is too long (horizon=16).
        # We need to slice it to n_obs_steps (2).
        SliceObservationStep(n_obs_steps=config.n_obs_steps),
        
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    if getattr(config, "use_delta", False):
        # We need the current state for delta calculation.
        # Since observation.state is a sequence (n_obs_steps=2), we extract the last step.
        input_steps.append(ExtractLastStateStep(source_key="observation.state", dest_key="current_state"))
        input_steps.append(AbsoluteToDeltaActionProcessorStep(state_key="current_state"))
        
    output_steps = []

    if getattr(config, "use_delta", False):
        # Reverse delta -> absolute (on normalized values)
        output_steps.append(DeltaToAbsoluteActionProcessorStep(state_key="current_state"))
        
    output_steps.extend([
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ])

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
