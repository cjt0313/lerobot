#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import ClassVar

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.core import PolicyAction, TransitionKey
from lerobot.processor.pipeline import ActionProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("absolute_to_delta")
@dataclass
class AbsoluteToDeltaActionProcessorStep(ActionProcessorStep):
    """
    Converts absolute actions to delta actions relative to the current state.
    
    Processing logic:
        delta_action = action - current_state

    Intended for use in the pre-processing pipeline (before normalization).
    
    Attributes:
        state_key: The key in the observation dictionary corresponding to the state 
                   (e.g., "observation.state") to subtract from the action.
    """
    state_key: str = "observation.state"

    def action(self, action: PolicyAction) -> PolicyAction:
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"AbsoluteToDeltaActionProcessorStep expects PolicyAction (Tensor), got {type(action)}")

        # Retrieve the observation from the current transition
        if self._current_transition is None:
             raise RuntimeError("AbsoluteToDeltaActionProcessorStep requires a transition context.")
             
        observation = self._current_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
             raise ValueError("Observation is missing in the transition.")
             
        current_state = observation.get(self.state_key)
        if current_state is None:
            raise KeyError(f"State key '{self.state_key}' not found in observation. Available keys: {list(observation.keys())}")

        # Ensure state is a tensor
        if not isinstance(current_state, torch.Tensor):
             # Try to convert if it's a list/array (though usually it's a tensor in pipeline)
             current_state = torch.tensor(current_state, device=action.device, dtype=action.dtype)

        # Handle broadcasting for chunked actions
        # Case 1: action is (batch, dim), state is (batch, dim) -> Direct subtraction
        # Case 2: action is (batch, horizon, dim), state is (batch, dim) -> Unsqueeze state
        # Case 3: action is (dim,), state is (dim,) -> Direct subtraction
        
        target_shape = action.shape
        state_shape = current_state.shape
        
        # Simple broadcasting check: if action has one more dimension than state (time dim), unsqueeze state
        if action.ndim == current_state.ndim + 1:
            current_state = current_state.unsqueeze(-2) # Add time dimension before the last dimension (features)

        return action - current_state

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # This step does not change the structure of features, just the values
        return features


@ProcessorStepRegistry.register("delta_to_absolute")
@dataclass
class DeltaToAbsoluteActionProcessorStep(ActionProcessorStep):
    """
    Converts delta actions back to absolute actions by adding the current state.
    
    Processing logic:
        action = delta_action + current_state

    Intended for use in the post-processing pipeline (after unnormalization).
    
    Attributes:
        state_key: The key in the observation dictionary corresponding to the state 
                   (e.g., "observation.state") to add to the action.
    """
    state_key: str = "observation.state"

    def action(self, action: PolicyAction) -> PolicyAction:
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"DeltaToAbsoluteActionProcessorStep expects PolicyAction (Tensor), got {type(action)}")

        if self._current_transition is None:
             raise RuntimeError("DeltaToAbsoluteActionProcessorStep requires a transition context.")

        observation = self._current_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
             raise ValueError("Observation is missing in the transition.")
             
        current_state = observation.get(self.state_key)
        if current_state is None:
            raise KeyError(f"State key '{self.state_key}' not found in observation. Available keys: {list(observation.keys())}")

        if not isinstance(current_state, torch.Tensor):
             current_state = torch.tensor(current_state, device=action.device, dtype=action.dtype)

        # Handle broadcasting (same as above)
        if action.ndim == current_state.ndim + 1:
            current_state = current_state.unsqueeze(-2)

        return action + current_state

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # This step does not change the structure of features, just the values
        return features
