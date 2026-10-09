"""Zeva causal-memory addon for FastWAM RoboTwin."""

from .behavior_prefix_adapter import (
    BehaviorPrefixAdapter, BehaviorPrefixAdapterConfig,
    ExactZevaPolicyInjectionAdapter, ExactZevaPolicyInjectionConfig,
    FastWAMPolicyInjectionPrior, gaussian_prior_nll,
)
from .causal_prompt import CausalPromptConfig, CausalPromptEncoder, task_tokens_from_context
from .causal_transition_encoder import CausalTransitionEncoder, CausalTransitionEncoderConfig
from .cte_losses import CTELossConfig, causal_transition_encoder_loss
from .lifecycle import CausalCTEHistory, CausalMemoryLifecycle, LifecycleConfig
from .stage1_sampling import CTETrainIndex, TaskBalancedCTEBatchSampler, build_cte_query_index, build_cte_training_index
from .memory import BriefInteractionTrace, PersistentInteractionMemory, PersistentInteractionMemoryConfig
from .retrieval import MemoryBank, RetrievalResult
from .task_context import TaskContextBank, TaskContextRetrievalResult, retrieve_task_context
from .static_task_context import (
    StaticTaskContextRetrievalConfig, StaticTaskContextRetrievalHead,
    StaticTaskContextRetriever, bidirectional_supervised_contrastive_loss,
)
from .schemas import CacheManifest, TransitionRecord, build_transition_view, transition_valid_mask
from .cache import PhaseEffectCache, save_phase_effect_cache
from .vae_adapter import FastWAMCTELatentEncoder, load_frozen_wan_vae, make_frame_encoder, validate_vae_metadata

__all__ = [
    "BehaviorPrefixAdapter",
    "BehaviorPrefixAdapterConfig",
    "ExactZevaPolicyInjectionAdapter",
    "ExactZevaPolicyInjectionConfig",
    "FastWAMPolicyInjectionPrior",
    "gaussian_prior_nll",
    "CausalPromptConfig",
    "CausalPromptEncoder",
    "task_tokens_from_context",
    "CausalTransitionEncoder",
    "CausalTransitionEncoderConfig",
    "CTELossConfig",
    "CausalMemoryLifecycle",
    "CausalCTEHistory",
    "LifecycleConfig",
    "CTETrainIndex",
    "TaskBalancedCTEBatchSampler",
    "build_cte_training_index",
    "build_cte_query_index",
    "BriefInteractionTrace",
    "PersistentInteractionMemory",
    "PersistentInteractionMemoryConfig",
    "MemoryBank",
    "RetrievalResult",
    "TaskContextBank",
    "TaskContextRetrievalResult",
    "retrieve_task_context",
    "StaticTaskContextRetrievalConfig",
    "StaticTaskContextRetrievalHead",
    "StaticTaskContextRetriever",
    "bidirectional_supervised_contrastive_loss",
    "CacheManifest",
    "TransitionRecord",
    "build_transition_view",
    "transition_valid_mask",
    "causal_transition_encoder_loss",
    "PhaseEffectCache",
    "save_phase_effect_cache",
    "FastWAMCTELatentEncoder",
    "make_frame_encoder",
    "load_frozen_wan_vae",
    "validate_vae_metadata",
]
