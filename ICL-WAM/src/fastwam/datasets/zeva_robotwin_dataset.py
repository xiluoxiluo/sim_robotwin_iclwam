"""Episode-aware views over the existing FastWAM RoboTwin dataset."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from fastwam.zeva.schemas import build_transition_view
from fastwam.zeva.cache import PhaseEffectCache
from fastwam.zeva.retrieval import MemoryBank
from fastwam.zeva.task_context import TaskContextBank, retrieve_task_context


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    task_id: int | str
    task_name: str
    instruction: str
    episode_step: int
    episode_num_steps: int
    attempt_id: int = 0


class ZevaRobotWinDataset(Dataset):
    """Wrap ``RobotVideoDataset`` and expose causal validity/alignment fields.

    The underlying FastWAM dataset remains the single RGB/action pipeline.  A
    sample is an episode-ordered 32-action window; callers should use
    ``shuffle=False`` for the Stage 1 sequential sampler.
    """

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        source_index = int(sample.get("dataset_index", index))
        if source_index != int(index):
            # The base LeRobot loader may retry a corrupt/padded request at a
            # random index. That is acceptable for ordinary FastWAM training,
            # but it would silently misjoin episode state and cache rows for
            # Zeva; fail loudly so the bad sample can be repaired/rebuilt.
            raise RuntimeError(
                f"Zeva requires deterministic source indexing: requested {index}, "
                f"underlying dataset returned {source_index}"
            )
        video = sample["video"]
        action = sample["action"]
        if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] != 9:
            raise ValueError(f"Zeva V1 expects sample video [3,9,H,W], got {tuple(video.shape)}")
        if action.ndim != 2 or action.shape != (32, 14):
            raise ValueError(f"Zeva V1 expects sample action [32,14], got {tuple(action.shape)}")
        frames = video.permute(1, 0, 2, 3).unsqueeze(0)
        actions = action.unsqueeze(0)
        frame_pad = sample.get("image_is_pad", torch.zeros(frames.shape[1], dtype=torch.bool))
        action_pad = sample.get("action_is_pad", torch.zeros(action.shape[0], dtype=torch.bool))
        aligned = build_transition_view(
            actions,
            frames,
            frame_valid=~frame_pad.unsqueeze(0),
            action_valid=~action_pad.unsqueeze(0),
        )
        sample.update({key: value[0] for key, value in aligned.items()})
        sample["frame_valid"] = (~frame_pad).bool()
        sample["action_valid"] = (~action_pad).bool()
        instruction = str(sample.get("prompt", sample.get("instruction", "")))
        sample["episode"] = EpisodeMetadata(
            episode_id=str(sample.get("episode_id", f"window-{index}")),
            task_id=sample.get("task_id", 0),
            task_name=str(sample.get("task_name", instruction)),
            instruction=instruction,
            episode_step=int(sample.get("episode_step", 0)),
            episode_num_steps=int(sample.get("episode_num_steps", 8)),
            attempt_id=int(sample.get("attempt_id", 0)),
        )
        return sample


class ZevaStage2Dataset(Dataset):
    """Attach causal prompt inputs to each frozen FastWAM training window."""

    def __init__(
        self,
        base_dataset: Dataset,
        cache: PhaseEffectCache,
        top_k: int = 4,
        bit_size: int = 4,
        pim_retrieval_mode: str = "phase",
        task_context_bank: TaskContextBank | None = None,
        task_context_top_k: int = 1,
        task_context_by_episode: dict[str, torch.Tensor] | None = None,
    ):
        if cache.manifest.schema_version not in {"zeva_fastwam_robotwin_cache_v3", "zeva_fastwam_robotwin_cache_v4"}:
            raise ValueError(
                "Stage 2 requires zeva_fastwam_robotwin_cache_v3/v4; older "
                "transition-level caches are incompatible with effect-window cadence"
            )
        self.base = ZevaRobotWinDataset(base_dataset)
        self.cache = cache
        self.top_k, self.bit_size = int(top_k), int(bit_size)
        self.pim_retrieval_mode = str(pim_retrieval_mode)
        if self.pim_retrieval_mode not in {"phase", "cross_task_effect"}:
            raise ValueError(
                "pim_retrieval_mode must be 'phase' or 'cross_task_effect'"
            )
        self.task_context_bank = task_context_bank
        self.task_context_top_k = int(task_context_top_k)
        self.task_context_by_episode = task_context_by_episode
        if task_context_bank is not None and task_context_by_episode is not None:
            raise ValueError("Select either text-bank or static episode task contexts")
        if self.task_context_top_k < 1:
            raise ValueError("task_context_top_k must be positive")
        if task_context_bank is not None and self.task_context_top_k > len(task_context_bank):
            raise ValueError("task_context_top_k must not exceed the task-context bank size")
        if cache.manifest.schema_version == "zeva_fastwam_robotwin_cache_v4":
            self._cached_window_indices = tuple(sorted({
                int(row["window_index"])
                for row_index, row in enumerate(cache.rows)
                if row.get("record_type") == "phase_query"
                and row.get("window_index") is not None
                and cache.get(row_index)["valid"]
            }))
        else:
            self._cached_window_indices = tuple(sorted({
                int(row["window_index"])
                for row_index, row in enumerate(cache.rows)
                if row.get("window_index") is not None
                and int(row.get("effect_index", 0 if int(row.get("transition_index", -1)) == 0 else -1)) == 0
                and cache.get(row_index)["valid"]
            }))
        if not self._cached_window_indices:
            raise ValueError("Stage 2 cache contains no complete window starts (transition_index=0)")
        if task_context_by_episode is not None:
            query_windows = set(self._cached_window_indices)
            required_episodes = {str(row["episode_id"]) for row in cache.rows if row.get("window_index") in query_windows}
            missing = required_episodes - task_context_by_episode.keys()
            if missing:
                raise ValueError(f"missing initial task contexts for cached episodes: {sorted(missing)[:5]}")
        self._window_rows: dict[int, int] = {}
        self._bit_history: dict[int, tuple[torch.Tensor, ...]] = {}
        episode_effects: dict[str, deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=self.bit_size))
        # Build the causal BIT prefix once. v4 cache rows are ordered with a
        # completed effect before a phase query at the same raw boundary, so
        # an effect ending at raw step s is visible to the query at s. Legacy
        # v3 rows retain their original window-local ordering.
        for row_index in range(len(cache.rows)):
            item = cache.get(row_index)
            window_index = item.get("window_index")
            episode_id = str(item["episode_id"])
            is_query = (
                item.get("record_type") == "phase_query"
                if cache.manifest.schema_version == "zeva_fastwam_robotwin_cache_v4"
                else int(item.get("effect_index", 0 if int(item["transition_index"]) == 0 else -1)) == 0
            )
            if window_index is not None and is_query and item["valid"]:
                window_index = int(window_index)
                if window_index in self._window_rows:
                    raise ValueError(f"duplicate cache window_index {window_index}")
                self._window_rows[window_index] = row_index
                self._bit_history[window_index] = tuple(episode_effects[episode_id])
            if item["valid"] and (
                cache.manifest.schema_version != "zeva_fastwam_robotwin_cache_v4"
                or item.get("record_type") == "effect"
            ):
                episode_effects[episode_id].append(item["effect"])
        self.bank = MemoryBank(
            phase_dim=int(cache.manifest.phase_dim),
            effect_dim=int(cache.manifest.effect_dim),
            top_k=self.top_k,
        )
        for row_index, row in enumerate(cache.rows):
            item = cache.get(row_index)
            if cache.manifest.schema_version == "zeva_fastwam_robotwin_cache_v4" and item.get("record_type") != "effect":
                continue
            self.bank.add(
                # PIM stores the phase at the beginning of the effect window;
                # use the same phase_pre representation for offline retrieval.
                item["phase_pre"],
                item["effect"],
                episode_id=item["episode_id"],
                task_id=item["task_id"],
                attempt_id=item["attempt_id"],
                transition_index=item["transition_index"],
                effect_index=item.get("effect_index"),
                episode_step=item.get("episode_step"),
                window_index=item.get("window_index"),
                valid=item["valid"],
            )

    def __len__(self) -> int:
        # Cache construction may retain only complete/valid windows. Expose
        # that sparse view so Stage 2 never asks for an uncached sample.
        return len(self._cached_window_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cached_window_indices:
            if index < 0 or index >= len(self._cached_window_indices):
                raise IndexError(index)
            requested_index = self._cached_window_indices[index]
        else:
            requested_index = int(index)
        sample = self.base[requested_index]
        # The cache is episode ordered and carries the originating window id;
        # use it when available so filtering or sharding cannot silently shift
        # the causal memory alignment.
        source_index = int(sample.get("dataset_index", requested_index))
        start = self._window_rows.get(source_index)
        if start is None:
            raise IndexError(f"cache has no valid records for dataset index {source_index}")
        current = self.cache.get(start)
        expected_window = current.get("window_index")
        if expected_window != source_index:
            raise ValueError(f"cache window_index mismatch: expected {source_index}, got {expected_window}")
        if self.cache.manifest.schema_version == "zeva_fastwam_robotwin_cache_v4":
            if current.get("record_type") != "phase_query":
                raise ValueError("Stage 2 window mapping must point to a v4 phase_query record")
            raw_step = current.get("raw_step")
            if raw_step is None or int(raw_step) != int(current.get("episode_step")):
                raise ValueError("v4 phase_query raw_step and episode_step must identify the same query boundary")
            if int(sample["episode"].episode_step) != int(raw_step):
                raise ValueError(
                    "Stage 2 sample/cache raw-step mismatch: "
                    f"sample={sample['episode'].episode_step}, cache={raw_step}"
                )
        episode_id = current["episode_id"]
        phase = current["phase_pre"]
        effect_dim = int(self.cache.manifest.effect_dim)
        bit_effects = torch.zeros((self.bit_size, effect_dim), dtype=torch.float32)
        bit_mask = torch.zeros(self.bit_size, dtype=torch.bool)
        previous = list(self._bit_history.get(source_index, ()))
        if previous:
            bit_effects[-len(previous):] = torch.stack(previous)
            bit_mask[-len(previous):] = True
        if self.pim_retrieval_mode == "phase":
            retrieved = self.bank.retrieve(
                phase,
                episode_id=episode_id,
                task_id=current["task_id"],
                top_k=self.top_k,
            )
        elif bool(bit_mask.any()):
            # Effect transfer is queried only from an already completed
            # effect in the causal prefix.  Querying the current cache row's
            # effect would leak the outcome of the action being predicted.
            retrieved = self.bank.retrieve_by_effect(
                bit_effects[bit_mask][-1],
                episode_id=episode_id,
                task_id=current["task_id"],
                top_k=self.top_k,
                cross_task=True,
            )
        else:
            # Keep the model contract [K,D] even before the first completed
            # effect is available; no cross-task evidence exists yet.
            retrieved = self.bank.retrieve(
                phase,
                episode_id=episode_id,
                task_id=current["task_id"],
                top_k=self.top_k,
            )
            retrieved = type(retrieved)(
                torch.zeros_like(retrieved.phases),
                torch.zeros_like(retrieved.effects),
                torch.zeros_like(retrieved.mask),
                torch.full_like(retrieved.scores, -torch.inf),
                (),
            )
        current_raw_step = current.get("raw_step")
        if current_raw_step is None:
            current_raw_step = current.get("episode_step", 0)
        sample.update({"phase": phase, "bit_effects": bit_effects, "bit_mask": bit_mask,
                       "raw_step": int(current_raw_step),
                       "pim_phases": retrieved.phases, "pim_effects": retrieved.effects, "pim_mask": retrieved.mask,
                       "behavior_memory": None, "behavior_memory_mask": None})
        if self.task_context_by_episode is not None:
            episode_id = str(sample["episode"].episode_id)
            if episode_id != str(current["episode_id"]):
                raise ValueError("sample/cache episode identity mismatch for static task context")
            sample["task_context"] = self.task_context_by_episode[episode_id].clone()
        elif self.task_context_bank is not None:
            task_context, task_context_result = retrieve_task_context(
                sample["context"], sample.get("context_mask"), self.task_context_bank,
                output_dim=self.task_context_bank.value_dim,
                top_k=self.task_context_top_k,
            )
            # Keep one vector per dataset item; the default DataLoader collate
            # then produces the model contract [B,global_dim].
            sample["task_context"] = task_context[0]
            sample["task_context_retrieval_indices"] = task_context_result.indices[0]
            sample["task_context_retrieval_scores"] = task_context_result.scores[0]
        # The trainer builds memory tokens through CausalPromptEncoder.
        sample.pop("behavior_memory"); sample.pop("behavior_memory_mask")
        sample.pop("episode", None)
        return sample
