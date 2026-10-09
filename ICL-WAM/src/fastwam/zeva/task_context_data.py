"""Offline demonstration bank and initial-readout pairs for static retrieval."""

from __future__ import annotations

from collections import defaultdict

import torch

from .static_task_context import (
    BEHAVIOR_KEY_SPACE, BEHAVIOR_VALUE_SPACE, READOUT_KIND, trajectory_prototype,
)
from .task_context import TaskContextBank


def iter_demo_trajectories(dataset):
    """Join valid four-action boundaries, one full demonstration at a time.

    The source is ZevaRobotWinDataset. Overlapping windows are deduplicated;
    padding never contributes to a trajectory key or behavior prototype.
    """
    grouped = defaultdict(list)
    for index in range(len(dataset)):
        sample = dataset[index]
        if "episode_id" not in sample or "episode_step" not in sample:
            raise ValueError("behavior bank requires real episode IDs and raw-step coordinates")
        episode = sample["episode"]
        if episode.episode_step % 4 == 0:
            grouped[str(episode.episode_id)].append((episode.episode_step, index))
    for episode_id in sorted(grouped):
        rows = sorted(grouped[episode_id])
        if not rows or rows[0][0] != 0:
            raise ValueError(f"episode {episode_id} has no initial observation at raw step zero")
        frames, actions = {}, {}
        initial = None
        task_id = instruction = None

        def add_unique(mapping, step, value, kind):
            previous = mapping.get(step)
            if previous is not None and not torch.allclose(previous, value, rtol=0.0, atol=1e-6):
                raise ValueError(f"inconsistent {kind} in episode {episode_id} at raw step {step}")
            if previous is None:
                # A retained frame view would otherwise keep its entire source window alive.
                mapping[step] = value.detach().cpu().clone()

        for step, index in rows:
            sample = dataset[index]
            episode = sample["episode"]
            if initial is None:
                initial = sample
                task_id, instruction = str(episode.task_id), episode.instruction
            if str(episode.task_id) != task_id or episode.instruction != instruction:
                raise ValueError(f"task/instruction changed within demonstration {episode_id}")
            for offset, frame in enumerate(sample["video"].permute(1, 0, 2, 3)):
                if bool(sample["frame_valid"][offset]):
                    add_unique(frames, step + 4 * offset, frame, "frame")
            for offset, action in enumerate(sample["transition_actions"]):
                if bool(sample["transition_valid"][offset].all()):
                    add_unique(actions, step + 4 * offset, action, "action")
        boundaries = [0]
        while boundaries[-1] in actions and boundaries[-1] + 4 in frames:
            boundaries.append(boundaries[-1] + 4)
        if len(boundaries) < 2 or 0 not in frames:
            raise ValueError(f"episode {episode_id} has no complete initial transition")
        if any(step > boundaries[-1] for step in frames):
            raise ValueError(f"episode {episode_id} contains a gap in its demonstrated prefix")
        yield {
            "episode_id": episode_id, "task_id": task_id, "instruction": instruction,
            "frames": torch.stack([frames[step] for step in boundaries]),
            "actions": torch.stack([actions[step] for step in boundaries[:-1]]),
            "context": initial["context"], "context_mask": initial["context_mask"],
            "num_transitions": len(boundaries) - 1,
        }


@torch.no_grad()
def build_behavior_bank(dataset, cte, policy, *, frame_encoder=None, temperature=0.07, metadata=None):
    """Pair a demonstrated CTE trajectory with its clean initial policy readout."""
    cte.eval().requires_grad_(False)
    policy.eval().requires_grad_(False)
    device = next(cte.parameters()).device
    entries, readouts = [], []
    for demo in iter_demo_trajectories(dataset):
        frames = demo["frames"].unsqueeze(0)
        initial_readout = policy.extract_task_context_readout(
            frames[:, 0], demo["context"].unsqueeze(0), demo["context_mask"].unsqueeze(0),
        )
        if frame_encoder is not None:
            frames = frame_encoder(frames)
        valid = torch.ones(frames.shape[:2], dtype=torch.bool, device=device)
        output = cte(frames.to(device), demo["actions"].unsqueeze(0).to(device), valid_mask=valid)
        key, value = trajectory_prototype(output, valid)
        entries.append({
            "retrieval_key": key[0].cpu(), "behavior_value": value[0].cpu(),
            "episode_id": demo["episode_id"], "task_id": demo["task_id"],
            "instruction": demo["instruction"], "num_transitions": demo["num_transitions"],
        })
        readouts.append(initial_readout[0].float().cpu())
    bank = TaskContextBank(
        entries, key_dim=cte.cfg.retrieval_dim, value_dim=cte.cfg.hidden_dim,
        temperature=temperature,
        metadata={
            **dict(metadata or {}), "key_space": BEHAVIOR_KEY_SPACE,
            "value_space": BEHAVIOR_VALUE_SPACE, "readout_kind": READOUT_KIND, "split": "train",
            "readout_dim": int(readouts[0].numel()) if readouts else 0,
        },
    )
    return bank, torch.stack(readouts)
