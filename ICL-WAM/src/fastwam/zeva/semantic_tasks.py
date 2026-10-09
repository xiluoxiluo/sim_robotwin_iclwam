"""Canonical RoboTwin task identity used by Zeva CTE/PIM.

LeRobot's ``task_index`` identifies a natural-language task string.  In the
RoboTwin 2.0 conversion this can be much finer-grained than the benchmark's
canonical task category, which makes it unsuitable as the Zeva same-task
identity.  This module keeps the two notions separate:

- raw task_index / instruction: FastWAM language condition
- semantic_task_id: canonical RoboTwin task, used by CTE task contrastive loss
  and same-task PIM retrieval
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROBOTWIN_CANONICAL_TASKS: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

_CANONICAL_SET = frozenset(ROBOTWIN_CANONICAL_TASKS)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def normalize_task_name(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


@dataclass(frozen=True)
class SemanticTaskIdentity:
    path: str
    sha256: str
    semantic_task_count: int
    episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "semantic_task_count": self.semantic_task_count,
            "episode_count": self.episode_count,
        }


class SemanticTaskMap:
    """Strict episode/raw-task-index -> canonical RoboTwin task resolver."""

    FORMAT = "robotwin_semantic_task_map_v1"

    def __init__(self, path: str | Path, payload: dict[str, Any]):
        self.path = Path(path).resolve()
        self.sha256 = sha256_file(self.path)

        fmt = str(payload.get("format", ""))
        if fmt != self.FORMAT:
            raise ValueError(
                f"semantic task map format mismatch: expected {self.FORMAT!r}, got {fmt!r}"
            )

        episode_mapping = payload.get("episode_to_semantic_task")
        task_index_mapping = payload.get("task_index_to_semantic_task", {})
        if not isinstance(episode_mapping, dict) or not episode_mapping:
            raise ValueError("semantic task map must contain non-empty episode_to_semantic_task")
        if not isinstance(task_index_mapping, dict):
            raise ValueError("task_index_to_semantic_task must be a mapping")

        self.episode_to_task = {
            int(k): normalize_task_name(v) for k, v in episode_mapping.items()
        }
        self.task_index_to_task = {
            str(k): normalize_task_name(v) for k, v in task_index_mapping.items()
        }

        all_tasks = set(self.episode_to_task.values()) | set(self.task_index_to_task.values())
        unknown = sorted(all_tasks - _CANONICAL_SET)
        if unknown:
            raise ValueError(
                "semantic task map contains non-canonical RoboTwin task names: "
                f"{unknown[:20]}"
            )

        self.semantic_tasks = tuple(sorted(set(self.episode_to_task.values())))
        if not self.semantic_tasks:
            raise ValueError("semantic task map resolved zero semantic tasks")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticTaskMap":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"semantic task map does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, payload)

    @property
    def identity(self) -> SemanticTaskIdentity:
        return SemanticTaskIdentity(
            path=str(self.path),
            sha256=self.sha256,
            semantic_task_count=len(self.semantic_tasks),
            episode_count=len(self.episode_to_task),
        )

    def resolve(
        self,
        *,
        episode_index: int | None,
        raw_task_index: int | str | None = None,
        strict: bool = True,
    ) -> str | None:
        if episode_index is not None:
            task = self.episode_to_task.get(int(episode_index))
            if task is not None:
                return task

        if raw_task_index is not None:
            task = self.task_index_to_task.get(str(raw_task_index))
            if task is not None:
                return task

        if strict:
            raise KeyError(
                "semantic task identity is missing for "
                f"episode_index={episode_index!r}, raw_task_index={raw_task_index!r}"
            )
        return None


def parse_episode_index_from_id(episode_id: str) -> int:
    marker = "::episode-"
    if marker not in str(episode_id):
        raise ValueError(f"cannot parse episode_index from episode_id={episode_id!r}")
    suffix = str(episode_id).rsplit(marker, 1)[1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"invalid episode_id={episode_id!r}") from exc
