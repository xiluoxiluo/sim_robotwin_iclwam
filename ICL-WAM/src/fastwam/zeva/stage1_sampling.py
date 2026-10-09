"""Deterministic, non-overlapping sampling helpers for Zeva Stage 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
import random
from tqdm import tqdm
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class CTETrainIndex:
    """Identity of one valid CTE source window in the base dataset."""

    dataset_index: int
    episode_id: str
    task_id: str | int
    episode_step: int


def _episode_value(episode: Any, name: str, default: Any = None) -> Any:
    if isinstance(episode, dict):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _task_key(task_id: str | int) -> str:
    return str(task_id)


def _is_valid_window(sample: dict[str, Any]) -> bool:
    transition_valid = sample.get("transition_valid")
    if transition_valid is None:
        return False
    # A source row with padding cannot establish the next causal window. It
    # is still useful to stop the cursor, but must not be selected as a
    # training row because its temporal target is incomplete.
    frame_valid = sample.get("frame_valid")
    return bool(transition_valid.any()) and (
        frame_valid is None or bool(frame_valid.all())
    ) and bool(transition_valid.all())


def build_cte_training_index(
    dataset: Sequence[dict[str, Any]],
    source_window_actions: int = 32,
    sample_stride: int = 1,
) -> list[CTETrainIndex]:
    """Build non-overlapping CTE windows from LeRobot episode metadata.

    Fast path for ZevaRobotWinDataset:
      - does NOT call dataset[index]
      - does NOT decode RGB/video
      - uses LeRobot episode boundaries directly
      - only reads task_index metadata

    The original slow path is kept as a fallback for other datasets.
    """

    source_window_actions = int(source_window_actions)
    sample_stride = int(sample_stride)

    if source_window_actions < 1 or sample_stride < 1:
        raise ValueError(
            "source_window_actions and sample_stride must be positive"
        )

    # ------------------------------------------------------------------
    # Fast metadata-only path for:
    #
    # ZevaRobotWinDataset
    #   -> RobotVideoDataset
    #      -> BaseLerobotDataset
    #         -> MultiLeRobotDataset
    #
    # RoboTwin Zeva V1 currently requires sample_stride == 1.
    # ------------------------------------------------------------------
    zeva_base = getattr(dataset, "base_dataset", None)

    semantic_resolver = (
        getattr(zeva_base, "resolve_semantic_task_id", None)
        if zeva_base is not None
        else None
    )

    lerobot_base = (
        getattr(zeva_base, "lerobot_dataset", None)
        if zeva_base is not None
        else None
    )

    multi_dataset = (
        getattr(lerobot_base, "multi_dataset", None)
        if lerobot_base is not None
        else None
    )

    inner_datasets = (
        getattr(multi_dataset, "_datasets", None)
        if multi_dataset is not None
        else None
    )

    if inner_datasets is not None and sample_stride == 1:
        if not callable(semantic_resolver):
            raise RuntimeError(
                "RoboTwin Zeva Stage-1 metadata fast path requires "
                "RobotVideoDataset.resolve_semantic_task_id(). "
                "Do not use raw LeRobot task_index as the CTE task identity."
            )

        rows: list[CTETrainIndex] = []

        global_dataset_offset = 0

        for inner_dataset in inner_datasets:
            episode_data_index = inner_dataset.episode_data_index

            episode_from = episode_data_index["from"]
            episode_to = episode_data_index["to"]

            # Actual episode ids retained after train/val splitting.
            if inner_dataset.episodes is None:
                episode_ids = list(range(inner_dataset.meta.total_episodes))
            else:
                episode_ids = list(inner_dataset.episodes)

            if len(episode_ids) != len(episode_from):
                raise RuntimeError(
                    "LeRobot episode metadata mismatch: "
                    f"{len(episode_ids)=}, {len(episode_from)=}"
                )

            # ----------------------------------------------------------
            # First determine all non-overlapping COMPLETE windows.
            #
            # A CTE source window contains:
            #   32 actions
            #   33 raw observations
            #
            # therefore:
            #   episode_length >= start + 33
            #
            # valid starts:
            #   0, 32, 64, ...
            # ----------------------------------------------------------
            selected: list[tuple[int, int, int, int]] = []

            # tuple:
            # (
            #   global_dataset_index,
            #   episode_id,
            #   episode_step,
            #   inner_dataset_index,
            # )

            for episode_pos, episode_id in tqdm(
                enumerate(episode_ids),
                total=len(episode_ids),
                desc="Building CTE metadata index",
                dynamic_ncols=True,
            ):
                ep_start = int(episode_from[episode_pos])
                ep_end = int(episode_to[episode_pos])
                episode_length = ep_end - ep_start

                # Need 33 frames for 32 actions.
                max_start = episode_length - (source_window_actions + 1)

                if max_start < 0:
                    continue

                for episode_step in range(
                    0,
                    max_start + 1,
                    source_window_actions,
                ):
                    inner_index = ep_start + episode_step

                    global_index = (
                        global_dataset_offset + inner_index
                    )

                    selected.append(
                        (
                            global_index,
                            int(episode_id),
                            int(episode_step),
                            int(inner_index),
                        )
                    )

            # ----------------------------------------------------------
            # task_index is parquet/HF metadata, not video.
            #
            # Read only task_index for the selected rows.
            # This is dramatically cheaper than calling dataset[index].
            # ----------------------------------------------------------
            if selected:
                selected_inner_indices = [
                    item[3] for item in selected
                ]

                raw_dataset = inner_dataset.hf_dataset.with_format(None)

                if "task_index" in raw_dataset.column_names:
                    task_values = raw_dataset[
                        selected_inner_indices
                    ]["task_index"]
                else:
                    task_values = [0] * len(selected)

                if len(task_values) != len(selected):
                    raise RuntimeError(
                        "task_index metadata count does not match "
                        "selected CTE windows"
                    )

                dataset_root = str(inner_dataset.root)

                for (
                    global_index,
                    episode_id,
                    episode_step,
                    _inner_index,
                ), task_value in zip(
                    selected,
                    task_values,
                    strict=True,
                ):
                    task_id = semantic_resolver(
                        episode_index=int(episode_id),
                        raw_task_index=task_value,
                        strict=True,
                    )

                    rows.append(
                        CTETrainIndex(
                            dataset_index=global_index,
                            episode_id=(
                                f"{dataset_root}"
                                f"::episode-{episode_id}"
                            ),
                            task_id=task_id,
                            episode_step=episode_step,
                        )
                    )

            global_dataset_offset += int(inner_dataset.num_frames)

        print(
            "[CTE index] metadata-only fast path: "
            f"selected {len(rows)} non-overlapping windows "
            f"from {len(dataset)} source rows"
        )

        return rows

    # ------------------------------------------------------------------
    # Fallback:
    # Preserve old behavior for an unsupported dataset.
    #
    # WARNING: this path decodes samples and can be very slow.
    # ------------------------------------------------------------------
    next_valid_step: dict[str, int] = {}
    rows: list[CTETrainIndex] = []

    for dataset_index, sample in enumerate(dataset):
        episode = sample.get("episode")

        if episode is None:
            raise ValueError(
                "CTE samples must contain episode metadata"
            )

        episode_id = str(
            _episode_value(
                episode,
                "episode_id",
                f"episode-{dataset_index}",
            )
        )

        episode_step = int(
            _episode_value(
                episode,
                "episode_step",
                0,
            )
        )

        expected = next_valid_step.get(episode_id)

        if expected is not None:
            if episode_step < expected:
                continue

            if episode_step != expected:
                next_valid_step.pop(episode_id, None)

        if not _is_valid_window(sample):
            next_valid_step.pop(episode_id, None)
            continue

        task_id = _episode_value(
            episode,
            "task_id",
            0,
        )

        if task_id in (None, 0, "0"):
            task_id = _episode_value(
                episode,
                "task_name",
                task_id,
            )

        if not isinstance(task_id, (str, int)):
            task_id = str(task_id)

        rows.append(
            CTETrainIndex(
                dataset_index=int(dataset_index),
                episode_id=episode_id,
                task_id=task_id,
                episode_step=episode_step,
            )
        )

        next_valid_step[episode_id] = (
            episode_step
            + source_window_actions * sample_stride
        )

    return rows


def build_cte_query_index(
    dataset: Sequence[dict[str, Any]],
    sample_stride: int = 1,
    transition_steps: int = 4,
) -> list[CTETrainIndex]:
    """Index every complete source window eligible for a phase query.

    Unlike Stage 1 optimization, cache construction must expose deployment
    query positions such as raw steps 24/48. It therefore retains overlapping
    windows; the cache builder later deduplicates their shared boundaries and
    runs one full-episode CTE pass.
    """

    sample_stride = int(sample_stride)
    transition_steps = int(transition_steps)
    if sample_stride < 1 or transition_steps < 1:
        raise ValueError("sample_stride and transition_steps must be positive")
    rows: list[CTETrainIndex] = []
    for dataset_index, sample in enumerate(dataset):
        episode = sample.get("episode")
        if episode is None:
            raise ValueError("CTE samples must contain episode metadata")
        if not _is_valid_window(sample):
            continue
        episode_id = str(_episode_value(episode, "episode_id", f"episode-{dataset_index}"))
        episode_step = int(_episode_value(episode, "episode_step", 0))
        # CTE boundaries occur after complete transition groups. A dataset
        # window can start at every raw action step, but starts between these
        # boundaries cannot be represented by the grouped action stream.
        if episode_step % (transition_steps * sample_stride) != 0:
            continue
        task_id = _episode_value(episode, "task_id", 0)
        if task_id in (None, 0, "0"):
            task_id = _episode_value(episode, "task_name", task_id)
        if not isinstance(task_id, (str, int)):
            task_id = str(task_id)
        rows.append(
            CTETrainIndex(
                dataset_index=int(dataset_index),
                episode_id=episode_id,
                task_id=task_id,
                episode_step=episode_step,
            )
        )
    return rows


def _take_prefer_distinct_episode(
    pool: list[CTETrainIndex], count: int, rng: random.Random
) -> list[CTETrainIndex]:
    """Take rows while preferring distinct episodes for positive pairs."""

    if count > len(pool):
        raise ValueError("cannot take more rows than are available in a task pool")
    shuffled = list(pool)
    rng.shuffle(shuffled)
    selected: list[CTETrainIndex] = []
    used_episodes: set[str] = set()
    for row in shuffled:
        if row.episode_id not in used_episodes:
            selected.append(row)
            used_episodes.add(row.episode_id)
            if len(selected) == count:
                return selected
    selected.extend(row for row in shuffled if row not in selected)
    return selected[:count]


class TaskBalancedCTEBatchSampler:
    """Yield deterministic positive-aware task-balanced CTE batches.

    This optimized implementation preserves the existing sampling policy while
    avoiding repeated whole-pool copies/scans in the hot path.

    Main optimizations
    ------------------
    1. Build immutable task/episode indices once in ``__init__``.
    2. At each epoch, shuffle each row exactly once and construct a task-local
       episode-interleaved queue. Taking 2/4 rows is then O(count), rather than
       copying + shuffling + filtering the whole task pool every time.
    3. Track ``remaining_total`` and ``pair_units_total`` incrementally instead
       of repeatedly summing every task pool.
    4. ``__len__`` is O(1): every complete batch consumes ``batch_size`` unique
       rows and the final remainder is padded, therefore the exact number of
       yielded batches is ``ceil(num_rows / batch_size)``.
    5. Add a rank-0 tqdm progress bar for epoch batch-plan construction.

    Existing call sites remain compatible. ``show_progress`` is optional and
    defaults to True.
    """

    def __init__(
        self,
        rows: Iterable[CTETrainIndex],
        batch_size: int,
        *,
        tasks_per_batch: int | None = None,
        samples_per_task: int = 2,
        seed: int = 0,
        show_progress: bool = True,
    ) -> None:
        self.rows = tuple(rows)
        self.batch_size = int(batch_size)
        self.samples_per_task = int(samples_per_task)

        if self.batch_size < 1 or self.samples_per_task < 1:
            raise ValueError(
                "batch_size and samples_per_task must be positive"
            )

        if tasks_per_batch is None:
            tasks_per_batch = max(
                1,
                self.batch_size // self.samples_per_task,
            )

        self.tasks_per_batch = int(tasks_per_batch)

        if self.tasks_per_batch < 1:
            raise ValueError("tasks_per_batch must be positive")

        if (
            self.tasks_per_batch
            * self.samples_per_task
            > self.batch_size
        ):
            raise ValueError(
                "tasks_per_batch * samples_per_task exceeds batch_size"
            )

        self.seed = int(seed)
        self.epoch = 0
        self.show_progress = bool(show_progress)

        # --------------------------------------------------------------
        # Static index: build ONCE, not once per epoch.
        # --------------------------------------------------------------
        rows_by_task: dict[str, list[CTETrainIndex]] = defaultdict(list)
        for row in self.rows:
            rows_by_task[_task_key(row.task_id)].append(row)

        self._rows_by_task: dict[str, tuple[CTETrainIndex, ...]] = {
            key: tuple(task_rows)
            for key, task_rows in rows_by_task.items()
        }
        self._task_keys = tuple(sorted(self._rows_by_task))

        # Group every task by episode once.  Epoch-specific sampling only
        # needs to copy/shuffle these small episode buckets.
        episode_rows_by_task: dict[
            str,
            tuple[tuple[CTETrainIndex, ...], ...],
        ] = {}

        for key in self._task_keys:
            by_episode: dict[str, list[CTETrainIndex]] = defaultdict(list)

            for row in self._rows_by_task[key]:
                by_episode[row.episode_id].append(row)

            episode_rows_by_task[key] = tuple(
                tuple(by_episode[episode_id])
                for episode_id in sorted(by_episode)
            )

        self._episode_rows_by_task = episode_rows_by_task

        # Exact because all source rows are consumed once and only the final
        # remainder is padded with wrap-around rows.
        self._num_batches = (
            (len(self.rows) + self.batch_size - 1) // self.batch_size
            if self.rows
            else 0
        )

        # Diagnostics. Existing trainer code does not depend on them.
        self.balanced_batches = 0
        self.positive_batches = 0
        self.singleton_only_batches = 0
        self.wraparound_rows = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _progress_enabled(self) -> bool:
        """Show one progress bar only on rank 0 in distributed training."""

        if not self.show_progress:
            return False

        rank = os.environ.get(
            "RANK",
            os.environ.get("LOCAL_RANK", "0"),
        )
        return str(rank) in {"0", ""}

    def _build_epoch_task_order(
        self,
        key: str,
        rng: random.Random,
    ) -> list[CTETrainIndex]:
        """Create one O(N) episode-interleaved row order for a task.

        Rows inside every episode are shuffled once. Episode buckets are then
        consumed round-robin, so consecutive rows prefer distinct episodes.
        This replaces the old repeated ``copy -> shuffle -> filter`` path.
        """

        buckets = [
            list(bucket)
            for bucket in self._episode_rows_by_task[key]
        ]

        for bucket in buckets:
            rng.shuffle(bucket)

        rng.shuffle(buckets)

        ordered: list[CTETrainIndex] = []
        active = buckets

        # Every loop body pops one source row, so total complexity is O(N).
        while active:
            next_active: list[list[CTETrainIndex]] = []

            for bucket in active:
                ordered.append(bucket.pop())
                if bucket:
                    next_active.append(bucket)

            active = next_active

        return ordered

    def __iter__(self) -> Iterator[list[CTETrainIndex]]:
        if not self.rows:
            return

        rng = random.Random(self.seed + self.epoch)

        # --------------------------------------------------------------
        # Epoch-local queues.
        # Each source row is shuffled once, then consumed by a cursor.
        # --------------------------------------------------------------
        pool_rows = {
            key: self._build_epoch_task_order(key, rng)
            for key in self._task_keys
        }
        pool_pos = {key: 0 for key in self._task_keys}
        remaining_by_task = {
            key: len(pool_rows[key])
            for key in self._task_keys
        }

        remaining_total = len(self.rows)
        pair_units_total = sum(
            count // 2
            for count in remaining_by_task.values()
        )

        batches: list[list[CTETrainIndex]] = []

        self.balanced_batches = 0
        self.positive_batches = 0
        self.singleton_only_batches = 0
        self.wraparound_rows = 0

        progress = tqdm(
            total=self._num_batches,
            desc=f"Planning CTE batches (epoch {self.epoch})",
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.2,
            leave=False,
            disable=not self._progress_enabled(),
        )

        # --------------------------------------------------------------
        # Hot-path helpers.
        # --------------------------------------------------------------
        def take_from_task(
            key: str,
            count: int,
        ) -> list[CTETrainIndex]:
            """Take ``count`` rows in O(count) via cursor movement."""

            nonlocal remaining_total, pair_units_total

            remaining = remaining_by_task[key]
            if remaining < count:
                raise RuntimeError(
                    f"task {key} contains only {remaining} rows, "
                    f"cannot take {count}"
                )

            pair_units_before = remaining // 2

            start = pool_pos[key]
            end = start + count
            selected = pool_rows[key][start:end]
            pool_pos[key] = end

            remaining -= count
            remaining_by_task[key] = remaining
            remaining_total -= count

            # Incrementally maintain sum(floor(task_remaining / 2)).
            pair_units_total += (
                remaining // 2 - pair_units_before
            )

            return selected

        def take_single_preserve_pairs(
            preferred_exclude: set[str] | None = None,
        ) -> CTETrainIndex | None:
            """Pick one filler with a single pass over task metadata.

            Priority:
              1. task not already represented in the current batch;
              2. odd-sized task pool (consuming one preserves pair capacity);
              3. smaller remaining task pool;
              4. seeded random tie break.

            Unlike the previous implementation this does not repeatedly build
            3-4 temporary task lists for every filler sample.
            """

            excluded = (
                set()
                if preferred_exclude is None
                else preferred_exclude
            )

            best_score: tuple[bool, bool, int] | None = None
            tied_keys: list[str] = []

            for key in self._task_keys:
                count = remaining_by_task[key]
                if count <= 0:
                    continue

                score = (
                    key in excluded,
                    count % 2 == 0,
                    count,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    tied_keys = [key]
                elif score == best_score:
                    tied_keys.append(key)

            if not tied_keys:
                return None

            key = rng.choice(tied_keys)
            return take_from_task(key, 1)[0]

        def task_counts(
            batch: list[CTETrainIndex],
        ) -> dict[str, int]:
            counts: dict[str, int] = defaultdict(int)

            for row in batch:
                counts[_task_key(row.task_id)] += 1

            return counts

        def register_batch(
            batch: list[CTETrainIndex],
            *,
            primary_balanced: bool = False,
        ) -> None:
            if primary_balanced:
                self.balanced_batches += 1

            counts = task_counts(batch)

            if any(count >= 2 for count in counts.values()):
                self.positive_batches += 1
            else:
                self.singleton_only_batches += 1

            batches.append(batch)
            progress.update(1)

            # Keep tqdm informative without refreshing every single batch.
            if (
                progress.n == self._num_batches
                or progress.n % 32 == 0
            ):
                progress.set_postfix(
                    balanced=self.balanced_batches,
                    positive=self.positive_batches,
                    remain=remaining_total,
                    refresh=False,
                )

        try:
            # ==========================================================
            # Tier 1: configured primary layout, e.g. 4 tasks x 4 rows.
            # ==========================================================
            while remaining_total >= self.batch_size:
                available = [
                    key
                    for key in self._task_keys
                    if remaining_by_task[key]
                    >= self.samples_per_task
                ]

                if len(available) < self.tasks_per_batch:
                    break

                selected_tasks = rng.sample(
                    available,
                    self.tasks_per_batch,
                )

                batch: list[CTETrainIndex] = []

                for key in selected_tasks:
                    batch.extend(
                        take_from_task(
                            key,
                            self.samples_per_task,
                        )
                    )

                represented_tasks = {
                    _task_key(row.task_id)
                    for row in batch
                }

                while len(batch) < self.batch_size:
                    filler = take_single_preserve_pairs(
                        represented_tasks
                    )

                    if filler is None:
                        break

                    batch.append(filler)
                    represented_tasks.add(
                        _task_key(filler.task_id)
                    )

                # Given remaining_total >= batch_size on entry, reaching this
                # branch indicates an internal accounting bug.
                if len(batch) != self.batch_size:
                    raise RuntimeError(
                        "CTE sampler failed to fill a primary batch"
                    )

                register_batch(
                    batch,
                    primary_balanced=True,
                )

            # ==========================================================
            # Tier 2 / 3: spread same-task positive pairs across the
            # remaining full batches, then use pair-preserving fillers.
            # ==========================================================
            while remaining_total >= self.batch_size:
                full_batches_left = max(
                    remaining_total // self.batch_size,
                    1,
                )

                batch: list[CTETrainIndex] = []

                if pair_units_total > 0:
                    pair_keys = [
                        key
                        for key in self._task_keys
                        if remaining_by_task[key] >= 2
                    ]

                    desired_pair_groups = (
                        pair_units_total
                        + full_batches_left
                        - 1
                    ) // full_batches_left

                    desired_pair_groups = min(
                        max(desired_pair_groups, 1),
                        self.batch_size // 2,
                        len(pair_keys),
                    )

                    selected_pair_tasks = rng.sample(
                        pair_keys,
                        desired_pair_groups,
                    )

                    for key in selected_pair_tasks:
                        batch.extend(
                            take_from_task(key, 2)
                        )

                represented_tasks = {
                    _task_key(row.task_id)
                    for row in batch
                }

                while len(batch) < self.batch_size:
                    filler = take_single_preserve_pairs(
                        represented_tasks
                    )

                    if filler is None:
                        break

                    batch.append(filler)
                    represented_tasks.add(
                        _task_key(filler.task_id)
                    )

                if len(batch) != self.batch_size:
                    raise RuntimeError(
                        "CTE sampler failed to fill an adaptive batch"
                    )

                register_batch(batch)

            # ==========================================================
            # Final remainder: consume every remaining source row exactly
            # once, then pad only the missing slots with wrap-around rows.
            # ==========================================================
            if remaining_total:
                batch: list[CTETrainIndex] = []

                for key in self._task_keys:
                    count = remaining_by_task[key]
                    if count:
                        batch.extend(
                            take_from_task(key, count)
                        )

                rng.shuffle(batch)
                used_in_batch = set(batch)
                counts = task_counts(batch)

                # First: turn singleton tasks into positives when possible.
                singleton_task_keys = [
                    key
                    for key, count in counts.items()
                    if count == 1
                ]
                rng.shuffle(singleton_task_keys)

                for key in singleton_task_keys:
                    if len(batch) >= self.batch_size:
                        break

                    # Prefer a different row from the same task. On a tiny
                    # dataset, allow true wrap-around of the same row.
                    candidates = [
                        row
                        for row in self._rows_by_task.get(key, ())
                        if row not in used_in_batch
                    ]

                    if not candidates:
                        candidates = list(
                            self._rows_by_task.get(key, ())
                        )

                    if not candidates:
                        continue

                    chosen = rng.choice(candidates)
                    batch.append(chosen)
                    used_in_batch.add(chosen)
                    self.wraparound_rows += 1

                # Second: add fresh same-task pairs if at least two slots
                # remain. This executes only for one final small batch, so a
                # little extra bookkeeping here is insignificant.
                while len(batch) + 2 <= self.batch_size:
                    candidate_tasks: list[str] = []
                    unused_cache: dict[
                        str,
                        list[CTETrainIndex],
                    ] = {}

                    for key in self._task_keys:
                        unused = [
                            row
                            for row in self._rows_by_task[key]
                            if row not in used_in_batch
                        ]

                        if len(unused) >= 2:
                            candidate_tasks.append(key)
                            unused_cache[key] = unused

                    if not candidate_tasks:
                        break

                    key = rng.choice(candidate_tasks)
                    unused = unused_cache[key]

                    # Prefer distinct episodes for this wrap-around pair.
                    by_episode: dict[
                        str,
                        list[CTETrainIndex],
                    ] = defaultdict(list)

                    for row in unused:
                        by_episode[row.episode_id].append(row)

                    episode_keys = list(by_episode)
                    rng.shuffle(episode_keys)

                    selected: list[CTETrainIndex] = []
                    for episode_id in episode_keys:
                        selected.append(
                            rng.choice(by_episode[episode_id])
                        )
                        if len(selected) == 2:
                            break

                    if len(selected) < 2:
                        remaining_candidates = [
                            row
                            for row in unused
                            if row not in selected
                        ]
                        rng.shuffle(remaining_candidates)
                        selected.extend(
                            remaining_candidates[
                                : 2 - len(selected)
                            ]
                        )

                    for row in selected:
                        batch.append(row)
                        used_in_batch.add(row)
                        self.wraparound_rows += 1

                # Final generic padding. Usually only 0-1 rows are needed.
                unused_global = [
                    row
                    for row in self.rows
                    if row not in used_in_batch
                ]
                rng.shuffle(unused_global)

                while len(batch) < self.batch_size:
                    if unused_global:
                        chosen = unused_global.pop()
                    else:
                        chosen = rng.choice(self.rows)

                    batch.append(chosen)
                    used_in_batch.add(chosen)
                    self.wraparound_rows += 1

                register_batch(batch)

        finally:
            progress.close()

        # Preserve existing behavior: the batch order itself is also
        # deterministic but randomized for seed + epoch.
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        """Exact O(1) number of batches for the current row set."""

        return self._num_batches

