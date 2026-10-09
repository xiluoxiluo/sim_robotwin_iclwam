"""
cd /data/share/1919650160032350208/zjj/ICL-WAM
export DIFFSYNTH_MODEL_BASE_PATH="/data/share/1919650160032350208/zjj/fastwam/checkpoints"
export PYTHONPATH="/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0,1 \
python scripts/precompute_text_embeds.py \
--config-name train \
task=robotwin_zeva_fastwam_3cam_384 \
+overwrite=false \
data.train.text_embedding_cache_dir=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/text_embeds_cache_new \
data.val.text_embedding_cache_dir=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/text_embeds_cache_new
          """

import hashlib
import json
import logging
import os
import re
import socket
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16

# Optional environment variables:
#   PRECOMPUTE_NUM_GPUS=4       -> when launched with plain `python`, use only 4 visible GPUs
#   PRECOMPUTE_BATCH_SIZE=32     -> per-GPU batch size
#
# If neither is set, plain `python` automatically uses all visible GPUs.
# `torchrun` remains fully supported and takes precedence over automatic spawning.


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _is_external_distributed_launch() -> bool:
    """Return True when the process was launched by torchrun/slurm-style env vars."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _find_free_port() -> int:
    """Pick a free localhost TCP port for automatic single-node spawning."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_num_gpus_for_auto_spawn(cfg: DictConfig) -> int:
    """Resolve how many GPUs to use when the script is started with plain `python`."""
    if not torch.cuda.is_available():
        return 0

    available = torch.cuda.device_count()
    if available <= 0:
        return 0

    raw_num_gpus = os.environ.get("PRECOMPUTE_NUM_GPUS")
    if raw_num_gpus is None:
        raw_num_gpus = cfg.get("precompute_num_gpus", None)

    if raw_num_gpus is None:
        return available

    requested = int(raw_num_gpus)
    if requested < 1:
        raise ValueError(f"precompute_num_gpus must be >= 1, got {requested}")
    if requested > available:
        raise ValueError(
            f"Requested {requested} GPUs, but only {available} CUDA device(s) are visible. "
            "Check CUDA_VISIBLE_DEVICES or PRECOMPUTE_NUM_GPUS."
        )
    return requested


def _resolve_batch_size(cfg: DictConfig) -> int:
    """Resolve per-GPU text-encoder batch size."""
    raw_batch_size = os.environ.get("PRECOMPUTE_BATCH_SIZE")
    if raw_batch_size is None:
        raw_batch_size = cfg.get("text_embed_batch_size", DEFAULT_BATCH_SIZE)

    batch_size = int(raw_batch_size)
    if batch_size < 1:
        raise ValueError(f"text_embed_batch_size must be >= 1, got {batch_size}")
    return batch_size


def _init_distributed():
    """Initialize one process per GPU from torchrun/auto-spawn environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig, *, verbose: bool = True):
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    context_lens = set()

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )

        for ds in raw_dirs:
            ds_str = str(ds)
            if ds_str not in dataset_dirs:
                dataset_dirs.append(ds_str)

        cache_dir_path = Path(str(cache_dir)).expanduser()
        if cache_dir_path not in cache_dirs:
            cache_dirs.append(cache_dir_path)

        context_len = node.get("context_len")
        if context_len is not None:
            context_lens.add(int(context_len))

        if verbose:
            logger.info(
                "Discovered dataset node `%s` with %d dataset_dirs.",
                node_path,
                len(raw_dirs),
            )

    return dataset_dirs, cache_dirs, context_lens


def _resolve_context_len(context_lens: set[int]) -> int:
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _read_unique_prompts(dataset_dirs: list[str], *, verbose: bool = True) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")

        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                task = str(record["task"])
                prompt = DEFAULT_PROMPT.format(task=task)
                total_task_rows += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)

    if verbose:
        logger.info(
            "Loaded %d task rows from %d datasets, deduplicated to %d prompts.",
            total_task_rows,
            len(dataset_dirs),
            len(prompts),
        )
    return prompts


def _get_override_prompt(override_instruction: Any) -> str | None:
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    return DEFAULT_PROMPT.format(task=task)


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _cache_filename(prompt: str, context_len: int, enc_id: str) -> str:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{hashed}.t5_len{context_len}.{enc_id}.pt"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    try:
        torch.save(payload, str(tmp_path))
        os.replace(tmp_path, output_path)
    finally:
        # If torch.save failed, do not leave stale temporary files behind.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _distributed_sum(values: list[int], device: torch.device, is_distributed: bool) -> list[int]:
    """Sum integer counters across ranks."""
    if not is_distributed:
        return values

    tensor = torch.tensor(values, device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [int(v) for v in tensor.cpu().tolist()]


def _aggregate_stats(
    stats: dict[str, dict[str, int]],
    cache_dirs: list[Path],
    device: torch.device,
    is_distributed: bool,
    rank: int,
):
    if not is_distributed:
        return

    if not cache_dirs:
        return

    counts_tensor = torch.tensor(
        [
            [
                stats[str(cache_dir)]["new"],
                stats[str(cache_dir)]["overwrite"],
                stats[str(cache_dir)]["skip"],
            ]
            for cache_dir in cache_dirs
        ],
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        counts_cpu = counts_tensor.cpu()
        for idx, cache_dir in enumerate(cache_dirs):
            key = str(cache_dir)
            stats[key]["new"] = int(counts_cpu[idx, 0].item())
            stats[key]["overwrite"] = int(counts_cpu[idx, 1].item())
            stats[key]["skip"] = int(counts_cpu[idx, 2].item())


def _run_precompute(cfg: DictConfig):
    """Worker body. In distributed mode every process owns exactly one GPU."""
    is_distributed, rank, world_size, local_rank = _init_distributed()

    try:
        is_main = (not is_distributed) or rank == 0
        if is_distributed:
            logger.info(
                "[rank %d/%d] worker started on cuda:%d",
                rank,
                world_size,
                local_rank,
            )
        elif torch.cuda.is_available():
            logger.info("Single-GPU worker started on cuda:0")
        else:
            logger.info("CUDA unavailable; running on CPU")

        overwrite = _to_bool(cfg.get("overwrite", True))
        batch_size = _resolve_batch_size(cfg)

        model_cfg = cfg.model
        if model_cfg is None:
            raise ValueError("`cfg.model` is required.")
        if cfg.data is None:
            raise ValueError("`cfg.data` is required.")

        dataset_dirs, cache_dirs, context_lens = _collect_dataset_settings(
            cfg.data,
            verbose=is_main,
        )
        if not cache_dirs:
            raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")

        context_len = _resolve_context_len(context_lens)
        model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
        tokenizer_model_id = str(
            model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID)
        )
        redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
        enc_id = _model_id_to_enc_id(model_id)

        override_prompt = _get_override_prompt(cfg.get("override_instruction"))
        if override_prompt is not None:
            all_prompts = [override_prompt]
            if is_main:
                logger.info(
                    "Using override_instruction; skipping dataset scan and encoding exactly 1 prompt."
                )
        else:
            if not dataset_dirs:
                raise ValueError("No `dataset_dirs` found under `cfg.data`.")
            all_prompts = _read_unique_prompts(dataset_dirs, verbose=is_main)

        if not all_prompts:
            if is_main:
                logger.warning("No prompts found from tasks.jsonl; nothing to do.")
            return

        # Deterministic rank sharding. Every prompt is owned by exactly one rank,
        # so different GPUs never write the same prompt cache file.
        local_prompts = all_prompts[rank::world_size] if is_distributed else all_prompts

        stats = {
            str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
            for cache_dir in cache_dirs
        }

        # Filter cache hits BEFORE loading the 5B text encoder. This makes reruns cheap
        # when overwrite=False and most/all prompt embeddings already exist.
        fully_cached_local = 0
        if not overwrite:
            prompts_to_encode: list[str] = []
            for prompt in local_prompts:
                filename = _cache_filename(prompt, context_len, enc_id)
                fully_cached = True
                for cache_dir in cache_dirs:
                    if not (cache_dir / filename).exists():
                        fully_cached = False
                        break

                if fully_cached:
                    fully_cached_local += 1
                    for cache_dir in cache_dirs:
                        stats[str(cache_dir)]["skip"] += 1
                else:
                    prompts_to_encode.append(prompt)

            local_prompts = prompts_to_encode

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda:0")
        else:
            device = torch.device("cpu")

        fully_cached_global, prompts_to_encode_global = _distributed_sum(
            [fully_cached_local, len(local_prompts)],
            device,
            is_distributed,
        )

        if is_main and not overwrite:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                prompts_to_encode_global,
            )

        if is_distributed:
            logger.info(
                "[rank %d/%d] assigned %d prompt(s) after cache filtering.",
                rank,
                world_size,
                len(local_prompts),
            )

        # Nothing to encode globally: aggregate skip stats and exit without loading model.
        if prompts_to_encode_global == 0:
            _aggregate_stats(stats, cache_dirs, device, is_distributed, rank)
            if is_main:
                logger.info("All text embeddings are already cached; model loading skipped.")
                for cache_dir in cache_dirs:
                    key = str(cache_dir)
                    logger.info(
                        "Cache dir: %s | new=%d overwrite=%d skip=%d",
                        key,
                        stats[key]["new"],
                        stats[key]["overwrite"],
                        stats[key]["skip"],
                    )
            if is_distributed:
                dist.barrier()
            return

        torch_dtype = torch.bfloat16

        if is_main:
            logger.info(
                "Preparing text encoder with model_id=%s tokenizer_model_id=%s "
                "world_size=%d dtype=%s context_len=%d per_gpu_batch_size=%d overwrite=%s",
                model_id,
                tokenizer_model_id,
                world_size,
                torch_dtype,
                context_len,
                batch_size,
                overwrite,
            )

        _, text_config, _, tokenizer_config = _resolve_configs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
        )

        # Resolve/download model files safely in multi-process mode.
        #
        # Important: ``download_if_necessary()`` mutates each process-local config
        # object (e.g. it fills ``text_config.path``). A distributed barrier only
        # synchronizes execution; it does NOT copy that mutated Python object from
        # rank 0 to the other ranks. Therefore rank 0 downloads first, then every
        # non-main rank calls download_if_necessary() after the files already exist
        # so its own config object also receives a valid local path.
        if is_distributed:
            if is_main:
                logger.info("[rank 0/%d] resolving/downloading text encoder files", world_size)
                text_config.download_if_necessary()
                tokenizer_config.download_if_necessary()

            # Do not let other ranks inspect the cache until rank 0 has completed.
            dist.barrier()

            if not is_main:
                # Normally this is only a cache lookup on a single-node/shared-FS
                # launch. It is also safer for multi-node launches where each node
                # may have a node-local model cache.
                text_config.download_if_necessary()
                tokenizer_config.download_if_necessary()

            # Make sure every rank has finished resolving its local path before any
            # rank starts loading the model.
            dist.barrier()
        else:
            text_config.download_if_necessary()
            tokenizer_config.download_if_necessary()

        text_model_path = text_config.path
        tokenizer_path = tokenizer_config.path

        if text_model_path is None:
            raise RuntimeError(
                f"[rank {rank}/{world_size}] text_config.path is still None after "
                "download_if_necessary(). Check the model cache/model_id: "
                f"{model_id}"
            )
        if tokenizer_path is None:
            raise RuntimeError(
                f"[rank {rank}/{world_size}] tokenizer_config.path is still None after "
                "download_if_necessary(). Check tokenizer_model_id: "
                f"{tokenizer_model_id}"
            )

        logger.info(
            "[rank %d/%d] resolved text model path: %s",
            rank,
            world_size,
            text_model_path,
        )

        text_encoder = None
        tokenizer = None

        # Ranks with zero local work do not need to allocate a 5B model on their GPU.
        if local_prompts:
            logger.info(
                "[rank %d/%d] loading text encoder on %s",
                rank,
                world_size,
                device,
            )
            text_encoder = _load_registered_model(
                text_model_path,
                "wan_video_text_encoder",
                torch_dtype=torch_dtype,
                device=str(device),
            ).eval()
            tokenizer = HuggingfaceTokenizer(
                name=tokenizer_path,
                seq_len=context_len,
                clean="whitespace",
            )

        over_length_local = 0

        # A single progress bar from rank 0 avoids garbled multi-process terminal output.
        # It represents GPU0/rank0's shard; the final summary is global across all GPUs.
        with tqdm(
            total=len(local_prompts),
            desc=(
                f"Encoding prompts [GPU {local_rank}, rank {rank}/{world_size}]"
                if is_distributed
                else "Encoding prompts"
            ),
            unit="prompt",
            dynamic_ncols=True,
            disable=is_distributed and rank != 0,
        ) as pbar:
            if local_prompts:
                assert text_encoder is not None
                assert tokenizer is not None

                with torch.inference_mode():
                    for start in range(0, len(local_prompts), batch_size):
                        batch_prompts = local_prompts[start : start + batch_size]

                        ids, mask = tokenizer(
                            batch_prompts,
                            return_mask=True,
                            add_special_tokens=True,
                        )
                        ids = ids.to(device)
                        mask = mask.to(device=device, dtype=torch.bool)

                        over_length_local += int(mask.all(dim=1).sum().item())
                        context = text_encoder(ids, mask)

                        for i, prompt in enumerate(batch_prompts):
                            filename = _cache_filename(prompt, context_len, enc_id)
                            context_i = (
                                context[i]
                                .detach()
                                .to(device="cpu", dtype=torch.bfloat16)
                                .contiguous()
                            )
                            mask_i = (
                                mask[i]
                                .detach()
                                .to(device="cpu", dtype=torch.bool)
                                .contiguous()
                            )
                            payload = {
                                "context": context_i,
                                "mask": mask_i,
                            }

                            for cache_dir in cache_dirs:
                                cache_path = cache_dir / filename
                                key = str(cache_dir)

                                if cache_path.exists() and not overwrite:
                                    stats[key]["skip"] += 1
                                    continue

                                if cache_path.exists():
                                    stats[key]["overwrite"] += 1
                                else:
                                    stats[key]["new"] += 1

                                _atomic_torch_save(payload, cache_path)

                        pbar.update(len(batch_prompts))

                        # Release the largest temporary tensors as soon as each batch is done.
                        del context, ids, mask

        over_length_global, prompts_encoded_global = _distributed_sum(
            [over_length_local, len(local_prompts)],
            device,
            is_distributed,
        )

        _aggregate_stats(stats, cache_dirs, device, is_distributed, rank)

        if is_main:
            logger.info("Finished precomputing text embeddings with %d worker(s).", world_size)
            logger.info(
                "Over-length prompts (mask all True, i.e. no padding after "
                "truncation/max_length=%d): %d/%d",
                context_len,
                over_length_global,
                prompts_encoded_global,
            )
            for cache_dir in cache_dirs:
                key = str(cache_dir)
                logger.info(
                    "Cache dir: %s | new=%d overwrite=%d skip=%d",
                    key,
                    stats[key]["new"],
                    stats[key]["overwrite"],
                    stats[key]["skip"],
                )

        # Ensure all writes are complete before workers tear down the process group.
        if is_distributed:
            dist.barrier()

        # Explicitly drop model before destroying the process group.
        del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _spawn_worker(
    local_rank: int,
    world_size: int,
    cfg_container: dict[str, Any],
    master_addr: str,
    master_port: int,
):
    """Entry point used by torch.multiprocessing.spawn for plain-python multi-GPU mode."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Helpful NCCL failure behavior on recent PyTorch versions.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    setup_logging(log_level=logging.INFO)
    worker_cfg = OmegaConf.create(cfg_container)
    _run_precompute(worker_cfg)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    # 1) torchrun / scheduler launch: environment already defines ranks.
    if _is_external_distributed_launch():
        _run_precompute(cfg)
        return

    # 2) Plain `python script.py`: automatically use all visible GPUs.
    num_gpus = _resolve_num_gpus_for_auto_spawn(cfg)

    if num_gpus > 1:
        master_addr = "127.0.0.1"
        master_port = int(os.environ.get("MASTER_PORT", _find_free_port()))
        cfg_container = OmegaConf.to_container(cfg, resolve=False)
        assert isinstance(cfg_container, dict)

        logger.info(
            "Auto multi-GPU mode: spawning %d worker processes across %d visible GPU(s).",
            num_gpus,
            torch.cuda.device_count(),
        )
        logger.info(
            "Per-GPU batch size=%d; effective encoder batch size is approximately %d.",
            _resolve_batch_size(cfg),
            _resolve_batch_size(cfg) * num_gpus,
        )

        mp.spawn(
            _spawn_worker,
            args=(num_gpus, cfg_container, master_addr, master_port),
            nprocs=num_gpus,
            join=True,
        )
        return

    # 3) One visible GPU or CPU.
    _run_precompute(cfg)


if __name__ == "__main__":
    main()
