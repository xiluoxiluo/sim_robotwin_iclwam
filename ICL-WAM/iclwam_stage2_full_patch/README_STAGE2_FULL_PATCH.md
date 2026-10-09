# ICL-WAM Zeva → FastWAM：Stage-2 正式实现补丁

本补丁建立在已经验证完成的 Stage-1 CTE、VAE latent cache 和 phase/effect cache-v4 之上。目标不是改 Zeva 算法，而是把 RoboTwin 规模下昂贵且重复的离线计算改成可审计、可复现、可扩展的 prepared artifacts，同时保持训练与在线评测语义一致。

## 1. 已修复的问题

### Correctness

- Stage-2 独立配置强制 `data.train.use_text_embed_cache=true`，不再误继承 Stage-1 的 `false`。
- Behavior bank 同时绑定并检查：FastWAM base SHA、CTE SHA、dataset stats SHA、semantic task map SHA、latent-cache identity、camera order、VAE identity、video size。
- Behavior bank 的 CTE prototype 使用 `full_episode_prefix`，直接复用已验证的 Stage-1 BF16 latent cache；仅缺失 tail 走 Wan VAE。
- BF16 latent 在 CTE 前恢复为 FP32，与 Stage-1 cached training 数值契约一致。
- Static training retrieval 使用 leave-one-episode-out；在线评测仍使用初始 observation 现场 readout，不使用训练 episode cache。
- Prepared PIM 精确保持现有正式语义：same-task phase cosine、排除当前 episode、top-K；BIT 只包含当前 episode 已完成 effect。
- Prepared PIM candidate phase/effect 使用与 `MemoryBank.add()` 相同的 CPU FP32 L2 normalize。
- 16 卡 Stage-2 在构建大模型前根据 `LOCAL_RANK` 绑定 `cuda:<local_rank>`，避免 16 个进程都先占 `cuda:0`。
- Prepared PIM 绑定源 `phase/effect cache-v4` 的 manifest/index SHA；Stage-2 再次核对。
- Eval 链路显式透传 `task_context_mode/bank/retrieval_checkpoint/top_k`，避免 static artifacts 在 RoboTwin 子进程里被静默丢失。
- 最终 artifacts 使用临时文件/目录验证后原子提交，避免异常中断留下“看起来存在但不完整”的正式文件。

### Performance

- 不再为 behavior bank 对 129 万 window 做两遍视频 decode。
- CTE trajectory 复用 67GB latent cache。
- FastWAM initial readout 改成 batched，默认 batch=2，可按显存调大。
- Behavior bank 支持 16-GPU episode sharding。
- Static episode task context 一次离线向量化生成，Stage-2 不再启动时做 27k×27k Python retrieval。
- PIM top-K 一次离线 GPU 矩阵检索生成 mmap，Stage-2 `__getitem__` 不再遍历 36 万条 MemoryBank。
- PIM builder 只有 rank0 读取 165 万条 JSON index，避免 16 rank 重复占用大量主存。

## 2. 文件

```text
configs/task/robotwin_zeva_fastwam_stage2_3cam_384.yaml

src/fastwam/zeva/
├── cte_latent_cache.py
└── prepared_stage2.py

scripts/
├── build_zeva_robotwin_cache_from_latents.py      # 已验证 v4 dependency
├── build_zeva_robotwin_cache_from_latents.sh
├── build_zeva_behavior_bank_optimized.py
├── build_zeva_behavior_bank_optimized.sh
├── train_zeva_task_context_retrieval_optimized.py
├── train_zeva_task_context_retrieval_optimized.sh
├── build_zeva_static_episode_context_cache.py
├── build_zeva_static_episode_context_cache.sh
├── build_zeva_pim_retrieval_cache.py
├── build_zeva_pim_retrieval_cache.sh
├── validate_zeva_stage2_prepared.py
├── validate_zeva_stage2_prepared.sh
├── train_zeva_fastwam_prepared.py
├── train_zeva_fastwam_prepared.sh
└── eval_zeva_robotwin_fixed_attempts.py

patches/
└── apply_stage2_eval_patch.py
```

## 3. 覆盖并做静态检查

将压缩包内容覆盖到 ICL-WAM 根目录后：

```bash
cd /data/share/1919650160032350208/zjj/ICL-WAM
export PYTHONPATH=$PWD/src:$PYTHONPATH

python patches/apply_stage2_eval_patch.py

python -m py_compile \
  src/fastwam/zeva/cte_latent_cache.py \
  src/fastwam/zeva/prepared_stage2.py \
  scripts/build_zeva_behavior_bank_optimized.py \
  scripts/train_zeva_task_context_retrieval_optimized.py \
  scripts/build_zeva_static_episode_context_cache.py \
  scripts/build_zeva_pim_retrieval_cache.py \
  scripts/validate_zeva_stage2_prepared.py \
  scripts/train_zeva_fastwam_prepared.py \
  scripts/eval_zeva_robotwin_fixed_attempts.py
```

`apply_stage2_eval_patch.py` 是幂等的；再次运行不会重复插入。

## 4. 固定路径

```bash
ROOT=/data/share/1919650160032350208/zjj/ICL-WAM
DATA=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0

export BASE_CKPT=/请填写你的完整FastWAM基础checkpoint.pt
export CTE_CKPT=$ROOT/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt
export LATENT_CACHE=$DATA/zeva_cte_latent_cache_v1
export PHASE_CACHE=$DATA/zeva_phase_effect_cache_v4

export ART=$DATA/zeva_stage2_artifacts
mkdir -p "$ART"

export BEHAVIOR_BANK=$ART/behavior_bank.pt
export READOUT_CACHE=$ART/initial_readouts.pt
export RETRIEVAL_HEAD=$ART/static_retrieval_head.pt
export STATIC_CONTEXT_CACHE=$ART/static_episode_context.pt
export PIM_CACHE=$ART/prepared_pim_v1
```

当前正式 CTE SHA 应为：

```text
12ad6b677eb76cddabb509c8c2bb3b03872328fd01c6314529477c6b9dab7767
```

脚本会自己重新计算并验证，不依赖 README 中的字符串。

## 5. Step A：16 卡构建 Behavior Bank + Initial Readouts

```bash
GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
ZEVA_BEHAVIOR_READOUT_BATCH_SIZE=2 \
ZEVA_BEHAVIOR_READOUT_WORKERS=2 \
ZEVA_BEHAVIOR_TAIL_BATCH_SIZE=4 \
ZEVA_BEHAVIOR_TAIL_WORKERS=2 \
bash scripts/build_zeva_behavior_bank_optimized.sh
```

它会输出：

```text
behavior_bank.pt         # 每 episode 一个 key[128] + value[256]
initial_readouts.pt      # 每 episode 一个 FastWAM clean readout[3072]
```

算法定义没有变：

```text
key   = normalize(mean(CTE retrieval over full demo))
value = mean(CTE causal_interaction_state over full demo)
readout = mean(final FastWAM video/MoT hidden on initial image + text)
```

若显存很充裕，可把 `ZEVA_BEHAVIOR_READOUT_BATCH_SIZE` 从 2 提高到 4；第一次正式构建建议保持 2。

## 6. Step B：训练 Static Retrieval Head

单卡即可：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_zeva_task_context_retrieval_optimized.sh
```

结构保持 Zeva：

```text
FastWAM readout [3072]
→ Linear 1024 + LN + GELU + Dropout
→ key [128] normalized
```

训练仍使用 Zeva symmetric multi-positive supervised contrastive loss。脚本会额外输出 held-out task top1/top5、same-task/cross-task cosine。

## 7. Step C：生成训练专用 Static Episode Context Cache

```bash
CUDA_VISIBLE_DEVICES=0 \
ZEVA_STATIC_CONTEXT_BATCH_SIZE=512 \
bash scripts/build_zeva_static_episode_context_cache.sh
```

训练 episode 必须 leave-one-episode-out，因此这里预计算：

```text
initial readout
→ trained retrieval head
→ top-5 behavior bank (exclude self episode)
→ softmax weighted value[256]
```

脚本随机抽样与公共 `StaticTaskContextRetriever` 逐项对比，默认容差 `1e-5`。

**这个 cache 只用于训练加速。在线 RoboTwin 推理不会读取它。**

## 8. Step D：16 卡生成 Prepared PIM/BIT mmap

```bash
GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
ZEVA_PIM_QUERY_BATCH=1024 \
bash scripts/build_zeva_pim_retrieval_cache.sh
```

正式路径只支持 Zeva 默认：

```text
pim_retrieval_mode = phase
same semantic task
exclude current episode
cosine top-K
```

`cross_task_effect` 保留在 legacy diagnostic path，不混入正式 Stage-2。

Prepared mmap 每 query 保存：

```text
phase          [128]
BIT effects    [4,128] + mask
PIM phases     [4,128]
PIM effects    [4,128] + mask/scores
```

对当前 1,295,287 query，最终 prepared PIM 大约 8–9GB。构建时还需要 staging 临时空间，建议 `ART` 所在磁盘至少预留 15GB。

## 9. Step E：正式训练前总验证

单卡即可：

```bash
CUDA_VISIBLE_DEVICES=0 \
ZEVA_STAGE2_VALIDATE_SAMPLES=64 \
ZEVA_STAGE2_MODEL_SMOKE=1 \
ZEVA_STAGE2_BACKWARD_SMOKE=1 \
bash scripts/validate_zeva_stage2_prepared.sh
```

它验证：

- 所有 artifact SHA/identity；
- 50 semantic tasks；
- phase-cache source manifest/index SHA；
- prepared mmap shape/dtype；
- 随机 window 的 dataset_index / episode / task / raw_step；
- `video [3,9,384,320]`、`action [32,14]`、context、phase/BIT/PIM/static context 的完整合同；
- 一次真实 Frozen FastWAM + Zeva Stage-2 forward；
- 一次 addon backward，要求存在 finite 且 non-zero trainable gradients。

必须看到：

```text
PREPARED STAGE-2 VALIDATION: PASSED
FROZEN FASTWAM + ZEVA STAGE-2 FORWARD/BACKWARD: PASSED
```

再开始正式训练。

## 10. Step F：Stage-2A Policy Injection（16 卡）

```bash
GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/train_zeva_fastwam_prepared.sh
```

默认 task config：

```text
model.zeva.training_stage=policy_injection
```

这一步保持 Frozen FastWAM，只训练 policy-injection 分支（Gaussian action prior / action residual / task-global injection 对应参数）。

## 11. Step G：Stage-2B PIM Adapter（16 卡）

Stage-2A 生成 addon 后，设：

```bash
export POLICY_ADDON=/path/to/stage2A/checkpoints/weights/step_xxxxxx_addon.pt

GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/train_zeva_fastwam_prepared.sh \
  model.zeva.training_stage=pim_adapter \
  model.zeva.policy_checkpoint="$POLICY_ADDON"
```

现有 `Wan22Trainer` 会只加载 Stage-2A 的 policy-injection 范围，重置并训练 PIM adapter 所需的 CausalPrompt/prefix/gate 参数；FastWAM 仍保持 frozen。

## 12. 在线评测

补丁后的固定种子评测会把 static artifacts 显式透传进 RoboTwin 子进程：

```bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --ckpt "$BASE_CKPT" \
  --cte-checkpoint "$CTE_CKPT" \
  --addon-checkpoint /path/to/final_addon.pt \
  --task-context-bank "$BEHAVIOR_BANK" \
  --task-context-retrieval-checkpoint "$RETRIEVAL_HEAD" \
  --task-context-top-k 5 \
  --seed 42 \
  --mode pim_on
```

在线 static context 的语义仍然是：

```text
first observation + instruction
→ Frozen FastWAM readout
→ retrieval head
→ behavior bank top-K
→ one fixed task context for the attempt
```

在线 CTE/BIT/PIM 也仍然按真实执行历史更新；不会读取 `static_episode_context.pt` 或 `prepared_pim_v1`。

## 13. 为什么不直接修改旧 ZevaStage2Dataset

旧路径保留作诊断和兼容，正式训练使用 `PreparedZevaStage2Dataset`。这样：

- 不破坏现有 cross-task-effect 实验；
- 可以把 prepared path 与 legacy `MemoryBank` 做等价性校验；
- 线上 PIM 生命周期完全不受训练加速代码影响；
- 出问题时容易回退和定位。

## 14. 不需要重做的内容

以下都已经完成，不要重新生成：

```text
semantic task map                 ✅
VAE latent cache                  ✅
Stage-1 CTE 5000 steps            ✅
CTE readiness                     ✅
phase/effect cache-v4             ✅
cache-v4 deep recompute abs=0     ✅
```

本补丁从 Behavior Bank 开始继续。
