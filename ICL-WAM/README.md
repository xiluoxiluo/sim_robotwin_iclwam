# ICLWAM：把 Zeva 因果记忆接到 FastWAM

ICLWAM 在 FastWAM 的 RoboTwin action path 上加入 Zeva 的因果记忆模块。FastWAM 仍然负责根据图像、语言和本体状态生成动作；Zeva 负责从历史交互中提取当前 phase、短期 BIT 和跨尝试保留的 PIM，再把这些信息转换成 FastWAM action hidden space 中的条件。

一次推理可以简单理解为：

~~~text
历史图像和动作
        ↓
       CTE
        ↓
   BIT / PIM 检索
        ↓
  Causal Prompt + Gaussian Action Prior
        ↓
   冻结的 FastWAM action path
        ↓
      RoboTwin 动作
~~~

FastWAM 主干不重新训练。CTE 先单独训练，Stage 2 只训练 Zeva addon，部署时按 episode 在线更新 BIT/PIM。

本文只说明已有 FastWAM RoboTwin checkpoint 之后的训练和测试流程。FastWAM 本身的安装、预训练和数据下载请参考原项目文档。

## 1. 准备环境和数据

下面假设在仓库根目录 ICLWAM/ 执行命令：

~~~bash
cd /path/to/ICL-WAM/ICLWAM
export PYTHONPATH="$PWD/src"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
~~~

至少需要：

~~~text
data/robotwin2.0/robotwin2.0/       # RoboTwin LeRobot 数据
data/robotwin2.0/dataset_stats.json # 与该数据集对应的统计量
third_party/RoboTwin/               # RoboTwin 代码和 assets
/absolute/path/to/fastwam_base.pt   # 已有的 FastWAM RoboTwin checkpoint
~~~

当前 Zeva RoboTwin V1 的数据合同固定为：

- 三路相机：cam_high、cam_left_wrist、cam_right_wrist；
- action/state 维度：14；
- 一个 action chunk：32 个 action；
- 每 4 个 action 组成一个 CTE transition；
- 每 4 个 transition 形成一个 completed effect；
- 最终图像尺寸：[3, 384, 320]；
- action 和 state 使用同一个 dataset_stats.json。

Stage 1 和评测还需要 Wan2.2、tokenizer、VAE 以及 ActionDiT 初始化文件。若 ActionDiT 文件不存在，可以先生成：

~~~bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
~~~

首次使用数据集时，先生成文本 embedding cache：

~~~bash
python scripts/precompute_text_embeds.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384
~~~

如果 stats 或文本 cache 不在默认位置，需要在训练和测试命令中用 Hydra 覆盖对应路径。

## 2. 训练流程

已有 FastWAM checkpoint 后，按下面顺序执行：

~~~text
FastWAM checkpoint
        ↓
Stage 1：训练 CTE
        ↓
构建 phase/effect cache
        ↓
Stage 2A：训练 policy injection addon
        ↓（可选）
Stage 2B：加载 Stage 2A，只训练 PIM adapter
        ↓
RoboTwin 固定 seed 测试
~~~

最简路径先使用 task_context.mode=pooling，只验证 Zeva 到 FastWAM 的训练和部署链路；正式实验再构建 static task-context bank。

## 3. Stage 1：训练 CTE

CTE 学习历史视觉、动作与 phase/effect 的因果表示。它不需要 FastWAM checkpoint，但需要 RoboTwin 数据、统计量和 Wan VAE。

~~~bash
python scripts/train_zeva_cte.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte
~~~

多卡训练使用分布式启动器。第一个参数是每台机器的 GPU 数量；`batch_size` 按每个进程计算，`NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT` 可用于多机启动：

~~~bash
bash scripts/train_zeva_cte.sh 8 \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte
~~~

训练完成后主要文件是：

~~~text
runs/zeva_cte/cte.pt
runs/zeva_cte/config.yaml
runs/zeva_cte/dataset_manifest.json
runs/zeva_cte/metrics.jsonl
~~~

断点续训：

~~~bash
python scripts/train_zeva_cte.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte \
  resume=./runs/zeva_cte/cte.pt
~~~

分布式断点续训将同一组覆盖参数传给启动器即可：

~~~bash
bash scripts/train_zeva_cte.sh 8 \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  device=cuda \
  output_dir=./runs/zeva_cte \
  resume=./runs/zeva_cte/cte.pt
~~~

cte.pt 必须和后面的数据尺寸、相机顺序、action 维度保持一致。不要把 CTE checkpoint 当作 FastWAM 的 ckpt 使用。

## 4. 构建 phase/effect cache

Stage 2 使用离线 cache 对齐训练窗口中的 CTE phase、BIT 和 PIM 输入。cache 由冻结的 CTE 根据完整 episode 历史生成：

~~~bash
python scripts/build_zeva_robotwin_cache.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  model.zeva.cte.checkpoint=./runs/zeva_cte/cte.pt \
  model.zeva.cache.path=./data/robotwin2.0/zeva_cache/v4
~~~

生成目录至少包含：

~~~text
data/robotwin2.0/zeva_cache/v4/
├── manifest.json
├── episode_index.json
└── phase_effect-*.safetensors
~~~

如果更换数据集、dataset_stats.json、CTE checkpoint、相机顺序或 action 对齐参数，需要重新构建 cache。manifest.json 会检查这些身份信息，防止错误复用旧 cache。

## 5. Stage 2A：训练 Zeva policy injection addon

Stage 2A 使用已有 FastWAM checkpoint 作为冻结 backbone，只训练 Gaussian action prior、action-prior adapter 和 task/global behavior projector。

先用 pooling 路径跑通训练，不需要额外的 static task-context 文件：

~~~bash
BASE_CKPT=/absolute/path/to/fastwam_base.pt
CTE_CKPT=./runs/zeva_cte/cte.pt
CACHE_DIR=./data/robotwin2.0/zeva_cache/v4

python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.task_context.mode=pooling \
  output_dir=./runs/zeva_stage2a \
  mixed_precision=bf16
~~~

输出的 addon 通常位于：

~~~text
runs/zeva_stage2a/checkpoints/weights/step_XXXXXX_addon.pt
~~~

这个文件只保存 Zeva addon，不包含 FastWAM 主干。部署时仍然必须同时提供同一份 BASE_CKPT 和 CTE_CKPT。

Stage 2A 续训使用 state directory：

~~~bash
python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.task_context.mode=pooling \
  output_dir=./runs/zeva_stage2a \
  resume=./runs/zeva_stage2a/checkpoints/state/step_005000
~~~

## 6. Stage 2B：可选的 PIM adapter 训练

Stage 2B 先加载 Stage 2A addon，然后冻结 policy-injection 分支，只训练 CausalPromptEncoder、prefix_project 和 pim_gate。

~~~bash
POLICY_CKPT=./runs/zeva_stage2a/checkpoints/weights/step_010000_addon.pt

python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_pim_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.policy_checkpoint=$POLICY_CKPT \
  model.zeva.task_context.mode=pooling \
  output_dir=./runs/zeva_stage2b \
  mixed_precision=bf16
~~~

如果只验证 action-prior 路径，Stage 2A 完成后可以直接测试；如果要测试完整的 PIM residual，则使用 Stage 2B addon：

~~~bash
PIM_CKPT=./runs/zeva_stage2b/checkpoints/weights/step_XXXXXX_addon.pt
~~~

## 7. Static task context（正式实验可选）

默认配置的正式模式是 static，它会从初始图像、文本和 CTE 行为原型中读取 task context。没有这些文件时，可以使用 pooling 作为 baseline；使用 static 时，先构建 behavior bank 和 FastWAM 初始 readout：

~~~bash
python scripts/build_zeva_behavior_bank.py \
  --config-name train \
  task=robotwin_zeva_fastwam_static_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.task_context.bank_path=./runs/zeva_task_context/behavior_bank.pt \
  model.zeva.task_context.readout_cache_path=./runs/zeva_task_context/readouts.pt
~~~

再训练 retrieval head：

~~~bash
python scripts/train_zeva_task_context_retrieval.py \
  --config-name train \
  task=robotwin_zeva_fastwam_static_3cam_384 \
  model.zeva.task_context.bank_path=./runs/zeva_task_context/behavior_bank.pt \
  model.zeva.task_context.readout_cache_path=./runs/zeva_task_context/readouts.pt \
  model.zeva.task_context.retrieval_checkpoint=./runs/zeva_task_context/retrieval_head.pt
~~~

之后训练和测试时去掉 task_context.mode=pooling，并传入这三个 static 文件。static bank、retrieval head、FastWAM checkpoint、CTE checkpoint 和 stats 必须属于同一套实验，代码会检查它们的 hash 和尺寸。

例如 Stage 2A 使用 static task context：

~~~bash
STATIC_BANK=./runs/zeva_task_context/behavior_bank.pt
STATIC_READOUT=./runs/zeva_task_context/readouts.pt
STATIC_HEAD=./runs/zeva_task_context/retrieval_head.pt

python scripts/train_zeva_fastwam.py \
  --config-name train \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  ckpt=$BASE_CKPT \
  model.zeva.cte.checkpoint=$CTE_CKPT \
  model.zeva.cache.path=$CACHE_DIR \
  model.zeva.task_context.mode=static \
  model.zeva.task_context.bank_path=$STATIC_BANK \
  model.zeva.task_context.readout_cache_path=$STATIC_READOUT \
  model.zeva.task_context.retrieval_checkpoint=$STATIC_HEAD \
  output_dir=./runs/zeva_stage2a_static \
  mixed_precision=bf16
~~~

## 8. RoboTwin 测试

### 8.1 固定 seed 测试

先确认 RoboTwin assets 已安装，且 third_party/RoboTwin/policy/fastwam_policy 可以由脚本创建。对同一个 task 和 seed，建议分别测试 base、zeva_stage2、pim_shadow 和 pim_on。

base 只测试原始 FastWAM：

~~~bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --ckpt $BASE_CKPT \
  --seed 0 \
  --max-attempts 4 \
  --mode base
~~~

pim_shadow 运行 CTE/BIT/PIM lifecycle，但不加载 addon，也不改变 FastWAM 输出：

~~~bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --ckpt $BASE_CKPT \
  --cte-checkpoint $CTE_CKPT \
  --seed 0 \
  --max-attempts 4 \
  --mode pim_shadow
~~~

zeva_stage2 和 pim_on 需要 addon。若沿用 Stage 2 的 pooling baseline，直接用 Hydra 入口显式覆盖 task-context mode：

~~~bash
python experiments/robotwin/eval_robotwin_single.py \
  --config-name sim_robotwin_zeva.yaml \
  task=robotwin_zeva_fastwam_policy_3cam_384 \
  model.zeva.task_context.mode=pooling \
  ckpt=$BASE_CKPT \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  EVALUATION.cte_checkpoint=$CTE_CKPT \
  EVALUATION.addon_checkpoint=$PIM_CKPT \
  EVALUATION.zeva_mode=pim_on \
  EVALUATION.fixed_seed=0 \
  EVALUATION.max_attempts=4 \
  EVALUATION.action_horizon=32 \
  EVALUATION.replan_steps=8 \
  EVALUATION.skip_get_obs_within_replan=false
~~~

如果要测试 Stage 2A，只需把 addon 改成 Stage 2A addon，并把 zeva_mode 改成 zeva_stage2。

如果使用 static task context，可以改用固定 seed 封装入口，并传入 bank 和 retrieval head：

~~~bash
python scripts/eval_zeva_robotwin_fixed_attempts.py \
  --task click_alarmclock \
  --ckpt $BASE_CKPT \
  --cte-checkpoint $CTE_CKPT \
  --addon-checkpoint $PIM_CKPT \
  --task-context-bank ./runs/zeva_task_context/behavior_bank.pt \
  --task-context-retrieval-checkpoint ./runs/zeva_task_context/retrieval_head.pt \
  --task-context-top-k 5 \
  --seed 0 \
  --max-attempts 4 \
  --mode pim_on
~~~

### 8.2 四种测试模式

| 模式 | 需要的 checkpoint | 作用 |
| --- | --- | --- |
| base | FastWAM | 原始 FastWAM baseline |
| zeva_stage2 | FastWAM + CTE + Stage 2A addon | task behavior prefix + Gaussian action prior |
| pim_shadow | FastWAM + CTE | 只运行 memory lifecycle，检查 memory 是否影响 vanilla output |
| pim_on | FastWAM + CTE + Stage 2 addon | 完整 Zeva 条件，包括 PIM residual |

固定 seed 对比时，task、seed、max_attempts 和 replan_steps 要保持一致。max_attempts=4 表示同一个 fixed-seed episode 最多重试四次，不是四个独立 episode。

Zeva 测试必须满足：

- action_horizon=32；
- replan_steps 是 4 的倍数；
- skip_get_obs_within_replan=false，因为 CTE 需要每个已执行 action 对应的 after-frame；
- pim_top_k 与训练时的 prompt.persistent_length 一致，默认都是 4。

## 9. 结果和排错

常见输出位置：

~~~text
runs/zeva_cte/                       # CTE checkpoint 和日志
data/robotwin2.0/zeva_cache/v4/      # phase/effect cache
runs/zeva_stage2a/                   # Stage 2A addon
runs/zeva_stage2b/                   # Stage 2B addon
evaluate_results/robotwin/            # RoboTwin 测试结果和日志
~~~

常见问题：

1. requires an existing pretrained_norm_stats：检查 data/robotwin2.0/dataset_stats.json，并确认训练、cache 和评测使用同一份文件。
2. Missing text embedding cache：重新运行 scripts/precompute_text_embeds.py，并确认 context_len=128。
3. requires an existing frozen FastWAM base checkpoint：ckpt 必须是 FastWAM checkpoint，不能填 cte.pt、addon 或 cache 目录。
4. cache manifest mismatch：数据、stats、CTE、相机顺序或尺寸不一致，重新构建 cache。
5. static task context requires ...：没有 static bank 时，在训练和 pooling 测试命令中加入 model.zeva.task_context.mode=pooling。
6. addon checkpoint mismatch：addon 必须由同一个 FastWAM base、同一个 CTE 和同一套 task-context artifacts 训练得到。
7. CUDA 或 RoboTwin assets 报错：Stage 2 和正式 rollout 需要 CUDA、SAPIEN/Curobo/pytorch3d 以及 RoboTwin assets；CPU 只适合做小型代码 smoke test。

## 10. 本地代码检查

没有真实 checkpoint 或 RoboTwin 环境时，可以先运行：

~~~bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts experiments/robotwin/fastwam_policy
git diff --check
~~~

这些检查只能验证代码和接口，不能替代真实 RoboTwin rollout。最终比较应在相同 task、相同 seed 和相同 retry 设置下进行 base → zeva_stage2/pim_shadow → pim_on 对照。
