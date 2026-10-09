# Stage-2 全面审查结论

## 最终架构

正式路径保持用户确定的 Zeva→FastWAM 设计：Frozen base FastWAM + Frozen CTE；Static Task Context 独立于在线 BIT/PIM；Stage-2A 训练 policy-injection/action-prior 分支；Stage-2B 在其基础上训练 CausalPrompt/PIM adapter；不使用 FastWAM-Joint、IDM 或 optional-IDM，也不把 memory 做成第三 expert。

## 已验证/保持的 Zeva 语义

1. **Static Task Context**：初始 observation + instruction 的 frozen policy readout → learned projection → demonstration bank retrieval → fixed task context。
2. **Behavior bank**：每条 demonstration 的 CTE trajectory prototype 作为 retrieval key / behavior value；本补丁不改变当前 ICL-WAM 的 `trajectory_prototype()` 定义。
3. **BIT**：仅使用当前 episode 已经完成的 effect，最多 4 个，右对齐；effect 在结束边界之后才可见。
4. **PIM**：正式 prepared path 使用当前配置默认 `phase` retrieval：同 semantic task、排除当前 episode、cosine top-K。
5. **训练/推理隔离**：prepared static/PIM cache 只用于把冻结的训练条件离线化；在线 RoboTwin 仍通过 `StaticTaskContextSession + CausalCTEHistory + CausalMemoryLifecycle/PIM` 动态运行。
6. **Addon identity**：训练和在线评测都使用同一 `task_context_identity(static, bank SHA, head SHA, top_k)`，因此 addon checkpoint 可拒绝错误 bank/head。

## 关键 bug 修复

- Stage-1 task config 的 `use_text_embed_cache=false` 不能用于 Stage-2；新增独立 Stage-2 task config 并强制 true。
- Behavior bank 原实现会全 dataset 视频读取两遍；改 metadata-only episode plan。
- Behavior CTE 原实现重新跑所有 Wan VAE；改用已验证 latent cache，仅 tail fallback。
- FastWAM readout 原实现 27,225 episode batch=1；改 batch + 16 GPU sharding。
- Behavior artifacts 缺 semantic-map SHA check；补齐 CTE/latent/current dataset 三方一致性。
- Stage-2 static context 原启动路径约 O(N²) Python retrieval；改离线向量化 episode cache。
- Stage-2 PIM 原 `__getitem__` 遍历 MemoryBank；改离线 GPU retrieval + mmap。
- Prepared PIM 现在保持 `MemoryBank.add()` 的归一化及 stable-tie insertion-order 语义。
- PIM builder 不再由 16 rank 重复解析 165 万行 JSON；仅 rank0 staging。
- Stage-2 torchrun 在模型实例化前绑定 LOCAL_RANK，避免所有进程先占 GPU0。
- 最终复核修复 `train_zeva_fastwam_prepared.py` 的 device helper 自递归/漏赋值问题；现为 `torch.cuda.set_device(LOCAL_RANK)` 后显式返回 `cuda:<local_rank>`。
- Stage-2 16 rank 使用父 shell 一次生成的统一 RUN_DIR。
- 外层 RoboTwin eval 原先没有把 static bank/head override 传给 delegated process；补显式 EVALUATION→deploy 参数链。
- 正式 artifact 写入采用临时文件/目录验证后原子提交。

## 精度与维度合同

```text
Stage-1 cached latent: BF16 bits on disk
CTE input:            FP32 [B,T,48,24,20]
action groups:        FP32 [B,T-1,4,14]
CTE retrieval:        FP32 [...,128]
CTE state:            FP32 [...,256]
phase:                FP32 [128]
effect:               FP32 [128]
FastWAM readout:      FP32 [3072]
static head:          3072 -> 1024 -> 128
static task context:  FP32 [256]
BIT:                  FP32 [4,128] + bool [4]
PIM:                  FP32 [4,128] x phase/effect + bool [4]
FastWAM video:        [3,9,384,320]
FastWAM action:       [32,14]
```

不使用理论 VAE compression factor 反推 latent shape；正式 contract 以已经 bit-exact 验证的 `[9,48,24,20]` 和在线 tail VAE exact-shape match 为准。

## 已完成的代码级检查

- 所有新增/替换 Python 文件通过 `py_compile`。
- 所有 shell 文件通过 `bash -n`。
- evaluation patch 对当前 GitHub main 的目标 anchors 已逐个核对存在；patch script 使用 exact-anchor + idempotent marker，遇到代码版本不匹配会 fail-fast，不静默误改。
- Prepared Stage-2 validator 包含真实模型 forward + addon backward smoke，正式训练前应执行。

## 上游 Zeva 对照的边界

上游公开仓库能够直接确认 static retrieval head、symmetric supervised contrastive loss、top-K static retrieval 以及 online initial-observation readout/retrieval 路径；公开搜索中没有发现完整的 behavior-bank 构建脚本。因此本补丁**没有臆造新的 bank 定义**，而是保留 ICL-WAM 当前已经采用的 `trajectory_prototype()`（CTE retrieval/state trajectory mean）并只优化其数据与执行路径。
