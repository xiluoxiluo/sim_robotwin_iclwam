# ICL-WAM：按 Zeva 机制对齐的代码修改方案

> 目标：在 **不改 Zeva 核心机制** 的前提下，将 Zeva 的 frozen Cosmos policy backbone 替换为 frozen FastWAM。  
> 本文只列出当前仓库中**仍需要修改的部分**。已经与 Zeva / FastWAM 一致的代码不在本文重复说明。  
> 目标仓库：`xiluoxiluo/ICL-WAM`，以当前 `main` 分支为基准。

---

## 0. 修改后的正式训练/评测语义

正式实现只保留下面四种状态：

```text
Stage 1:
CTE training

Stage 2A:
Frozen FastWAM
Frozen CTE
Train:
  - PolicyInjectionPrior
  - action_prior_adapter
  - behavior_global_projector

Stage 2B:
Frozen FastWAM
Frozen CTE
Frozen Stage-2A policy-injection modules
Train:
  - CausalPromptEncoder
  - prefix_project
  - pim_gate
```

正式评测模式：

```text
base
  = vanilla FastWAM

zeva_stage2
  = FastWAM
  + Zeva task/global behavior prefix
  + Zeva Gaussian action prior
  + Zeva action-prior residual
  - PIM residual

pim_shadow
  = zeva_stage2
  + 完整运行 BIT/PIM lifecycle 和 retrieval
  - PIM residual injection

pim_on
  = zeva_stage2
  + BIT/PIM lifecycle
  + Causal Prompt
  + gated PIM residual injection
```

必须满足：

```text
pim_shadow ≡ zeva_stage2
base ≠ zeva_stage2
pim_on ≠ pim_shadow
```

---

# 1. `configs/model/zeva_fastwam.yaml`

## 1.1 修改默认参数

将当前：

```yaml
memory:
  pim_max_entries: 256

task_context:
  mode: pooling
  top_k: 1
  key_dim: 256

adapter:
  prior_dropout_rate: 0.0
```

修改为：

```yaml
zeva:
  training_stage: policy_injection

  memory:
    bit_size: 4
    pim_top_k: 4
    pim_max_entries: 64
    pim_retrieval_mode: phase
    merge_threshold: 0.85
    beta_phase: 0.5
    beta_effect: 0.5

  task_context:
    mode: static
    bank_path: null
    readout_cache_path: null
    retrieval_checkpoint: null
    top_k: 5
    key_dim: ${model.zeva.cte.retrieval_dim}
    value_dim: ${model.zeva.prompt.global_dim}
    temperature: 0.07

  adapter:
    mode: exact_zeva
    memory_dim: 256
    action_horizon: 32
    action_hidden_dim: 1024
    num_heads: 8
    mlp_ratio: 2.0
    gate_init: 0.0
    prior_hidden_dim: 256
    prior_num_heads: 4
    prior_loss_weight: 0.01
    prior_dropout_rate: 0.4
    prior_inference_guidance_scale: 0.5
    leading_condition_steps: 0
```

其中：

```yaml
training_stage: policy_injection
```

作为默认 Stage-2A。

`leading_condition_steps: 0` 保持不变。

## 1.2 正式配置不要再默认 text pooling

保留 `task_tokens_from_context()` 作为实验/ablation 工具即可，但正式 `zeva_fastwam.yaml` 默认必须是：

```yaml
task_context:
  mode: static
```

不要删除现有 `pooling` 支持，以便后续做 `Static Task Context vs Text Pooling` 消融。

---

# 2. 新增两个 Stage-2 task 配置

建议新增：

```text
configs/task/robotwin_zeva_fastwam_policy_3cam_384.yaml
configs/task/robotwin_zeva_fastwam_pim_3cam_384.yaml
```

## 2.1 `robotwin_zeva_fastwam_policy_3cam_384.yaml`

```yaml
# @package _global_

defaults:
  - robotwin_zeva_fastwam_static_3cam_384
  - _self_

model:
  zeva:
    training_stage: policy_injection
```

该阶段只训练：

```text
prior
action_prior_adapter
behavior_global_projector
```

PIM 输入即使存在于 dataset/cache 中，也不得参与 policy conditioning。

## 2.2 `robotwin_zeva_fastwam_pim_3cam_384.yaml`

```yaml
# @package _global_

defaults:
  - robotwin_zeva_fastwam_static_3cam_384
  - _self_

model:
  zeva:
    training_stage: pim_adapter
```

该阶段启动时必须：

```text
load Stage-2A addon checkpoint
freeze Stage-2A policy-injection modules
fresh-init PIM modules
train only PIM modules
```

Stage-2A checkpoint 通过现有 `resume=` 参数传入即可。

---

# 3. `src/fastwam/runtime.py`

当前已经有 `exact_zeva` 接口，不需要新建 adapter；只修改默认选择和传递 training stage。

## 3.1 `exact_zeva` 改为正式默认值

当前：

```python
adapter_mode = str(adapter_cfg.get("mode", "memory_residual"))
```

修改为：

```python
adapter_mode = str(adapter_cfg.get("mode", "exact_zeva"))
```

`memory_residual` 分支可以保留用于 ablation，但不得作为 Zeva 正式路径 fallback。

## 3.2 static task context 改为默认

当前：

```python
model.zeva_task_context_mode = str(
    zeva.get("task_context", {}).get("mode", "pooling")
)
```

修改为：

```python
model.zeva_task_context_mode = str(
    zeva.get("task_context", {}).get("mode", "static")
)
```

## 3.3 传入 training stage

在 `attach_zeva_addon()` 后增加：

```python
training_stage = str(zeva.get("training_stage", "policy_injection"))

if training_stage not in {"policy_injection", "pim_adapter"}:
    raise ValueError(
        "zeva.training_stage must be 'policy_injection' or 'pim_adapter'"
    )

model.set_zeva_training_stage(training_stage)
```

---

# 4. `src/fastwam/zeva/behavior_prefix_adapter.py`

## 4.1 修改 dataclass 默认 prior dropout

当前：

```python
prior_dropout_rate: float = 0.0
```

修改为：

```python
prior_dropout_rate: float = 0.4
```

配置文件和 dataclass 默认值都要修改。

## 4.2 增加两组明确的参数接口

在 `ExactZevaPolicyInjectionAdapter` 中增加：

```python
def policy_injection_parameters(self):
    yield from self.prior.parameters()
    yield from self.action_prior_adapter.parameters()
    yield from self.behavior_global_projector.parameters()


def pim_adapter_parameters(self):
    yield from self.prefix_project.parameters()
    yield self.pim_gate
```

后续 Trainer 不再使用：

```python
list(self.zeva_behavior_prefix_adapter.parameters())
```

作为全部可训练参数。

## 4.3 增加 PIM 模块重新初始化接口

```python
@torch.no_grad()
def reset_pim_parameters(self) -> None:
    nn.init.xavier_uniform_(self.prefix_project.weight)
    nn.init.zeros_(self.prefix_project.bias)
    self.pim_gate.fill_(float(self.config.prompt_gate_init))
```

Stage-2B 加载 Stage-2A checkpoint 时，不加载 `prefix_project` 和 `pim_gate`，然后调用此函数。

---

# 5. `src/fastwam/zeva/causal_prompt.py`

当前 `CausalPromptEncoder.forward()` 将一个 fused causal prompt 人为扩展成：

```text
[fused, current_phase, BIT_summary, PIM_summary]
```

正式路径改回 Zeva 的单 fused prompt。

## 5.1 修改返回类型

当前：

```python
def forward(...) -> tuple[Tensor, Tensor]:
```

修改为：

```python
def forward(...) -> Tensor:
```

## 5.2 删除 4-token 拼接

当前末尾的 `output_tokens` / `output_mask` 全部删除，改为：

```python
causal_prompt = self.fusion(
    torch.cat(
        (
            query,
            bit_context[:, 0],
            pim_context[:, 0],
        ),
        dim=-1,
    )
)

return causal_prompt
```

输出固定为：

```text
[B, prompt.hidden_dim]
```

当前配置即 `[B,256]`。

## 5.3 PIM availability 使用原始 `pim_mask`

不再构造：

```text
behavior_memory_mask: [B,4]
```

后续直接：

```python
has_pim = pim_mask.any(dim=-1, keepdim=True)
```

---

# 6. `ExactZevaPolicyInjectionAdapter` 改成单 Causal Prompt 接口

修改以下函数：

```text
causal_prompt_prefix()
behavior_prefix_slot()
prepend_behavior_prefix_slot()
```

不再接收：

```text
memory_tokens [B,4,D]
memory_mask   [B,4]
```

改成：

```text
causal_prompt [B,D]
pim_mask      [B,K]
```

## 6.1 `causal_prompt_prefix()`

```python
def causal_prompt_prefix(
    self,
    causal_prompt: Tensor,
    pim_mask: Tensor,
    *,
    enable_pim: bool = True,
) -> Tensor:
    if causal_prompt.ndim != 2:
        raise ValueError("causal_prompt must be [B,D]")

    if causal_prompt.shape[-1] != self.config.memory_dim:
        raise ValueError(
            f"causal_prompt last dim must be {self.config.memory_dim}"
        )

    if pim_mask.ndim != 2 or pim_mask.shape[0] != causal_prompt.shape[0]:
        raise ValueError("pim_mask must be [B,K]")

    prompt = self.prefix_project(
        causal_prompt.to(
            device=self.prefix_project.weight.device,
            dtype=self.prefix_project.weight.dtype,
        )
    )[:, None]

    if not enable_pim:
        return torch.zeros_like(prompt)

    has_pim = (
        pim_mask.any(dim=-1, keepdim=True)
        .to(device=prompt.device, dtype=prompt.dtype)
        .unsqueeze(-1)
    )

    return torch.tanh(self.pim_gate).to(prompt.dtype) * prompt * has_pim
```

## 6.2 `behavior_prefix_slot()`

```python
def behavior_prefix_slot(
    self,
    task_tokens: Tensor,
    causal_prompt: Tensor,
    pim_mask: Tensor,
    *,
    enable_pim: bool = True,
) -> Tensor:

    base_prefix = self.behavior_global_projector(
        task_tokens.to(
            device=self.behavior_global_projector.weight.device,
            dtype=self.behavior_global_projector.weight.dtype,
        )
    )[:, None]

    pim_delta = self.causal_prompt_prefix(
        causal_prompt,
        pim_mask,
        enable_pim=enable_pim,
    )

    return base_prefix + pim_delta.to(
        device=base_prefix.device,
        dtype=base_prefix.dtype,
    )
```

注意 `enable_pim=False` 只关闭 PIM residual，不能关闭 `base_prefix`。

## 6.3 `prepend_behavior_prefix_slot()`

修改为显式要求 `task_tokens`：

```python
def prepend_behavior_prefix_slot(
    self,
    context: Tensor,
    context_mask: Tensor,
    causal_prompt: Tensor,
    pim_mask: Tensor,
    task_tokens: Tensor,
    *,
    enable_pim: bool = True,
) -> tuple[Tensor, Tensor]:

    prefix = self.behavior_prefix_slot(
        task_tokens,
        causal_prompt,
        pim_mask,
        enable_pim=enable_pim,
    )

    slot_mask = torch.ones(
        (context.shape[0], 1),
        device=context.device,
        dtype=torch.bool,
    )

    return (
        torch.cat(
            (
                prefix.to(device=context.device, dtype=context.dtype),
                context,
            ),
            dim=1,
        ),
        torch.cat(
            (
                slot_mask,
                context_mask.to(device=context.device, dtype=torch.bool),
            ),
            dim=1,
        ),
    )
```

删除 `task_tokens=None` 的 fallback。

---

# 7. `src/fastwam/models/wan22/fastwam.py`

## 7.1 增加 training stage 状态

初始化 Zeva addon 属性处增加：

```python
self.zeva_training_stage = "policy_injection"
```

## 7.2 增加 `set_zeva_training_stage()`

```python
def set_zeva_training_stage(self, stage: str):
    stage = str(stage)

    if stage not in {"policy_injection", "pim_adapter"}:
        raise ValueError(
            "Zeva training stage must be "
            "'policy_injection' or 'pim_adapter'"
        )

    self.zeva_training_stage = stage
    return self
```

## 7.3 重写 `zeva_trainable_parameters()`

当前整套 addon 全部返回，必须改成按 stage：

```python
def zeva_trainable_parameters(self):
    if (
        not self.zeva_enabled
        or self.zeva_prompt_encoder is None
        or self.zeva_behavior_prefix_adapter is None
    ):
        return []

    adapter = self.zeva_behavior_prefix_adapter

    if self.zeva_training_stage == "policy_injection":
        return list(adapter.policy_injection_parameters())

    if self.zeva_training_stage == "pim_adapter":
        return (
            list(self.zeva_prompt_encoder.parameters())
            + list(adapter.pim_adapter_parameters())
        )

    raise RuntimeError(
        f"unsupported Zeva training stage: {self.zeva_training_stage}"
    )
```

## 7.4 增加统一 trainable-state 配置函数

```python
def configure_zeva_trainable_state(self) -> None:
    if not self.zeva_enabled:
        raise RuntimeError("Zeva addon is not attached")

    self.eval()
    self.requires_grad_(False)

    adapter = self.zeva_behavior_prefix_adapter
    stage = self.zeva_training_stage

    if stage == "policy_injection":
        adapter.prior.train()
        adapter.prior.requires_grad_(True)

        adapter.action_prior_adapter.train()
        adapter.action_prior_adapter.requires_grad_(True)

        adapter.behavior_global_projector.train()
        adapter.behavior_global_projector.requires_grad_(True)
        return

    if stage == "pim_adapter":
        self.zeva_prompt_encoder.train()
        self.zeva_prompt_encoder.requires_grad_(True)

        adapter.prefix_project.train()
        adapter.prefix_project.requires_grad_(True)

        adapter.pim_gate.requires_grad_(True)
        return

    raise RuntimeError(f"unsupported Zeva training stage: {stage}")
```

---

# 8. `src/fastwam/trainer.py`

## 8.1 修改 `_apply_dit_only_train_mode()`

当前 Zeva 分支整套 addon 都打开，替换为：

```python
if bool(getattr(model, "zeva_enabled", False)):
    model.configure_zeva_trainable_state()
    return
```

## 8.2 初始化 Trainer 时记录 stage

在 `self.zeva_training` 后增加：

```python
self.zeva_training_stage = (
    str(getattr(self.model, "zeva_training_stage", "policy_injection"))
    if self.zeva_training
    else None
)
```

日志增加：

```python
logger.info(
    "Zeva training stage: %s",
    self.zeva_training_stage,
)
```

---

# 9. Stage-2A / Stage-2B forward 分离

仍保留现有：

```python
_training_mode == "zeva_stage2"
```

无需修改 dataset 的 `_training_mode`。

真正训练阶段由 `self.zeva_training_stage` 控制。

## 9.1 static task context 不允许静默 fallback

建议：

```python
task_mode = str(
    getattr(self, "zeva_task_context_mode", "static")
)

if sample.get("task_context") is None:
    if task_mode == "static":
        raise ValueError(
            "formal Zeva training requires static task_context"
        )

    if task_mode != "pooling":
        raise ValueError(
            f"unsupported Zeva task_context mode: {task_mode}"
        )

    task_tokens = task_tokens_from_context(
        context,
        context_mask,
        task_dim,
    )
else:
    task_tokens = sample["task_context"].to(device=self.device)
```

## 9.2 Stage-2A 不构造 PIM Causal Prompt

```python
stage = self.zeva_training_stage
batch = context.shape[0]
prompt_dim = int(self.zeva_prompt_encoder.config.hidden_dim)
pim_k = int(self.zeva_prompt_encoder.config.persistent_length)

if stage == "policy_injection":
    causal_prompt = torch.zeros(
        (batch, prompt_dim),
        device=self.device,
        dtype=self.torch_dtype,
    )
    pim_mask_for_policy = torch.zeros(
        (batch, pim_k),
        device=self.device,
        dtype=torch.bool,
    )

elif stage == "pim_adapter":
    causal_prompt = self.zeva_prompt_encoder(
        task_tokens=task_tokens,
        current_phase=sample["phase"].to(self.device),
        bit_effects=sample["bit_effects"].to(self.device),
        bit_mask=sample["bit_mask"].to(self.device),
        pim_phases=sample["pim_phases"].to(self.device),
        pim_effects=sample["pim_effects"].to(self.device),
        pim_mask=sample["pim_mask"].to(self.device),
    )
    pim_mask_for_policy = sample["pim_mask"].to(
        self.device,
        dtype=torch.bool,
    )

else:
    raise RuntimeError(f"unsupported Zeva training stage: {stage}")
```

## 9.3 Gaussian action prior 两阶段都保留 forward

```python
zeva_prior_mean, zeva_prior_std = adapter.prior(
    task_tokens,
    sample["phase"].to(self.device),
    sample["bit_effects"].to(self.device),
    sample["bit_mask"].to(self.device),
)

zeva_action_residual = adapter.action_prior_residual(
    zeva_prior_mean,
    training=True,
)
```

区别只在参数是否 trainable。

---

# 10. `forward_zeva_action_train()` 修改参数语义

将：

```text
behavior_memory
behavior_memory_mask
```

改为：

```text
causal_prompt
pim_mask
```

prefix 注入：

```python
context, context_mask = self._prepend_exact_zeva_behavior_slot(
    context=context,
    context_mask=context_mask,
    causal_prompt=causal_prompt,
    pim_mask=pim_mask,
    task_tokens=zeva_task_tokens,
    enable_pim=(
        self.zeva_training_stage == "pim_adapter"
    ),
)
```

Stage-2A 保留 base task behavior prefix，但关闭 PIM residual；Stage-2B 打开 PIM residual。

## 10.1 prior NLL 只在 Stage-2A 加入总 loss

```python
if self.zeva_training_stage == "policy_injection":
    if zeva_prior_mean is None or zeva_prior_std is None:
        raise ValueError(
            "policy_injection stage requires action-prior outputs"
        )

    prior_loss = gaussian_prior_nll(
        clean_action.float(),
        zeva_prior_mean.float(),
        zeva_prior_std.float(),
        action_valid,
    )

    loss = (
        loss
        + prior_loss
        * float(
            self.zeva_behavior_prefix_adapter.config.prior_loss_weight
        )
    )

    prior_metrics = {
        "behavior_prior_nll": float(prior_loss.detach().cpu())
    }
else:
    prior_metrics = {}
```

PIM stage 中 action prior 继续参与 policy forward，但不再把固定 prior NLL 加入训练目标。

---

# 11. `zeva_parameter_report()` 改成 stage whitelist

```python
def zeva_parameter_report(self) -> dict[str, object]:
    trainable = [
        name for name, p in self.named_parameters()
        if p.requires_grad
    ]
    frozen = [
        name for name, p in self.named_parameters()
        if not p.requires_grad
    ]

    stage = self.zeva_training_stage

    if stage == "policy_injection":
        allowed = (
            "zeva_behavior_prefix_adapter.prior.",
            "zeva_behavior_prefix_adapter.action_prior_adapter.",
            "zeva_behavior_prefix_adapter.behavior_global_projector.",
        )
    elif stage == "pim_adapter":
        allowed = (
            "zeva_prompt_encoder.",
            "zeva_behavior_prefix_adapter.prefix_project.",
            "zeva_behavior_prefix_adapter.pim_gate",
        )
    else:
        raise RuntimeError(
            f"unsupported Zeva training stage: {stage}"
        )

    invalid = [
        name
        for name in trainable
        if not any(name.startswith(prefix) for prefix in allowed)
    ]

    if invalid:
        raise AssertionError(
            f"Unexpected trainable Zeva parameters "
            f"for stage={stage}: {invalid[:10]}"
        )

    return {
        "training_stage": stage,
        "trainable_names": trainable,
        "trainable_count": sum(
            self.get_parameter(name).numel()
            for name in trainable
        ),
        "frozen_count": sum(
            self.get_parameter(name).numel()
            for name in frozen
        ),
    }
```

---

# 12. `src/fastwam/zeva/checkpoint.py`

Stage-2B 从 Stage-2A 继续训练时，需要 selective load。

## 12.1 checkpoint 增加 `training_stage`

保存 payload 增加：

```python
"training_stage": str(training_stage),
```

`FastWAM.save_zeva_addon_checkpoint()` 也写：

```python
"training_stage": self.zeva_training_stage,
```

## 12.2 `load_addon_checkpoint()` 增加 `load_scope`

```python
load_scope: str = "all"
```

支持：

```text
all
policy_injection
```

## 12.3 Stage-2B 只加载 Stage-2A policy modules

```python
if load_scope == "policy_injection":
    adapter_state = payload["behavior_prefix_adapter"]

    keep_prefixes = (
        "prior.",
        "action_prior_adapter.",
        "behavior_global_projector.",
    )

    policy_state = {
        key: value
        for key, value in adapter_state.items()
        if key.startswith(keep_prefixes)
    }

    missing, unexpected = behavior_prefix_adapter.load_state_dict(
        policy_state,
        strict=False,
    )

    if unexpected:
        raise ValueError(
            f"unexpected policy-injection checkpoint keys: {unexpected}"
        )

    return payload
```

此时不加载：

```text
causal_prompt_encoder
prefix_project
pim_gate
```

---

# 13. `trainer.py` 的 resume/load 逻辑

```python
load_scope = "all"

if (
    self.zeva_training
    and self.zeva_training_stage == "pim_adapter"
):
    load_scope = "policy_injection"
```

加载：

```python
payload = self.accelerator.unwrap_model(
    self.model
).load_zeva_addon_checkpoint(
    str(resume_path),
    base_checkpoint_sha256=checkpoint_sha256(str(base_path)),
    cte_checkpoint_sha256=checkpoint_sha256(str(cte_path)),
    load_scope=load_scope,
)
```

PIM stage 必须检查：

```python
if payload.get("training_stage") != "policy_injection":
    raise ValueError(
        "pim_adapter training must initialize from "
        "a policy_injection checkpoint"
    )
```

然后：

```python
model.zeva_behavior_prefix_adapter.reset_pim_parameters()
model.configure_zeva_trainable_state()
```

---

# 14. 修正 `pim_shadow` / `zeva_stage2` inference 语义

`infer_action()` 允许值改为：

```python
if zeva_mode not in {
    "base",
    "zeva_stage2",
    "pim_shadow",
    "pim_on",
}:
    raise ValueError(...)
```

## 14.1 PIM gate 不再控制 action-prior residual

当前 exact path 中：

```python
if gate_override is not None and float(gate_override) == 0.0:
    gated_residual = torch.zeros_like(action_tokens)
```

这一逻辑必须从 `exact_zeva` 正式路径删除。

对于 exact Zeva：

```python
if zeva_action_residual is None:
    raise ValueError(
        "exact Zeva injection requires an action-prior residual"
    )

action_tokens = action_tokens + zeva_action_residual.to(
    device=action_tokens.device,
    dtype=action_tokens.dtype,
)
```

`gate_override` 如需保留，只用于 legacy `memory_residual` ablation。

---

# 15. `infer_action()` 的 prefix 行为修改

当前只有 `pim_on` 才 prepend behavior prefix，改为所有非 base Zeva mode 都使用 base behavior prefix：

```python
if (
    zeva_mode != "base"
    and self.zeva_injection_mode == "exact_zeva"
):
    if zeva_task_tokens is None:
        raise ValueError(
            "formal Zeva inference requires explicit "
            "static zeva_task_tokens"
        )

    context, context_mask = (
        self._prepend_exact_zeva_behavior_slot(
            context=context,
            context_mask=context_mask,
            causal_prompt=causal_prompt,
            pim_mask=pim_mask,
            task_tokens=zeva_task_tokens,
            enable_pim=(zeva_mode == "pim_on"),
        )
    )
```

因此：

```text
zeva_stage2:
  global behavior prefix ON
  action prior ON
  PIM residual OFF

pim_shadow:
  global behavior prefix ON
  action prior ON
  BIT/PIM lifecycle RUN
  PIM residual OFF

pim_on:
  global behavior prefix ON
  action prior ON
  BIT/PIM lifecycle RUN
  PIM residual ON
```

---

# 16. `configs/sim_robotwin_zeva.yaml`

保留默认：

```yaml
zeva_mode: pim_on
```

但允许值更新为：

```yaml
# base | zeva_stage2 | pim_shadow | pim_on
zeva_mode: pim_on
```

正式评测至少运行四组：

```text
base
zeva_stage2
pim_shadow
pim_on
```

不要再把 `pim_shadow` 定义成 vanilla FastWAM。

---

# 17. Formal inference 禁止 text-pooling fallback

当前 `infer_action()` 缺 `zeva_task_tokens` 时会调用 `task_tokens_from_context()`。

正式 exact Zeva 改成：

```python
if (
    zeva_mode != "base"
    and self.zeva_injection_mode == "exact_zeva"
    and zeva_task_tokens is None
):
    raise ValueError(
        "exact Zeva inference requires explicit static "
        "task-context tokens"
    )
```

需要 pooling ablation 时，由上层显式生成并传入 `zeva_task_tokens`，不要在 policy 内部静默 fallback。

---

# 18. 测试修改

目录：

```text
tests/zeva/
```

至少增加/修改以下测试。

## 18.1 `test_policy_injection_trainable_whitelist`

Stage-2A 只有：

```text
zeva_behavior_prefix_adapter.prior.*
zeva_behavior_prefix_adapter.action_prior_adapter.*
zeva_behavior_prefix_adapter.behavior_global_projector.*
```

可训练。

## 18.2 `test_pim_adapter_trainable_whitelist`

Stage-2B 只有：

```text
zeva_prompt_encoder.*
zeva_behavior_prefix_adapter.prefix_project.*
zeva_behavior_prefix_adapter.pim_gate
```

可训练。

## 18.3 `test_causal_prompt_returns_single_fused_vector`

```python
causal_prompt = prompt_encoder(...)
assert causal_prompt.shape == (B, hidden_dim)
```

删除 exact path 的 `[B,4,D]` 断言。

## 18.4 `test_pim_gate_zero_matches_zeva_stage2`

固定全部输入和 seed，比较：

```text
zeva_stage2
pim_on with pim_gate=0
```

要求：

```python
torch.testing.assert_close(
    stage2_action,
    pim_gate_zero_action,
    rtol=0.0,
    atol=0.0,
)
```

## 18.5 `test_pim_shadow_matches_zeva_stage2`

完整运行 BIT/PIM retrieval，但不注入：

```python
torch.testing.assert_close(
    stage2_action,
    shadow_action,
    rtol=0.0,
    atol=0.0,
)
```

## 18.6 `test_base_mode_matches_native_fastwam`

同权重、同输入、同 seed：

```text
FastWAM.infer_action(...)
FastWAM+Zeva.infer_action(zeva_mode="base", ...)
```

必须完全一致。

## 18.7 `test_pim_on_changes_policy_when_memory_is_valid`

nonzero `prefix_project` + nonzero `pim_gate` + valid PIM 时，不同 PIM 内容应改变 action。

## 18.8 `test_pim_stage_loads_only_policy_injection_checkpoint`

验证 `load_scope="policy_injection"` 只恢复：

```text
prior
action_prior_adapter
behavior_global_projector
```

而不恢复：

```text
CausalPromptEncoder
prefix_project
pim_gate
```

---

# 19. 现有测试同步修改

重点：

```text
tests/zeva/test_behavior_prefix_adapter.py
tests/zeva/test_fastwam_action_integration.py
```

删除 exact-Zeva 路径对：

```text
behavior_memory.shape == [B,4,D]
behavior_memory_mask.shape == [B,4]
memory_mask[:, -1]
```

的依赖。

改成：

```text
causal_prompt [B,D]
pim_mask      [B,K]
```

Legacy `BehaviorPrefixAdapter` 的 4-token 测试可保留为 ablation，但不要与 exact path 共用 contract。

---

# 20. 推荐训练命令

## 20.1 Stage-2A：Policy Injection

```bash
RUN_ID=zeva_policy_stage2 bash scripts/train_zero1.sh 8   task=robotwin_zeva_fastwam_policy_3cam_384   ckpt=/path/to/fastwam_base.pt   model.zeva.cte.checkpoint=/path/to/cte.pt   model.zeva.task_context.bank_path=/path/to/task_context_bank.pt   model.zeva.task_context.retrieval_checkpoint=/path/to/static_retrieval.pt
```

checkpoint 应记录：

```text
training_stage = policy_injection
```

## 20.2 Stage-2B：PIM Adapter

```bash
RUN_ID=zeva_pim_stage2 bash scripts/train_zero1.sh 8   task=robotwin_zeva_fastwam_pim_3cam_384   ckpt=/path/to/fastwam_base.pt   model.zeva.cte.checkpoint=/path/to/cte.pt   model.zeva.task_context.bank_path=/path/to/task_context_bank.pt   model.zeva.task_context.retrieval_checkpoint=/path/to/static_retrieval.pt   resume=/path/to/policy_stage2/step_XXXXX_addon.pt
```

启动日志必须清楚显示 Stage-2B 的 trainable/frozen 参数白名单。

---

# 21. 推荐实施顺序

```text
Step 1
  修改 YAML 默认值

Step 2
  runtime.py：
    exact_zeva 默认
    static 默认
    training_stage 传递

Step 3
  stage-specific parameter whitelist
  修改 trainer freeze/optimizer

Step 4
  checkpoint selective load
  完成 Stage-2A → Stage-2B 初始化链路

Step 5
  CausalPromptEncoder 改为单 fused vector

Step 6
  ExactZevaPolicyInjectionAdapter 改 causal_prompt + pim_mask 接口

Step 7
  修改 FastWAM Stage-2 forward

Step 8
  修正 zeva_stage2 / pim_shadow / pim_on inference semantics

Step 9
  更新 tests

Step 10
  先跑 unit tests，再跑固定 seed GPU equivalence tests
```

---

# 22. Codex 最终验收条件

```text
[ ] zeva_fastwam.yaml 默认 task_context.mode == static
[ ] pim_max_entries == 64
[ ] prior_dropout_rate == 0.4
[ ] exact_zeva 是正式默认 adapter
[ ] training_stage 支持 policy_injection / pim_adapter

[ ] Stage-2A 只训练 prior/action adapter/global projector
[ ] Stage-2B 只训练 CausalPromptEncoder/prefix projector/PIM gate
[ ] Stage-2B 从 Stage-2A checkpoint selective-load policy modules
[ ] Stage-2B PIM modules fresh-init

[ ] CausalPromptEncoder exact path 输出 [B,D]
[ ] PIM availability 由原始 pim_mask.any() 控制
[ ] PIM gate 不控制 action-prior residual

[ ] base == native FastWAM
[ ] pim_shadow == zeva_stage2
[ ] pim_gate=0 == zeva_stage2
[ ] valid PIM + nonzero gate 可以改变 action

[ ] formal exact-Zeva inference 不再静默使用 text pooling
[ ] pooling 仅作为显式 ablation 保留
```

只有以上全部通过，才认为此次修改完成。
