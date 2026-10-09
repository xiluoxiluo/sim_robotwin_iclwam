# ICL-WAM Remaining Code Modifications

## 1. 修复 Stage-2B 初始化与断点续训语义

### 文件

```text
src/fastwam/trainer.py
configs/model/zeva_fastwam.yaml
configs/task/robotwin_zeva_fastwam_pim_3cam_384.yaml
```

---

### 1.1 `configs/model/zeva_fastwam.yaml`

在 `zeva:` 下新增：

```yaml
zeva:
  training_stage: policy_injection

  # 仅用于 pim_adapter 第一次从 Stage-2A 初始化。
  # 不作为当前 stage 的断点续训入口。
  policy_checkpoint: null
```

---

### 1.2 `configs/task/robotwin_zeva_fastwam_pim_3cam_384.yaml`

修改为：

```yaml
# @package _global_

defaults:
  - robotwin_zeva_fastwam_static_3cam_384
  - _self_

model:
  zeva:
    training_stage: pim_adapter
    policy_checkpoint: null
```

启动 Stage-2B 时显式传入：

```bash
model.zeva.policy_checkpoint=/path/to/stage2A_addon.pt
```

`resume=` 只用于恢复当前 `pim_adapter` 自己的 checkpoint。

---

### 1.3 `src/fastwam/trainer.py`

新增辅助函数：

```python
def _get_zeva_policy_checkpoint(self) -> Path | None:
    if not self.zeva_training:
        return None

    zeva_cfg = self.cfg.model.get("zeva", {})
    value = zeva_cfg.get("policy_checkpoint")

    if value in (None, "", "None", "null"):
        return None

    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(
            f"zeva.policy_checkpoint does not exist: {path}"
        )

    return path
```

---

### 1.4 在 `Wan22Trainer.__init__()` 中增加 Stage-2B 首次初始化

在 optimizer 创建之前、`_apply_dit_only_train_mode()` 之后执行：

```python
self._apply_dit_only_train_mode(self.model)

if (
    self.zeva_training
    and self.zeva_training_stage == "pim_adapter"
):
    self._initialize_pim_stage_from_policy_checkpoint()
```

新增函数：

```python
def _initialize_pim_stage_from_policy_checkpoint(self) -> None:
    policy_checkpoint = self._get_zeva_policy_checkpoint()

    # 有 resume 时，表示恢复当前 pim_adapter，
    # 不应再次从 Stage-2A 初始化。
    if self.resume not in (None, "", "None", "null"):
        return

    if policy_checkpoint is None:
        raise ValueError(
            "pim_adapter training requires "
            "model.zeva.policy_checkpoint when resume is not set"
        )

    base_path = self.cfg.get("ckpt")
    zeva_cfg = self.cfg.model.get("zeva", {})
    cte_path = zeva_cfg.get("cte", {}).get("checkpoint")

    if base_path in (None, "", "None", "null"):
        raise ValueError(
            "pim_adapter initialization requires cfg.ckpt"
        )

    if cte_path in (None, "", "None", "null"):
        raise ValueError(
            "pim_adapter initialization requires "
            "model.zeva.cte.checkpoint"
        )

    payload = self.model.load_zeva_addon_checkpoint(
        str(policy_checkpoint),
        base_checkpoint_sha256=checkpoint_sha256(
            str(base_path)
        ),
        cte_checkpoint_sha256=checkpoint_sha256(
            str(cte_path)
        ),
        load_scope="policy_injection",
    )

    if payload.get("training_stage") != "policy_injection":
        raise ValueError(
            "model.zeva.policy_checkpoint must be a "
            "policy_injection checkpoint"
        )

    self.model.zeva_behavior_prefix_adapter.reset_pim_parameters()
    self.model.configure_zeva_trainable_state()

    logger.info(
        "Initialized pim_adapter from policy checkpoint: %s",
        policy_checkpoint,
    )
```

---

### 1.5 重写 `_resume_or_load_checkpoint()` 中 Zeva checkpoint 逻辑

删除以下逻辑：

```python
load_scope = (
    "policy_injection"
    if self.zeva_training_stage == "pim_adapter"
    else "all"
)
```

删除：

```python
if self.zeva_training_stage == "pim_adapter":
    if payload.get("training_stage") != "policy_injection":
        ...
    model.zeva_behavior_prefix_adapter.reset_pim_parameters()
```

`resume` 对当前 stage 永远使用完整恢复：

```python
if self.zeva_training:
    model = self.accelerator.unwrap_model(self.model)

    payload = model.load_zeva_addon_checkpoint(
        str(addon_path),
        base_checkpoint_sha256=checkpoint_sha256(
            str(base_path)
        ),
        cte_checkpoint_sha256=checkpoint_sha256(
            str(cte_path)
        ),
        load_scope="all",
    )

    checkpoint_stage = payload.get("training_stage")

    if checkpoint_stage != self.zeva_training_stage:
        raise ValueError(
            "Zeva resume stage mismatch: "
            f"checkpoint={checkpoint_stage}, "
            f"current={self.zeva_training_stage}"
        )

    # 正常恢复 optimizer / scheduler / global step
    optimizer_payload = torch.load(
        resume_path / "optimizer_scheduler.pt",
        map_location="cpu",
    )

    self.optimizer.load_state_dict(
        optimizer_payload["optimizer"]
    )
    self.scheduler.load_state_dict(
        optimizer_payload["scheduler"]
    )
```

最终语义：

```text
policy_checkpoint
  policy_injection -> pim_adapter 初始化

resume
  policy_injection -> policy_injection 续训
  pim_adapter      -> pim_adapter 续训
```

---

## 2. 对齐 Zeva Action Prior 的 BIT Attention

### 文件

```text
src/fastwam/zeva/behavior_prefix_adapter.py
```

### 函数

```python
FastWAMPolicyInjectionPrior.forward()
```

---

### 2.1 删除当前 BIT padding mask

删除：

```python
padding = ~bit_mask
empty = ~bit_mask.any(dim=-1)
padding = padding.clone()
padding[empty, 0] = False

effect_context, _ = self.effect_attention(
    phase_query[:, None],
    bit,
    bit,
    key_padding_mask=padding,
    need_weights=False,
)
```

---

### 2.2 替换为

```python
bit_mask = bit_mask.to(torch.bool)

bit_input = torch.where(
    bit_mask.unsqueeze(-1),
    bit_effects,
    self.bos_effect.expand(
        batch,
        cfg.effect_history_length,
        -1,
    ),
)

bit = (
    self.effect_project(bit_input)
    + self.effect_position
)

effect_context, _ = self.effect_attention(
    phase_query[:, None],
    bit,
    bit,
    need_weights=False,
)

effect_context = self.effect_norm(
    effect_context[:, 0]
)
```

不要修改 `CausalPromptEncoder` 中 BIT/PIM 的 `key_padding_mask`。

---

## 3. 对齐 `CausalPromptEncoder` 初始化

### 文件

```text
src/fastwam/zeva/causal_prompt.py
```

---

### 3.1 修改参数定义

将：

```python
self.brief_position = nn.Parameter(
    torch.randn(
        1,
        cfg.brief_length,
        cfg.hidden_dim,
    ) * 0.02
)

self.persistent_position = nn.Parameter(
    torch.randn(
        1,
        cfg.persistent_length,
        cfg.hidden_dim,
    ) * 0.02
)

self.bos_brief = nn.Parameter(
    torch.randn(
        1,
        1,
        cfg.hidden_dim,
    ) * 0.02
)

self.bos_persistent = nn.Parameter(
    torch.randn(
        1,
        1,
        cfg.hidden_dim,
    ) * 0.02
)
```

改为：

```python
self.brief_position = nn.Parameter(
    torch.empty(
        1,
        cfg.brief_length,
        cfg.hidden_dim,
    )
)

self.persistent_position = nn.Parameter(
    torch.empty(
        1,
        cfg.persistent_length,
        cfg.hidden_dim,
    )
)

self.bos_brief = nn.Parameter(
    torch.empty(
        1,
        1,
        cfg.hidden_dim,
    )
)

self.bos_persistent = nn.Parameter(
    torch.empty(
        1,
        1,
        cfg.hidden_dim,
    )
)
```

---

### 3.2 在 `__init__()` 末尾调用

```python
self.reset_parameters()
```

---

### 3.3 新增

```python
def reset_parameters(self) -> None:
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif (
            isinstance(module, nn.LayerNorm)
            and module.elementwise_affine
        ):
            nn.init.ones_(
                module.weight
            )

            nn.init.zeros_(
                module.bias
            )

    self.brief_attention._reset_parameters()
    self.persistent_attention._reset_parameters()

    nn.init.normal_(
        self.brief_position,
        std=0.02,
    )

    nn.init.normal_(
        self.persistent_position,
        std=0.02,
    )

    nn.init.normal_(
        self.bos_brief,
        std=0.02,
    )

    nn.init.normal_(
        self.bos_persistent,
        std=0.02,
    )
```

---

## 4. Stage-2B 增加 Zeva PIM 学习率倍率

### 文件

```text
src/fastwam/trainer.py
```

---

### 4.1 替换当前 Zeva optimizer 参数构造

当前：

```python
if self.zeva_training:
    trainable_params = list(
        self.model.zeva_trainable_parameters()
    )
```

修改为：

```python
if self.zeva_training:
    if self.zeva_training_stage == "policy_injection":
        trainable_params = list(
            self.model.zeva_trainable_parameters()
        )

        optimizer_params = trainable_params

    elif self.zeva_training_stage == "pim_adapter":
        adapter = (
            self.model
            .zeva_behavior_prefix_adapter
        )

        prompt_params = list(
            self.model
            .zeva_prompt_encoder
            .parameters()
        )

        projector_params = list(
            adapter
            .prefix_project
            .parameters()
        )

        gate_params = [
            adapter.pim_gate
        ]

        trainable_params = (
            prompt_params
            + projector_params
            + gate_params
        )

        optimizer_params = [
            {
                "params": prompt_params,
                "lr": self.learning_rate * 5.0,
            },
            {
                "params": projector_params,
                "lr": self.learning_rate * 5.0,
            },
            {
                "params": gate_params,
                "lr": self.learning_rate,
            },
        ]

    else:
        raise RuntimeError(
            f"unsupported Zeva training stage: "
            f"{self.zeva_training_stage}"
        )
else:
    trainable_params = list(
        self.model.dit.parameters()
    )

    proprio_encoder = getattr(
        self.model,
        "proprio_encoder",
        None,
    )

    if proprio_encoder is not None:
        trainable_params.extend(
            list(
                proprio_encoder.parameters()
            )
        )

    optimizer_params = trainable_params
```

然后 optimizer 修改为：

```python
self.optimizer = torch.optim.AdamW(
    optimizer_params,
    lr=self.learning_rate,
    weight_decay=self.weight_decay,
    betas=(0.9, 0.95),
)
```

---

### 4.2 增加日志

```python
if (
    self.zeva_training
    and self.zeva_training_stage == "pim_adapter"
):
    logger.info(
        "PIM LR multipliers: "
        "prompt_encoder=5.0 "
        "prefix_project=5.0 "
        "pim_gate=1.0"
    )
```

---

## 5. 增加真正的 Native FastWAM 等价测试

### 文件

```text
tests/zeva/test_fastwam_action_integration.py
```

---

### 5.1 增加 native model factory

新增一个不挂载 Zeva addon 的模型：

```python
def _native_inference_model():
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)

    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32

    model.action_expert = _InferenceActionExpert()
    model.video_expert = _InferenceVideoExpert()
    model.mot = _InferenceMoT()

    model.infer_action_scheduler = (
        _InferenceScheduler()
    )

    model.proprio_dim = None
    model.proprio_encoder = None

    model.zeva_enabled = False
    model.zeva_prompt_encoder = None
    model.zeva_behavior_prefix_adapter = None
    model.zeva_injection_mode = "exact_zeva"
    model.zeva_training_stage = "policy_injection"

    model._encode_input_image_latents_tensor = MethodType(
        lambda self, input_image, tiled=False:
        input_image.new_zeros(
            (1, 4, 1, 1, 1)
        ),
        model,
    )

    model._build_mot_attention_mask = MethodType(
        lambda self,
        video_seq_len,
        action_seq_len,
        **kwargs:
        torch.zeros(
            video_seq_len + action_seq_len,
            video_seq_len + action_seq_len,
        ),
        model,
    )

    return model.eval()
```

---

### 5.2 增加权重同步函数

```python
def _copy_fastwam_test_weights(
    source,
    target,
):
    target.action_expert.load_state_dict(
        source.action_expert.state_dict()
    )
```

如果 fake video/MoT 后续加入参数，也一并同步：

```python
if any(
    True
    for _ in source.video_expert.parameters()
):
    target.video_expert.load_state_dict(
        source.video_expert.state_dict()
    )

if any(
    True
    for _ in source.mot.parameters()
):
    target.mot.load_state_dict(
        source.mot.state_dict()
    )
```

---

### 5.3 增加测试

```python
def test_base_mode_matches_true_native_fastwam():
    zeva_model = _inference_model()
    native_model = _native_inference_model()

    _copy_fastwam_test_weights(
        zeva_model,
        native_model,
    )

    kwargs = {
        "prompt": None,
        "context": torch.zeros(
            1,
            2,
            4,
        ),
        "context_mask": torch.ones(
            1,
            2,
            dtype=torch.bool,
        ),
        "input_image": torch.zeros(
            1,
            3,
            16,
            16,
        ),
        "action_horizon": 32,
        "num_inference_steps": 1,
        "seed": 17,
    }

    native_action = native_model.infer_action(
        zeva_mode="base",
        **kwargs,
    )["action"]

    addon_base_action = (
        zeva_model.infer_action(
            zeva_mode="base",
            **kwargs,
        )["action"]
    )

    torch.testing.assert_close(
        native_action,
        addon_base_action,
        rtol=0.0,
        atol=0.0,
    )
```

---

## 6. 增加 Frozen FastWAM 权重不变测试

### 文件

```text
tests/zeva/test_fastwam_action_integration.py
```

新增辅助函数：

```python
def _clone_non_zeva_state(model):
    return {
        name: value.detach().clone()
        for name, value
        in model.state_dict().items()
        if not name.startswith(
            (
                "zeva_prompt_encoder.",
                "zeva_behavior_prefix_adapter.",
            )
        )
    }
```

新增测试：

```python
def test_policy_injection_step_does_not_modify_fastwam():
    model = _stage_model()
    model.set_zeva_training_stage(
        "policy_injection"
    )
    model.configure_zeva_trainable_state()

    before = _clone_non_zeva_state(
        model
    )

    optimizer = torch.optim.AdamW(
        model.zeva_trainable_parameters(),
        lr=1e-4,
    )

    adapter = (
        model
        .zeva_behavior_prefix_adapter
    )

    task_tokens = torch.randn(
        2,
        adapter.config.global_dim,
    )

    phase = torch.randn(
        2,
        adapter.config.phase_dim,
    )

    bit_effects = torch.randn(
        2,
        adapter.config.effect_history_length,
        adapter.config.effect_dim,
    )

    bit_mask = torch.ones(
        2,
        adapter.config.effect_history_length,
        dtype=torch.bool,
    )

    prior_mean, prior_std = adapter.prior(
        task_tokens,
        phase,
        bit_effects,
        bit_mask,
    )

    residual = adapter.action_prior_residual(
        prior_mean,
        training=True,
    )

    loss = (
        residual.float().square().mean()
        + prior_std.float().mean()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()
    optimizer.step()

    after = _clone_non_zeva_state(
        model
    )

    assert before.keys() == after.keys()

    for name in before:
        torch.testing.assert_close(
            before[name],
            after[name],
            rtol=0.0,
            atol=0.0,
        )
```

---

## 7. 增加 Stage-2B Resume 测试

### 文件

```text
tests/zeva/test_fastwam_action_integration.py
```

至少增加两个测试：

```python
def test_pim_stage_initializes_from_policy_checkpoint():
    ...
```

要求：

```text
Stage-2A checkpoint
→ load_scope=policy_injection

恢复：
  prior
  action_prior_adapter
  behavior_global_projector

不恢复：
  CausalPromptEncoder
  prefix_project
  pim_gate
```

以及：

```python
def test_pim_stage_resume_restores_pim_checkpoint():
    ...
```

要求：

```text
pim_adapter checkpoint
→ load_scope=all

恢复：
  prior
  action_prior_adapter
  behavior_global_projector
  CausalPromptEncoder
  prefix_project
  pim_gate
  optimizer
  scheduler
  global_step

不得调用：
  reset_pim_parameters()
```

---

## 8. 最终检查命令

```bash
pytest -q \
  tests/zeva/test_behavior_prefix_adapter.py \
  tests/zeva/test_fastwam_action_integration.py \
  tests/zeva/test_hydra_config.py
```

重点检查：

```text
[ ] policy_injection -> pim_adapter 初始化正常
[ ] pim_adapter -> pim_adapter resume 正常
[ ] Action Prior invalid BIT slot 以 BOS token 参与 attention
[ ] CausalPromptEncoder 使用 Zeva 初始化方式
[ ] PIM encoder/projector LR 为 base_lr * 5
[ ] PIM gate LR 为 base_lr
[ ] true native FastWAM == addon base mode
[ ] Stage-2 optimizer step 不修改任何 FastWAM 权重
```
