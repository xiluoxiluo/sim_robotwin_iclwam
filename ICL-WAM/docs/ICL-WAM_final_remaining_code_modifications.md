# ICL-WAM Remaining Code Modifications — Final Pass

## 1. 修复 Cosine Scheduler 破坏 PIM 5:5:1 LR 比例

### 文件

```text
src/fastwam/trainer.py
```

---

### 1.1 修改 import

当前：

```python
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
```

修改为：

```python
from torch.optim.lr_scheduler import (
    ConstantLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
```

并确认文件顶部已有：

```python
import math
```

如果没有，则增加：

```python
import math
```

---

### 1.2 修改 `_build_scheduler()`

找到：

```python
def _build_scheduler(
    self,
    scheduler_type,
    total_train_steps: int,
    warmup_steps: int = 0,
):
```

将当前 cosine 分支：

```python
if scheduler_type == "cosine":
    main_scheduler = CosineAnnealingLR(
        self.optimizer,
        T_max=remaining_steps,
        eta_min=self.learning_rate * 0.01,
    )
```

替换为：

```python
if scheduler_type == "cosine":

    def cosine_factor(step: int) -> float:
        progress = min(
            max(
                float(step)
                / float(max(remaining_steps, 1)),
                0.0,
            ),
            1.0,
        )

        cosine = 0.5 * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )

        # Keep every optimizer parameter group's
        # own base LR ratio unchanged.
        #
        # Example for pim_adapter:
        #
        # prompt_encoder: 5 * base_lr
        # prefix_project: 5 * base_lr
        # pim_gate:       1 * base_lr
        #
        # LambdaLR multiplies all groups by
        # the same scalar schedule.
        return 0.01 + 0.99 * cosine

    main_scheduler = LambdaLR(
        self.optimizer,
        lr_lambda=cosine_factor,
    )
```

保留：

```python
elif scheduler_type == "constant":
    main_scheduler = ConstantLR(
        self.optimizer,
        factor=1.0,
        total_iters=remaining_steps,
    )
```

---

### 1.3 推荐直接替换完整 `_build_scheduler()`

```python
def _build_scheduler(
    self,
    scheduler_type,
    total_train_steps: int,
    warmup_steps: int = 0,
):
    scheduler_type = (
        str(scheduler_type)
        .strip()
        .lower()
    )

    total_train_steps = max(
        int(total_train_steps),
        1,
    )

    warmup_steps = min(
        max(
            int(warmup_steps),
            0,
        ),
        total_train_steps - 1,
    )

    remaining_steps = max(
        total_train_steps
        - warmup_steps,
        1,
    )

    if scheduler_type == "cosine":

        def cosine_factor(
            step: int,
        ) -> float:
            progress = min(
                max(
                    float(step)
                    / float(
                        max(
                            remaining_steps,
                            1,
                        )
                    ),
                    0.0,
                ),
                1.0,
            )

            cosine = 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )

            # Final LR = 1% of each
            # parameter group's own
            # initial LR.
            return (
                0.01
                + 0.99 * cosine
            )

        main_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=cosine_factor,
        )

    elif scheduler_type == "constant":
        main_scheduler = ConstantLR(
            self.optimizer,
            factor=1.0,
            total_iters=remaining_steps,
        )

    else:
        raise ValueError(
            "Unsupported "
            "lr_scheduler_type: "
            f"{scheduler_type}. "
            "Expected one of: "
            "['cosine', 'constant']."
        )

    if warmup_steps <= 0:
        return main_scheduler

    warmup_scheduler = LinearLR(
        self.optimizer,
        start_factor=(
            1.0
            / float(
                warmup_steps
            )
        ),
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return SequentialLR(
        self.optimizer,
        schedulers=[
            warmup_scheduler,
            main_scheduler,
        ],
        milestones=[
            warmup_steps
        ],
    )
```

---

## 2. 增加 PIM LR Ratio 测试

### 文件

```text
tests/zeva/test_trainer_scheduler.py
```

如果该文件不存在，新建。

---

### 2.1 新增测试文件

```python
import math

import torch

from fastwam.trainer import Wan22Trainer
```

新增 helper：

```python
def _make_scheduler(
    optimizer,
    *,
    scheduler_type="cosine",
    total_train_steps=100,
    warmup_steps=0,
):
    trainer = (
        Wan22Trainer
        .__new__(
            Wan22Trainer
        )
    )

    trainer.optimizer = optimizer
    trainer.learning_rate = 1e-4

    return (
        trainer
        ._build_scheduler(
            scheduler_type=
            scheduler_type,
            total_train_steps=
            total_train_steps,
            warmup_steps=
            warmup_steps,
        )
    )
```

---

### 2.2 增加 cosine ratio 测试

```python
def test_pim_lr_ratio_is_preserved_by_cosine():
    p_prompt = torch.nn.Parameter(
        torch.zeros(1)
    )

    p_projector = (
        torch.nn.Parameter(
            torch.zeros(1)
        )
    )

    p_gate = torch.nn.Parameter(
        torch.zeros(1)
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p_prompt],
                "lr": 5e-4,
            },
            {
                "params": [p_projector],
                "lr": 5e-4,
            },
            {
                "params": [p_gate],
                "lr": 1e-4,
            },
        ]
    )

    scheduler = _make_scheduler(
        optimizer,
        total_train_steps=100,
        warmup_steps=0,
    )

    checkpoints = {
        0,
        1,
        10,
        25,
        50,
        75,
        99,
        100,
    }

    for step in range(101):
        if step in checkpoints:
            lrs = [
                group["lr"]
                for group
                in optimizer.param_groups
            ]

            assert math.isclose(
                lrs[0] / lrs[2],
                5.0,
                rel_tol=1e-6,
                abs_tol=1e-8,
            )

            assert math.isclose(
                lrs[1] / lrs[2],
                5.0,
                rel_tol=1e-6,
                abs_tol=1e-8,
            )

        if step < 100:
            optimizer.step()
            scheduler.step()
```

---

### 2.3 增加最终 LR 测试

```python
def test_cosine_final_lr_is_one_percent_per_group():
    params = [
        torch.nn.Parameter(
            torch.zeros(1)
        )
        for _ in range(3)
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [params[0]],
                "lr": 5e-4,
            },
            {
                "params": [params[1]],
                "lr": 5e-4,
            },
            {
                "params": [params[2]],
                "lr": 1e-4,
            },
        ]
    )

    scheduler = _make_scheduler(
        optimizer,
        total_train_steps=100,
        warmup_steps=0,
    )

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    lrs = [
        group["lr"]
        for group
        in optimizer.param_groups
    ]

    assert math.isclose(
        lrs[0],
        5e-6,
        rel_tol=1e-5,
    )

    assert math.isclose(
        lrs[1],
        5e-6,
        rel_tol=1e-5,
    )

    assert math.isclose(
        lrs[2],
        1e-6,
        rel_tol=1e-5,
    )
```

---

## 3. 增加 Stage-2B Trainer-Level Resume 回归测试

### 文件

推荐：

```text
tests/zeva/test_trainer_resume.py
```

如果已有 trainer checkpoint 测试文件，则放入已有文件。

---

## 3.1 测试目标

完整验证：

```text
pim_adapter checkpoint
        ↓
resume=checkpoint_state_dir
        ↓
load_scope="all"
        ↓
恢复：
  CausalPromptEncoder
  prefix_project
  pim_gate
  optimizer
  scheduler
  global_step
  epoch
  batch_in_epoch

且：
  不调用 reset_pim_parameters()
```

---

## 3.2 新增最小 Fake Model

```python
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from fastwam.trainer import Wan22Trainer
```

```python
class _FakeExactAdapter(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.prior = nn.Linear(
            2,
            2,
        )

        self.action_prior_adapter = (
            nn.Linear(
                2,
                2,
            )
        )

        self.behavior_global_projector = (
            nn.Linear(
                2,
                2,
            )
        )

        self.prefix_project = nn.Linear(
            2,
            2,
        )

        self.pim_gate = nn.Parameter(
            torch.zeros(())
        )

        self.reset_calls = 0

    def policy_injection_parameters(
        self,
    ):
        yield from (
            self.prior.parameters()
        )

        yield from (
            self
            .action_prior_adapter
            .parameters()
        )

        yield from (
            self
            .behavior_global_projector
            .parameters()
        )

    def pim_adapter_parameters(
        self,
    ):
        yield from (
            self
            .prefix_project
            .parameters()
        )

        yield self.pim_gate

    def reset_pim_parameters(
        self,
    ):
        self.reset_calls += 1

        nn.init.zeros_(
            self
            .prefix_project
            .weight
        )

        nn.init.zeros_(
            self
            .prefix_project
            .bias
        )

        with torch.no_grad():
            self.pim_gate.zero_()
```

---

### 3.3 Fake FastWAM

```python
class _FakeZevaModel(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.zeva_enabled = True
        self.zeva_training_stage = (
            "pim_adapter"
        )

        self.base = nn.Linear(
            2,
            2,
        )

        self.zeva_prompt_encoder = (
            nn.Linear(
                2,
                2,
            )
        )

        self.zeva_behavior_prefix_adapter = (
            _FakeExactAdapter()
        )

    def configure_zeva_trainable_state(
        self,
    ):
        self.eval()
        self.requires_grad_(False)

        self.zeva_prompt_encoder.train()
        self.zeva_prompt_encoder.requires_grad_(
            True
        )

        adapter = (
            self
            .zeva_behavior_prefix_adapter
        )

        adapter.prefix_project.train()
        adapter.prefix_project.requires_grad_(
            True
        )

        adapter.pim_gate.requires_grad_(
            True
        )

    def zeva_trainable_parameters(
        self,
    ):
        yield from (
            self
            .zeva_prompt_encoder
            .parameters()
        )

        yield from (
            self
            .zeva_behavior_prefix_adapter
            .pim_adapter_parameters()
        )

    def zeva_parameter_report(
        self,
    ):
        trainable_names = [
            name
            for name, parameter
            in self.named_parameters()
            if parameter.requires_grad
        ]

        frozen_names = [
            name
            for name, parameter
            in self.named_parameters()
            if not parameter.requires_grad
        ]

        return {
            "training_stage":
            self.zeva_training_stage,

            "trainable_names":
            trainable_names,

            "frozen_names":
            frozen_names,

            "trainable_count":
            sum(
                parameter.numel()
                for parameter
                in self.parameters()
                if parameter.requires_grad
            ),

            "frozen_count":
            sum(
                parameter.numel()
                for parameter
                in self.parameters()
                if not parameter.requires_grad
            ),
        }

    def load_zeva_addon_checkpoint(
        self,
        path,
        *,
        base_checkpoint_sha256,
        cte_checkpoint_sha256,
        load_scope,
    ):
        assert load_scope == "all"

        payload = torch.load(
            path,
            map_location="cpu",
        )

        self
        .zeva_prompt_encoder
        .load_state_dict(
            payload[
                "causal_prompt_encoder"
            ]
        )

        self
        .zeva_behavior_prefix_adapter
        .load_state_dict(
            payload[
                "behavior_prefix_adapter"
            ]
        )

        return payload
```

---

## 3.4 Fake accelerator

```python
class _FakeAccelerator:
    def unwrap_model(
        self,
        model,
    ):
        return model
```

---

## 3.5 创建 checkpoint helper

```python
def _write_pim_resume_checkpoint(
    tmp_path: Path,
    model: _FakeZevaModel,
):
    checkpoint_root = (
        tmp_path
        / "checkpoints"
    )

    weights_dir = (
        checkpoint_root
        / "weights"
    )

    state_dir = (
        checkpoint_root
        / "state"
    )

    weights_dir.mkdir(
        parents=True
    )

    state_dir.mkdir(
        parents=True
    )

    step_name = "step_000123"

    addon_path = (
        weights_dir
        / f"{step_name}_addon.pt"
    )

    resume_dir = (
        state_dir
        / step_name
    )

    resume_dir.mkdir()

    torch.save(
        {
            "causal_prompt_encoder":
            model
            .zeva_prompt_encoder
            .state_dict(),

            "behavior_prefix_adapter":
            model
            .zeva_behavior_prefix_adapter
            .state_dict(),

            "training_stage":
            "pim_adapter",
        },
        addon_path,
    )

    return (
        addon_path,
        resume_dir,
    )
```

---

## 3.6 完整 Resume Test

```python
def test_pim_stage_resume_restores_all_state(
    tmp_path,
    monkeypatch,
):
    source = _FakeZevaModel()

    source.configure_zeva_trainable_state()

    with torch.no_grad():
        source
        .zeva_prompt_encoder
        .weight
        .fill_(1.25)

        source
        .zeva_prompt_encoder
        .bias
        .fill_(1.5)

        source
        .zeva_behavior_prefix_adapter
        .prefix_project
        .weight
        .fill_(2.25)

        source
        .zeva_behavior_prefix_adapter
        .prefix_project
        .bias
        .fill_(2.5)

        source
        .zeva_behavior_prefix_adapter
        .pim_gate
        .fill_(0.75)

    (
        addon_path,
        resume_dir,
    ) = _write_pim_resume_checkpoint(
        tmp_path,
        source,
    )

    target = _FakeZevaModel()
    target.configure_zeva_trainable_state()

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                list(
                    target
                    .zeva_prompt_encoder
                    .parameters()
                ),

                "lr":
                5e-4,
            },
            {
                "params":
                list(
                    target
                    .zeva_behavior_prefix_adapter
                    .prefix_project
                    .parameters()
                ),

                "lr":
                5e-4,
            },
            {
                "params":
                [
                    target
                    .zeva_behavior_prefix_adapter
                    .pim_gate
                ],

                "lr":
                1e-4,
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    # Create optimizer state.
    loss = sum(
        parameter.sum()
        for parameter
        in target.parameters()
        if parameter.requires_grad
    )

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(
        set_to_none=True
    )

    torch.save(
        {
            "optimizer":
            optimizer.state_dict(),

            "scheduler":
            scheduler.state_dict(),
        },
        resume_dir
        / "optimizer_scheduler.pt",
    )

    with (
        resume_dir
        / "trainer_state.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "global_step": 123,
                "epoch": 4,
                "batch_in_epoch": 17,
            },
            handle,
        )

    # trainer.py hashes base and CTE
    # checkpoints. Stub hashes in this
    # unit test.
    monkeypatch.setattr(
        "fastwam.trainer."
        "checkpoint_sha256",
        lambda _: "fake-sha",
    )

    trainer = (
        Wan22Trainer
        .__new__(
            Wan22Trainer
        )
    )

    trainer.model = target
    trainer.optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params":
                    list(
                        target
                        .zeva_prompt_encoder
                        .parameters()
                    ),
                    "lr":
                    5e-4,
                },
                {
                    "params":
                    list(
                        target
                        .zeva_behavior_prefix_adapter
                        .prefix_project
                        .parameters()
                    ),
                    "lr":
                    5e-4,
                },
                {
                    "params":
                    [
                        target
                        .zeva_behavior_prefix_adapter
                        .pim_gate
                    ],
                    "lr":
                    1e-4,
                },
            ]
        )
    )

    trainer.scheduler = (
        torch.optim.lr_scheduler
        .LambdaLR(
            trainer.optimizer,
            lr_lambda=lambda _: 1.0,
        )
    )

    trainer.accelerator = (
        _FakeAccelerator()
    )

    trainer.zeva_training = True

    trainer.zeva_training_stage = (
        "pim_adapter"
    )

    trainer.resume = str(
        resume_dir
    )

    trainer.global_step = 0
    trainer.epoch = 0
    trainer.batch_in_epoch = 0

    trainer.cfg = {
        "ckpt":
        str(
            tmp_path
            / "base.pt"
        ),

        "model": {
            "zeva": {
                "cte": {
                    "checkpoint":
                    str(
                        tmp_path
                        / "cte.pt"
                    )
                }
            }
        },
    }

    trainer._resume_or_load_checkpoint()

    # Stage-2B must not reset fresh PIM
    # parameters during resume.
    assert (
        target
        .zeva_behavior_prefix_adapter
        .reset_calls
        == 0
    )

    torch.testing.assert_close(
        target
        .zeva_prompt_encoder
        .weight,
        source
        .zeva_prompt_encoder
        .weight,
    )

    torch.testing.assert_close(
        target
        .zeva_prompt_encoder
        .bias,
        source
        .zeva_prompt_encoder
        .bias,
    )

    torch.testing.assert_close(
        target
        .zeva_behavior_prefix_adapter
        .prefix_project
        .weight,
        source
        .zeva_behavior_prefix_adapter
        .prefix_project
        .weight,
    )

    torch.testing.assert_close(
        target
        .zeva_behavior_prefix_adapter
        .prefix_project
        .bias,
        source
        .zeva_behavior_prefix_adapter
        .prefix_project
        .bias,
    )

    torch.testing.assert_close(
        target
        .zeva_behavior_prefix_adapter
        .pim_gate,
        source
        .zeva_behavior_prefix_adapter
        .pim_gate,
    )

    assert trainer.global_step == 123
    assert trainer.epoch == 4
    assert trainer.batch_in_epoch == 17
```

---

## 4. 增加 Stage-2A → Stage-2B 初始化测试

### 文件

```text
tests/zeva/test_trainer_resume.py
```

新增：

```python
def test_pim_stage_initialization_uses_policy_checkpoint_only(
    tmp_path,
    monkeypatch,
):
    source = _FakeZevaModel()

    with torch.no_grad():
        source
        .zeva_behavior_prefix_adapter
        .prior
        .weight
        .fill_(1.0)

        source
        .zeva_behavior_prefix_adapter
        .action_prior_adapter
        .weight
        .fill_(2.0)

        source
        .zeva_behavior_prefix_adapter
        .behavior_global_projector
        .weight
        .fill_(3.0)

    policy_path = (
        tmp_path
        / "stage2a.pt"
    )

    torch.save(
        {
            "causal_prompt_encoder":
            source
            .zeva_prompt_encoder
            .state_dict(),

            "behavior_prefix_adapter":
            source
            .zeva_behavior_prefix_adapter
            .state_dict(),

            "training_stage":
            "policy_injection",
        },
        policy_path,
    )

    target = _FakeZevaModel()

    loaded_scope = {}

    original_loader = (
        target
        .load_zeva_addon_checkpoint
    )

    def wrapped_loader(
        path,
        *,
        base_checkpoint_sha256,
        cte_checkpoint_sha256,
        load_scope,
    ):
        loaded_scope["value"] = (
            load_scope
        )

        return original_loader(
            path,
            base_checkpoint_sha256=
            base_checkpoint_sha256,
            cte_checkpoint_sha256=
            cte_checkpoint_sha256,
            load_scope=
            load_scope,
        )

    target.load_zeva_addon_checkpoint = (
        wrapped_loader
    )

    monkeypatch.setattr(
        "fastwam.trainer."
        "checkpoint_sha256",
        lambda _: "fake-sha",
    )

    trainer = (
        Wan22Trainer
        .__new__(
            Wan22Trainer
        )
    )

    trainer.model = target
    trainer.zeva_training = True
    trainer.zeva_training_stage = (
        "pim_adapter"
    )
    trainer.resume = None

    trainer.cfg = {
        "ckpt":
        str(
            tmp_path
            / "base.pt"
        ),

        "model": {
            "zeva": {
                "policy_checkpoint":
                str(
                    policy_path
                ),

                "cte": {
                    "checkpoint":
                    str(
                        tmp_path
                        / "cte.pt"
                    )
                },
            }
        },
    }

    trainer._initialize_pim_stage_from_policy_checkpoint()

    assert (
        loaded_scope["value"]
        == "policy_injection"
    )

    assert (
        target
        .zeva_behavior_prefix_adapter
        .reset_calls
        == 1
    )
```

---

## 5. 如果 Fake Loader 不支持 `load_scope="policy_injection"`

将 `_FakeZevaModel.load_zeva_addon_checkpoint()` 修改为：

```python
def load_zeva_addon_checkpoint(
    self,
    path,
    *,
    base_checkpoint_sha256,
    cte_checkpoint_sha256,
    load_scope,
):
    payload = torch.load(
        path,
        map_location="cpu",
    )

    if load_scope == "all":
        self
        .zeva_prompt_encoder
        .load_state_dict(
            payload[
                "causal_prompt_encoder"
            ]
        )

        self
        .zeva_behavior_prefix_adapter
        .load_state_dict(
            payload[
                "behavior_prefix_adapter"
            ]
        )

        return payload

    if load_scope == "policy_injection":
        source_state = payload[
            "behavior_prefix_adapter"
        ]

        target_state = (
            self
            .zeva_behavior_prefix_adapter
            .state_dict()
        )

        prefixes = (
            "prior.",
            "action_prior_adapter.",
            "behavior_global_projector.",
        )

        for name, value in (
            source_state.items()
        ):
            if name.startswith(
                prefixes
            ):
                target_state[name] = value

        self
        .zeva_behavior_prefix_adapter
        .load_state_dict(
            target_state
        )

        return payload

    raise ValueError(
        f"unsupported load_scope: "
        f"{load_scope}"
    )
```

---

## 6. 推荐增加 Scheduler 日志

### 文件

```text
src/fastwam/trainer.py
```

在 scheduler 创建后增加：

```python
if (
    self.zeva_training
    and self.zeva_training_stage
    == "pim_adapter"
):
    logger.info(
        "PIM optimizer initial LRs: %s",
        [
            float(
                group["lr"]
            )
            for group
            in self.optimizer.param_groups
        ],
    )
```

在训练第一次 scheduler step 后可选增加一次 debug：

```python
if (
    self.zeva_training
    and self.zeva_training_stage
    == "pim_adapter"
    and self.global_step == 1
):
    lrs = [
        float(group["lr"])
        for group
        in self.optimizer.param_groups
    ]

    logger.info(
        "PIM optimizer step-1 LRs: %s",
        lrs,
    )
```

---

## 7. 最终测试命令

```bash
pytest -q \
  tests/zeva/test_behavior_prefix_adapter.py \
  tests/zeva/test_fastwam_action_integration.py \
  tests/zeva/test_trainer_scheduler.py \
  tests/zeva/test_trainer_resume.py \
  tests/zeva/test_hydra_config.py
```

---

## 8. 最终必须满足

```text
[ ] prompt_encoder LR / pim_gate LR == 5
[ ] prefix_project LR / pim_gate LR == 5

[ ] 上述倍率在：
    warmup 后
    cosine 中段
    cosine 尾段
    始终保持

[ ] Stage-2B resume:
    load_scope == all

[ ] Stage-2B resume:
    reset_pim_parameters 调用次数 == 0

[ ] Stage-2B resume:
    CausalPromptEncoder 恢复

[ ] Stage-2B resume:
    prefix_project 恢复

[ ] Stage-2B resume:
    pim_gate 恢复

[ ] Stage-2B resume:
    optimizer 恢复

[ ] Stage-2B resume:
    scheduler 恢复

[ ] Stage-2B resume:
    global_step / epoch / batch_in_epoch 恢复

[ ] Stage-2A -> Stage-2B 初始化:
    load_scope == policy_injection

[ ] Stage-2A -> Stage-2B 初始化:
    reset_pim_parameters 调用次数 == 1
```
