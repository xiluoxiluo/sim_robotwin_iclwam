# ICL-WAM Final Code Modification — Resume Boundary Fix

## 1. 修复 Stage-2B Resume 时错误检查旧 `policy_checkpoint`

### 文件

```text
src/fastwam/trainer.py
```

### 函数

```python
Wan22Trainer._initialize_pim_stage_from_policy_checkpoint()
```

---

## 1.1 当前问题

当前代码顺序：

```python
def _initialize_pim_stage_from_policy_checkpoint(self) -> None:
    policy_checkpoint = self._get_zeva_policy_checkpoint()

    if self.resume not in (None, "", "None", "null"):
        return
```

问题：

```text
Stage-2B resume
    ↓
仍先执行 _get_zeva_policy_checkpoint()
    ↓
检查旧 Stage-2A checkpoint 是否存在
    ↓
旧机器路径失效时提前 FileNotFoundError
    ↓
真正的 Stage-2B resume 根本没有机会执行
```

例如：

```yaml
model:
  zeva:
    training_stage: pim_adapter
    policy_checkpoint: /old_machine/checkpoints/stage2A.pt

resume: /new_machine/run/checkpoints/state/step_010000
```

即使：

```text
resume checkpoint 完整有效
```

但：

```text
/old_machine/checkpoints/stage2A.pt
```

不存在时，当前代码仍会直接报错。

---

## 1.2 正确修改

将：

```python
def _initialize_pim_stage_from_policy_checkpoint(self) -> None:
    policy_checkpoint = self._get_zeva_policy_checkpoint()

    if self.resume not in (None, "", "None", "null"):
        return
```

修改为：

```python
def _initialize_pim_stage_from_policy_checkpoint(self) -> None:
    # A real pim_adapter resume restores the complete
    # Stage-2B state and must not depend on the old
    # Stage-2A initialization checkpoint.
    if self.resume not in (None, "", "None", "null"):
        return

    policy_checkpoint = self._get_zeva_policy_checkpoint()
```

---

## 1.3 推荐直接替换完整函数

```python
def _initialize_pim_stage_from_policy_checkpoint(
    self,
) -> None:
    # `resume` means restoring an already-running
    # training stage.
    #
    # For pim_adapter resume:
    #
    #   Stage-2B checkpoint
    #       ↓
    #   restore addon + optimizer + scheduler + step
    #
    # It must NOT inspect or require the original
    # Stage-2A initialization checkpoint.
    if self.resume not in (
        None,
        "",
        "None",
        "null",
    ):
        return

    # Reaching here means this is the FIRST
    # initialization of Stage-2B from Stage-2A.
    policy_checkpoint = (
        self._get_zeva_policy_checkpoint()
    )

    if policy_checkpoint is None:
        raise ValueError(
            "pim_adapter training requires "
            "model.zeva.policy_checkpoint "
            "when resume is not set"
        )

    base_path = self.cfg.get(
        "ckpt"
    )

    zeva_cfg = self.cfg.model.get(
        "zeva",
        {},
    )

    cte_path = (
        zeva_cfg
        .get(
            "cte",
            {},
        )
        .get(
            "checkpoint"
        )
        if hasattr(
            zeva_cfg,
            "get",
        )
        else None
    )

    if base_path in (
        None,
        "",
        "None",
        "null",
    ):
        raise ValueError(
            "pim_adapter initialization "
            "requires cfg.ckpt"
        )

    if cte_path in (
        None,
        "",
        "None",
        "null",
    ):
        raise ValueError(
            "pim_adapter initialization "
            "requires "
            "model.zeva.cte.checkpoint"
        )

    model = self.model

    payload = (
        model
        .load_zeva_addon_checkpoint(
            str(
                policy_checkpoint
            ),
            base_checkpoint_sha256=
            checkpoint_sha256(
                str(
                    base_path
                )
            ),
            cte_checkpoint_sha256=
            checkpoint_sha256(
                str(
                    cte_path
                )
            ),
            load_scope=
            "policy_injection",
        )
    )

    checkpoint_stage = (
        payload.get(
            "training_stage"
        )
    )

    if checkpoint_stage != (
        "policy_injection"
    ):
        raise ValueError(
            "model.zeva.policy_checkpoint "
            "must be a policy_injection "
            "checkpoint, got "
            f"{checkpoint_stage}"
        )

    (
        model
        .zeva_behavior_prefix_adapter
        .reset_pim_parameters()
    )

    model.configure_zeva_trainable_state()

    logger.info(
        "Initialized pim_adapter "
        "from policy checkpoint: %s",
        policy_checkpoint,
    )
```

---

# 2. 增加 Resume 不依赖旧 Stage-2A Checkpoint 的回归测试

### 文件

```text
tests/zeva/test_trainer_resume.py
```

新增测试：

```python
def test_pim_resume_does_not_require_policy_checkpoint(
    tmp_path,
):
    trainer = (
        Wan22Trainer
        .__new__(
            Wan22Trainer
        )
    )

    trainer.zeva_training = True

    trainer.zeva_training_stage = (
        "pim_adapter"
    )

    # The existence of the resume path is
    # irrelevant to this helper itself.
    # Only the fact that resume is set matters.
    trainer.resume = str(
        tmp_path
        / "resume_state"
    )

    trainer.cfg = OmegaConf.create(
        {
            "model": {
                "zeva": {
                    # Intentionally invalid.
                    # A Stage-2B resume must not
                    # inspect this path.
                    "policy_checkpoint":
                    (
                        "/definitely/"
                        "not/exist/"
                        "stage2A.pt"
                    )
                }
            }
        }
    )

    # Must return immediately.
    #
    # Before the fix this would call:
    #
    # _get_zeva_policy_checkpoint()
    #
    # and raise FileNotFoundError.
    trainer._initialize_pim_stage_from_policy_checkpoint()
```

---

# 3. 增加更严格的调用级测试

推荐同时增加一个 monkeypatch 测试，保证 `resume != null` 时：

```text
_get_zeva_policy_checkpoint()
```

根本不会被调用。

```python
def test_pim_resume_skips_policy_checkpoint_lookup(
    monkeypatch,
):
    trainer = (
        Wan22Trainer
        .__new__(
            Wan22Trainer
        )
    )

    trainer.zeva_training = True

    trainer.zeva_training_stage = (
        "pim_adapter"
    )

    trainer.resume = (
        "/tmp/"
        "stage2b_resume"
    )

    called = {
        "policy_lookup": 0,
    }

    def fail_policy_lookup():
        called[
            "policy_lookup"
        ] += 1

        raise AssertionError(
            "policy checkpoint lookup "
            "must not run during "
            "pim_adapter resume"
        )

    monkeypatch.setattr(
        trainer,
        "_get_zeva_policy_checkpoint",
        fail_policy_lookup,
    )

    trainer._initialize_pim_stage_from_policy_checkpoint()

    assert (
        called[
            "policy_lookup"
        ]
        == 0
    )
```

---

# 4. 保留原 Stage-2A → Stage-2B 初始化测试

已有：

```python
test_pim_stage_initialization_uses_policy_checkpoint_only
```

继续保留。

应满足：

```text
resume == null
    ↓
读取 model.zeva.policy_checkpoint
    ↓
load_scope == "policy_injection"
    ↓
恢复：
  prior
  action_prior_adapter
  behavior_global_projector
    ↓
reset_pim_parameters() == 1
```

---

# 5. 保留原 Stage-2B Resume 测试

已有：

```python
test_pim_stage_resume_restores_all_state
```

继续保留。

应满足：

```text
resume != null
    ↓
不读取 policy_checkpoint
    ↓
load_scope == "all"
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

reset_pim_parameters() == 0
```

---

# 6. 最终语义

修改后必须严格满足：

```text
CASE A — 第一次训练 Stage-2B

training_stage = pim_adapter
resume = null
policy_checkpoint = Stage-2A checkpoint

流程：

Stage-2A checkpoint
    ↓
load_scope = policy_injection
    ↓
恢复 Stage-2A policy modules
    ↓
fresh CausalPrompt / PIM projector / gate
    ↓
reset_pim_parameters()
    ↓
训练 Stage-2B
```

以及：

```text
CASE B — Stage-2B 断点续训

training_stage = pim_adapter
resume = Stage-2B state directory

policy_checkpoint:
  可以存在
  可以不存在
  可以是旧机器失效路径

流程：

resume
    ↓
直接跳过 Stage-2A initialization path
    ↓
load Stage-2B addon with load_scope = all
    ↓
restore optimizer
    ↓
restore scheduler
    ↓
restore global_step / epoch / batch_in_epoch
```

---

# 7. 最终测试命令

```bash
pytest -q \
  tests/zeva/test_trainer_resume.py \
  tests/zeva/test_trainer_scheduler.py \
  tests/zeva/test_fastwam_action_integration.py \
  tests/zeva/test_behavior_prefix_adapter.py \
  tests/zeva/test_hydra_config.py
```

---

# 8. 最终检查项

```text
[ ] resume != null 时，
    _get_zeva_policy_checkpoint()
    调用次数 == 0

[ ] resume != null 时，
    旧 policy_checkpoint 不存在
    也不会报错

[ ] resume == null 时，
    pim_adapter 必须要求
    policy_checkpoint

[ ] Stage-2A → Stage-2B：
    load_scope == policy_injection

[ ] Stage-2A → Stage-2B：
    reset_pim_parameters() == 1

[ ] Stage-2B resume：
    load_scope == all

[ ] Stage-2B resume：
    reset_pim_parameters() == 0

[ ] Stage-2B resume：
    optimizer / scheduler /
    global_step / epoch /
    batch_in_epoch 全部恢复

[ ] PIM LR ratio 始终保持：
    5 : 5 : 1
```
