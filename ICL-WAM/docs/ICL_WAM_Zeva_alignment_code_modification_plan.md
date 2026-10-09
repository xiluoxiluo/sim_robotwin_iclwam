# ICL-WAM Zeva Alignment Code Modification Plan

## 修改目标

保持 Zeva 核心机制，仅修复评测语义和 residual injection 细节。

保留： - CTE - BIT/PIM - Phase Retrieval - Effect Retrieval - Static
Task Context - Causal Prompt - Behavior Prefix - Gaussian Action Prior -
Frozen FastWAM backbone

禁止： - 修改 FastWAM 主干 - 增加 memory expert - 引入 FastWAM-Joint -
引入 IDM

------------------------------------------------------------------------

# 1. 修复 pim_shadow

## 问题

当前 pim_shadow 可能仍执行：

    retrieval
    +
    behavior prefix injection
    +
    action prior residual

导致：

    base != pim_shadow

不符合 Zeva shadow baseline。

------------------------------------------------------------------------

# 2. 增加 Runtime Flag

新增：

``` python
@dataclass
class ZevaRuntimeConfig:
    enable_memory: bool = False
    enable_retrieval: bool = False
    enable_prefix_injection: bool = False
    enable_action_prior: bool = False
```

模式：

## base

``` python
enable_memory=False
enable_retrieval=False
enable_prefix_injection=False
enable_action_prior=False
```

## pim_shadow

``` python
enable_memory=True
enable_retrieval=True
enable_prefix_injection=False
enable_action_prior=False
```

## pim_on

``` python
enable_memory=True
enable_retrieval=True
enable_prefix_injection=True
enable_action_prior=True
```

------------------------------------------------------------------------

# 3. eval脚本修改

文件：

    scripts/eval_zeva_robotwin_fixed_attempts.py

修改：

``` python
if mode == "pim_on":
    require addon_checkpoint

elif mode == "pim_shadow":
    addon_checkpoint = None

elif mode == "base":
    addon_checkpoint = None
```

------------------------------------------------------------------------

# 4. deploy_policy 修改

文件：

    experiments/robotwin/fastwam_policy/deploy_policy.py

修改：

``` python
if mode == "pim_on":
    load_zeva_addon()

elif mode == "pim_shadow":
    initialize_memory_only()

elif mode == "base":
    disable_zeva()
```

禁止：

``` python
if mode != "base":
    load_addon()
```

------------------------------------------------------------------------

# 5. FastWAM injection修改

文件：

    src/fastwam/models/wan22/fastwam.py

不要：

``` python
if mode != base:
    inject()
```

改：

``` python
if enable_prefix_injection:
    action_hidden += prefix_residual

if enable_action_prior:
    action_hidden += action_prior_residual
```

确保：

pim_shadow运行memory，但不改变FastWAM输出。

------------------------------------------------------------------------

# 6. Action Prior增加独立gate

文件：

    src/fastwam/zeva/behavior_prefix_adapter.py

增加：

``` python
self.action_prior_gate = nn.Parameter(
    torch.tensor(0.0)
)
```

修改：

``` python
def action_prior_residual(self, prior):

    residual = self.action_prior_adapter(prior)

    return torch.tanh(
        self.action_prior_gate
    ) * residual
```

最终：

$$
h'=h+g_pP+g_aA
$$

------------------------------------------------------------------------

# 7. 增加测试

新增：

    tests/test_pim_shadow.py

验证：

## base == pim_shadow

``` python
assert torch.allclose(
    action_base,
    action_shadow,
    atol=1e-6
)
```

## pim_on改变输出

``` python
assert not torch.allclose(
    action_base,
    action_on
)
```

## FastWAM冻结

``` python
assert all(
    not p.requires_grad
    for p in fastwam.parameters()
)
```

------------------------------------------------------------------------

# 最终结构

    History trajectory

            |
            v

           CTE

            |
        BIT / PIM

            |
        Retrieval


    ================


    base:

    FastWAM only


    pim_shadow:

    FastWAM
    +
    memory lifecycle
    (no injection)


    pim_on:

    FastWAM
    +
    Behavior Prefix Residual
    +
    Action Prior Residual

------------------------------------------------------------------------

# 论文描述

"We preserve Zeva's causal memory framework and replace only the frozen
Cosmos policy with frozen FastWAM. Shadow mode performs memory lifecycle
updates without policy conditioning, while online mode enables residual
memory conditioning."
