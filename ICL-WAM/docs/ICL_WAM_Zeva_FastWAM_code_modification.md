# ICL-WAM: Zeva → FastWAM 代码级修改方案

## 目标

保持 Zeva 核心机制：

-   CTE
-   BIT
-   PIM
-   Phase Retrieval
-   Static Task Context
-   Causal Prompt
-   Policy Injection

仅替换 Frozen Cosmos Policy 为 Frozen FastWAM。

FastWAM 不作为新的 memory expert。

------------------------------------------------------------------------

# P0: 修正 Stage2 torch.no_grad 范围

## 文件

src/fastwam/trainer.py\
src/fastwam/models/\*

## 原则

冻结参数：

``` python
for p in fastwam.parameters():
    p.requires_grad = False
```

但是不能阻断 adapter 梯度。

错误：

``` python
with torch.no_grad():
    video_tokens = fastwam.video_encoder(x)
```

修改：

``` python
video_tokens = fastwam.video_encoder(
    x,
    behavior_prefix
)
```

只允许纯 backbone feature extraction 使用 no_grad。

------------------------------------------------------------------------

# P1: 增加 PIM dropout

文件：

src/fastwam/zeva/memory.py

增加：

``` yaml
pim_context_dropout: 0.2
pim_support_dropout: 0.2
```

训练阶段随机 dropout retrieval context/support。

Inference 不启用。

------------------------------------------------------------------------

# P1: Behavior Prefix 注入检查

文件：

src/fastwam/zeva/behavior_prefix_adapter.py

保持：

Causal Prompt → Prefix Projector → FastWAM Action Expert Cross Attention

不要修改成：

``` python
torch.cat([action_tokens, memory_tokens])
```

避免破坏 FastWAM ActionDiT token 分布。

------------------------------------------------------------------------

# P1: Action Prior 保持 residual

文件：

src/fastwam/zeva/policy_prior.py

保持：

``` python
action_hidden += gate * action_prior
```

不要改为 prior token concat。

Gate zero initialization：

``` python
nn.init.zeros_(gate.weight)
nn.init.zeros_(gate.bias)
```

保证：

ICL-WAM 初始化等价 FastWAM。

------------------------------------------------------------------------

# P2: Static Task Context

文件：

src/fastwam/zeva/static_task_context.py

保持：

initial observation + instruction

不要简化为 instruction embedding only。

视觉输入替换：

Cosmos feature → FastWAM first-frame representation。

------------------------------------------------------------------------

# P2: CTE action dimension

文件：

src/fastwam/zeva/causal_transition_encoder.py

只修改：

``` python
action_dim
```

Cosmos action dim:

8

FastWAM:

14

不要修改：

-   phase token
-   effect_pre
-   effect_post
-   causal history

------------------------------------------------------------------------

# P2: 生命周期检查

文件：

src/fastwam/zeva/lifecycle.py

保持：

action execute → observation update → CTE transition → effect complete →
PIM update

禁止提前写入 PIM。

Retry:

BIT clear, PIM keep

New episode:

BIT clear, PIM clear

------------------------------------------------------------------------

# P2: 增加 Freeze Test

新增：

tests/test_freeze_fastwam.py

检查：

``` python
for p in fastwam.parameters():
    assert not p.requires_grad
```

检查：

``` python
behavior_projector.grad != None
action_prior_projector.grad != None
gate.grad != None
```

------------------------------------------------------------------------

# 不修改模块

保持：

src/fastwam/zeva/

-   causal_transition_encoder.py
-   cte_losses.py
-   memory.py
-   retrieval.py
-   causal_prompt.py

------------------------------------------------------------------------

# 禁止修改

不要增加：

-   Memory Expert
-   FastWAM-Joint
-   IDM

不要修改：

-   FastWAM MoT routing

------------------------------------------------------------------------

# 最终结构

Frozen FastWAM

Video Expert \| Action Expert \| Action

↑

Behavior Prefix + Action Prior

↑

Zeva Memory:

CTE BIT PIM Phase Retrieval Causal Prompt

------------------------------------------------------------------------

# 修改优先级

1.  修正 torch.no_grad 范围
2.  增加 PIM dropout
3.  增加 freeze/gradient test
4.  检查 prefix cross attention
5.  检查 action prior residual
6.  action_dim 适配
