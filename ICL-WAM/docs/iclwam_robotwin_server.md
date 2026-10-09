# ICL-WAM RoboTwin HTTP 推理

入口：`experiments/robotwin/iclwam_server.py`。服务复用本仓库
`WorldActionRobotWinPolicy` 的模型加载、三相机拼图、归一化、静态 task-context
检索和 CTE/BIT/PIM 生命周期。

## 启动

在已安装 ICL-WAM 模型依赖的环境中安装 HTTP 依赖：

```bash
python -m pip install -r experiments/robotwin/iclwam_policy/requirements.txt
```

以下命令使用本机已有的 Stage-2B 检查点：

```bash
cd /home/ubuntu/sim_robotwin_iclwam/ICL-WAM
python experiments/robotwin/iclwam_server.py \
  --checkpoint ../CKPT/fastwam_ckpt.pt \
  --dataset_stats ../CKPT/robotw_stats.json \
  --cte_checkpoint ../CKPT/cte.pt \
  --addon_checkpoint ../CKPT/step_005000_addon.pt \
  --task_context_bank ../CKPT/behavior_bank.pt \
  --task_context_retrieval_checkpoint ../CKPT/static_retrieval_head.pt \
  --model_base_path /home/ubuntu/zhangjj/FastWAM/checkpoints/ \
  --mode pim_on --device cuda --vae_device_mode cpu \
  --host 0.0.0.0 --port 8765
```

`--model_base_path` 是包含 `Wan-AI/` 的目录，用于定位 Wan VAE、T5 和 tokenizer。
默认禁止下载；需要联网获取缺失资源时可用 `--allow_download`。
默认配置 `robotwin_zeva_fastwam_pim_3cam_384` 对应 Stage-2B；Stage-2A 使用
`--sim_task robotwin_zeva_fastwam_policy_3cam_384 --mode zeva_stage2`。
可重复传 `--override key=value` 修改 Hydra 配置。检查点、bank 和 retrieval head
须来自匹配的训练流程，原有 SHA256 和 task-context 一致性检查仍生效。
`pooling` 仅适用于该模式训练的 addon，不能替代缺失的 static 文件。

预测 horizon 固定为 32，默认队列执行 24 步后重规划；`--replan_steps` 在 Zeva
模式下须为 4 的倍数，最大 32。`--num_inference_steps` 默认 10。
`--vae_device_mode cpu` 同时覆盖主模型及 CTE 的 VAE 编码，主模型仍在 GPU 上；
也支持 `--vae_device_mode gpu`。

## RoboTwin 客户端

仿真环境只需 `requests`、`json-numpy`、`numpy`，无需加载模型。将配套 policy
链接到实际使用的 RoboTwin 根目录；以下以本仓库的 RoboTwin 为例：

```bash
cd /home/ubuntu/sim_robotwin_iclwam/ICL-WAM/third_party/RoboTwin
ln -s ../../../experiments/robotwin/iclwam_policy policy/iclwam_policy
python script/eval_policy.py \
  --config policy/iclwam_policy/deploy_policy.yml \
  --overrides \
  task_name YOUR_TASK task_config demo_randomized \
  ckpt_setting iclwam_http policy_name iclwam_policy \
  server_url http://127.0.0.1:8765 \
  fixed_seed 0 max_attempts 4 eval_num_episodes 1 \
  skip_get_obs_within_replan false
```

已有同名链接时跳过 `ln -s`。此 HTTP policy 直接使用 RoboTwin evaluator，
不通过本地模型入口 `eval_robotwin_single.py` 启动。
本仓库 evaluator 会调用 `begin_attempt(attempt_id)` 和
`finalize_attempt(env, success)`；外部 RoboTwin 版本也需要这两个回调才能保留
跨尝试 PIM。只有 `reset_model` / `eval` 时可以进行单次 rollout，无法正确完成重试。
重试须保持同一环境 seed 和指令，不同 episode 即使指令相同也必须 reset。
一个服务进程对应一个仿真环境；并行环境使用独立进程和端口。
客户端对推理服务的 HTTP 请求不使用系统代理，避免本机流量经过 Clash。

### 使用独立的 ICL-WAM client

原始 FastWAM client 和批量评测脚本保持不变。ICL-WAM 使用并列的新文件
`/home/ubuntu/robotwin_realted/client/client_skip_unstable_seed_iclwam.py` 和
`eval_all_robotwin_zjj_iclwam.sh`。ICL client 每个仿真步请求一条动作，每个 rollout
开始时 `/reset`，正常成功或失败结束时 `/finalize`。
进度显示仍按 24 个环境步分组，因此会看到每组一个 `24/24` 进度条（最后一组按剩余步数显示），
组内 HTTP 仍逐步请求并执行动作。

终端一，启动 ICL-WAM 服务：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fastwam
cd /home/ubuntu/sim_robotwin_iclwam/ICL-WAM
python experiments/robotwin/iclwam_server.py \
  --checkpoint ../CKPT/fastwam_ckpt.pt \
  --dataset_stats ../CKPT/robotw_stats.json \
  --cte_checkpoint ../CKPT/cte.pt \
  --addon_checkpoint ../CKPT/step_005000_addon.pt \
  --task_context_bank ../CKPT/behavior_bank.pt \
  --task_context_retrieval_checkpoint ../CKPT/static_retrieval_head.pt \
  --model_base_path /home/ubuntu/zhangjj/FastWAM/checkpoints \
  --mode pim_on --device cuda --vae_device_mode cpu \
  --host 127.0.0.1 --port 8765
```

等日志出现 `ICL-WAM ready`。另开终端二，在 RoboTwin 环境测试一个任务、一个 seed：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /home/ubuntu/robotwin_realted/client
NUM_EPISODES=1 \
SEED=0 \
TASK_CONFIG=demo_randomized \
EVAL_LOG_DIR=/home/ubuntu/robotwin_realted/log_iclwam_smoke \
SLEEP_SECONDS=0 \
ROBOTWIN_STEP_TIMING=0 \
bash eval_all_robotwin_zjj_iclwam.sh
```

当前批量脚本中的任务是 `put_object_cabinet`。它从 seed `2000` 开始（由 `SEED=0`
按脚本原有规则换算），先做 expert seed 检查，再运行策略；查看该日志目录下的任务
日志、`results.json` 和生成的视频确认 rollout 结果。第一次可把 `NUM_EPISODES` 留为 1。

### 固定 seed 的多次重试实验

要验证同一个失败 seed 的跨尝试记忆，不要重复启动服务。使用新增驱动脚本：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /home/ubuntu/robotwin_realted/client
~/anaconda3/envs/RoboTwin/bin/python run_iclwam_fixed_seed_retry.py \
  --task put_object_cabinet \
  --task-config demo_randomized \
  --sim-seed 2002 \
  --attempts 4 \
  --server-url http://127.0.0.1:8765 \
  --output-dir /home/ubuntu/robotwin_realted/log_iclwam_retry
```

脚本通过 `--fixed_sim_seed` 直接使用实际仿真 seed `2002`，最多运行
4 次同一任务、同一 seed、同一指令。第一次失败后保留服务进程，下一次通过连续的
`/reset(attempt_id, session_id)` 复用 PIM；根据 `results.json` 中的任务结果判断成功，
一旦成功就停止后续尝试。客户端正常退出不代表任务成功。每次 attempt 独立保存
`client.log`、RoboTwin 结果和视频；总配置、命令、退出码、耗时、服务健康状态保存在
`<output-dir>/<task>_seed_<seed>_<时间戳>/config.json`。如果第一次已经成功，实验会直接
结束，不能据此评价 retry 提升；应选择一次确认失败的 seed。

要记录 PIM 实际检索结果，在**启动服务**的命令中增加
`--pim-trace /home/ubuntu/robotwin_realted/log_iclwam_retry/pim_retrieval.jsonl`，
然后运行上述固定 seed 重试脚本。每次重新规划时追加一行 JSON，包含 `session_id`、
`attempt_id`、`step_id`、`pim_entry_count`、查询 phase 向量，以及 `matches` 中每条
检索结果的相似度 `score`、来源 `source` 和 phase/effect 向量。查看
`source.first_attempt_id`、`source.last_attempt_id` 和 `source.observation_count`
可判断条目是否来自之前的尝试，以及后来是否被合并更新；`matches: []` 表示当次
没有检索到可用条目。
phase/effect 是模型潜变量，不能直接读成自然语言或图像。文件采用追加方式，
多轮实验可用 `session_id` 区分。已完成的运行无法补录，修改后需重启服务进程。
`--attempts` 不能超过服务启动时的 `--max_attempts`（默认 4）；要运行 10 次，
先以 `--max_attempts 10` 重启服务。若 expert 预检查跳过 seed，策略尚未运行，
不会生成 `results.json`，也不能把这次跳过计作一次 PIM 失败尝试。

要测基线，先停止 ICL-WAM 服务，再用相同命令以 `--mode base` 重启（端口仍用 8765），
然后仍用这个 ICL-WAM client 和同一个 seed，换一个 `EVAL_LOG_DIR` 再跑一次。
客户端保持单步 HTTP 传输，只是服务端关闭 Zeva 条件；
这样不需要同时在一张 GPU 上加载两份大模型。正式比较时应扩大 episodes，并比较成功率，
而非只看单次成功与否。

这个现有脚本每个 seed 只跑一次，不会在同一个 seed 失败后重试；因此可以比较本次
rollout 的成功率，但不能测试跨 retry 保留 PIM 的收益。若要验证 retry memory，需使用
会在固定 seed 上调用多次 `begin_attempt` 的 evaluator。

## HTTP 协议

- `GET /health`：模式、重规划长度、每次返回动作数和当前状态。
- `POST /reset`：`{"attempt_id":0}` 开始新 episode，返回 `session_id`。
  重试时传已 finalize 的 `session_id` 和连续递增的 `attempt_id`，保留本 episode
  的 PIM，清除队列和 CTE/BIT。默认最多 4 次尝试，可用 `--max_attempts` 调整。
- `POST /act`：保留 FastWAM 的 `image0`、`image1`、`image2`、`proprio`、
  `language_instruction` 字段，依次为 head/left-wrist/right-wrist RGB 和 14 维 qpos。
  数组接受 `json_numpy.dumps` 字符串或 JSON 列表，图像为 RGB 整数 `[0,255]`。
  推荐带 `session_id` 和从 0 开始的 `step_id`。返回
  `{"status":"success","action":[[...14 floats...]],"session_id":"...","step_id":0}`。
- `POST /finalize`：传当前 `session_id`、已执行动作数 `step_id`、布尔 `success`，
  并尽可能携带上述四个数组组成的真实终止观测，以补齐最后一个完整的四动作
  transition。缺少可靠 after-frame 或不足四步的尾部会丢弃，不虚构填充。

**与原 FastWAM server 的差异：每次 `action` 是 `[1,14]`，不是整段预测。**
模型依旧按 chunk 推理，动作队列保留在服务端。必须先执行返回动作，再把真实
执行后观测发给下一次 `/act`。旧客户端不能跳过、重复动作或重复使用首帧。
首次 `/act` 可省略 `/reset`，便于单次旧协议接入；切换 episode 必须显式 reset。

同一 session 最近一次相同 `step_id` 的 `/act` 重发返回缓存动作，不推进记忆。
配套客户端遇到连接错误或超时时重发一次相同请求。`/reset` 不自动重试；若响应
丢失，重新开始 episode。仿真执行异常时应放弃本次尝试、重建环境，并从
`attempt_id=0` 开始，不把未执行动作提交为有效 transition。
参数错误返回 400、状态冲突返回 409、推理异常返回 500；异常不会返回零动作成功响应。

## 测试

```bash
python -m pytest tests/robotwin/test_iclwam_server.py
```

协议与生命周期测试使用小型 CTE 和假动作模型，不需要 GPU 或大模型权重。
