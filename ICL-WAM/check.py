from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(
    "/data/share/1919650160032350208/"
    "foundation_model/datasets/robotwin2.0"
)

# 读取 task_index -> instruction
task_file = ROOT / "meta" / "tasks.parquet"

if task_file.exists():
    tasks = pd.read_parquet(task_file)
else:
    task_file = ROOT / "meta" / "tasks.jsonl"
    tasks = pd.read_json(task_file, lines=True)

print("tasks columns:", list(tasks.columns))
print("total task strings:", len(tasks))

task_map = {
    int(row.task_index): str(row.task)
    for row in tasks[["task_index", "task"]].itertuples(index=False)
}

# 专门检查可能的任务边界
episode_ids = [
    0, 1, 48, 49, 50, 51,
    548, 549, 550, 551,
    999, 1000,
    2498, 2499, 2500, 2501,
    2999, 3000,
]

for ep in episode_ids:
    matches = list(
        (ROOT / "data").rglob(
            f"episode_{ep:06d}.parquet"
        )
    )

    if not matches:
        print(f"\nepisode {ep}: FILE NOT FOUND")
        continue

    path = matches[0]

    table = pq.read_table(
        path,
        columns=["task_index"],
    )

    raw_indices = sorted({
        int(x)
        for x in table.column("task_index").to_pylist()
    })

    print("\n" + "=" * 80)
    print(f"episode {ep}")
    print(f"path: {path}")
    print(f"num task_index: {len(raw_indices)}")

    for idx in raw_indices[:8]:
        print(
            f"  {idx}: "
            f"{task_map.get(idx, '<missing>')}"
        )