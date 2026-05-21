# Matching Module

这个目录下包含了三种不同的匹配 (Reranking) 策略实现。每种脚本都接受相同的输入输出格式，但内部对文档的打分和排序逻辑不同。

## 1. 脚本说明

### `matching.py` (Baseline: Aggregated)
- **核心逻辑**: **综合聚合 (Weighted Aggregation)**
- **算法**:
    1. 计算每个文档针对每个 Subquery 的匹配得分 (Score_i)。
    2. 对该文档的所有 Subquery 得分进行聚： `FinalScore = 0.5 * Max(Score_i) + 0.5 * Mean(Score_i)`。
- **特点**: 平衡了“单点突破”（完美匹配某一个子查询）和“整体覆盖”（覆盖多个子查询）的能力。这是默认的 Baseline 策略。

### `matching_round_robin.py` (Round-Robin)
- **核心逻辑**: **轮询选择 (Round-Robin)**
- **算法**:
    1. 将 Query 分解为 N 个 Subquery。
    2. 对每个 Subquery，分别找出得分最高的 Top K 文档，形成 N 个有序列表。
    3. 轮循这 N 个列表，依次取出 Rank 1, Rank 2... 的文档加入最终结果列表。
    4. 去重（如果一个文档已经被选入，则跳过）。
- **特点**: 强制多样性。确保每个子查询“最喜欢”的文档都能排在前面，防止某个子查询（如 Subquery 2）的主导地位压制了另一个子查询（如 Subquery 1）。

### `matching_max_score.py` (Max-Score)
- **核心逻辑**: **最大子查询得分 (Max-Subquery-Score)**
- **算法**:
    1. 计算每个文档针对每个 Subquery 的匹配得分。
    2. 文档的最终得分为其在**任意一个**子查询上获得的**最高分**：`FinalScore = Max(Score_i)`。
- **特点**: 鼓励“偏科”。只要文档能完美回答其中某一个子查询（即使与其他子查询完全无关），就能获得极高排名。适合“拼图式”检索（需要分别找两篇文档来拼凑答案）。

## 2. 使用方法 (Usage)

三个脚本的命令行参数完全一致。

```bash
# 激活环境
conda activate vllm
# 启动 Embedding 服务 (如果尚未启动)
# bash ../../utils/serve_embedding.sh

# 运行 Matching (以 Baseline 为例)
python -m absQA.matching.matching \
    --query_file ../results/bamboogle/query_typed_result.jsonl \
    --doc_file ../results/bamboogle/converted_triple_result.jsonl \
    --output_file ../results/bamboogle/matching_results/matching_baseline.jsonl

# 运行 Round-Robin
python -m absQA.matching.matching_round_robin \
    --query_file ../results/bamboogle/query_typed_result.jsonl \
    --doc_file ../results/bamboogle/converted_triple_result.jsonl \
    --output_file ../results/bamboogle/matching_results/matching_rr.jsonl

# 运行 Max-Score
python -m absQA.matching.matching_max_score \
    --query_file ../results/bamboogle/query_typed_result.jsonl \
    --doc_file ../results/bamboogle/converted_triple_result.jsonl \
    --output_file ../results/bamboogle/matching_results/matching_max.jsonl
```

### 参数详解
- `--query_file`: 输入的 Query 文件，需包含 Typed Triples (Step 5 Output)。
- `--doc_file`: 输入的 Document Triple 文件 (Step 1 Output: `converted_triple_result.jsonl`)。
    - 注意：代码内部会根据 Query ID 查找对应的 Docs。
- `--output_file`: 输出的 JSONL 文件，包含 `ranked_doc_ids`。
- `--alpha_type` (可选, default=0.5): 实体类型匹配的权重。
- `--doc_threshold` (可选, default=0.3): 文档过滤阈值。

## 3. 注意事项
- 由于文件已被移动到 package 目录下，建议在 `absQA2` 根目录下使用 `python -m absQA.matching.xxx` 的方式调用，以避免相对导入错误。
- 确保 `__init__.py` 存在于 `absQA` 和 `absQA/matching` 目录中。
