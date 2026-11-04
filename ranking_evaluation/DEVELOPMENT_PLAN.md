# Person 3 — Hybrid Ranking & Evaluation 开发计划与记录

负责人：Yu Yue

工作分支：`yuyue`
创建日期：2026-11-04（Asia/Shanghai）

## 目标与当前状态

在 `ranking_evaluation/` 实现 BM25 与 SBERT 排名融合、可选 Cross-Encoder 重排、检索评估及比较实验。所有相关代码、文档、验证记录和实验配置通过 Git 提交到 `yuyue` 分支，最终集成另行进行。

当前已完成：个人分支同步 Person 1 的实现，基础提交为 `259d2a9`；P1 的拟议接口、模拟示例和验证已完成。RRF、重排和评估尚未实现；模拟验证不代表真实实验结果。

## 职责与依赖

```text
Person 1：原始 claim → 未排序、已去重的候选论文及元数据
Person 2：同一候选集 → BM25 排名 + SBERT 排名
Person 3：两份排名 → RRF → 可选 Cross-Encoder → 排名与评估结果
Person 4：排序后的论文 → 证据提取与验证
```

- 不重复实现 Person 2 的 BM25、SBERT；先以模拟排名完成独立开发。
- 最终端到端联调及真实方法比较依赖 Person 2 的输出。
- Cross-Encoder 使用原始 claim 和候选论文文本，不依赖 BM25/SBERT 分数可比。
- 固定 SciFact benchmark 与 live search 分开。Person 1 的在线结果没有自动获得 SciFact 标准答案，不能直接用于计算基准 Recall/MRR/nDCG。

## 现有接口与待确认约定

Person 1 的已确认接口见：

- [交接说明](../live_search/docs/retrieval_integration.md)
- [输出结构](../live_search/src/paper_search/retrieval_models.py)
- [论文元数据](../live_search/src/paper_search/metadata.py)

使用 `RetrievalResult.original_claim` 和 `papers`。每个条目包含 `identity`、`identity_aliases`、`paper`、`provenance`、`source_records`。保留这些信息，避免融合或重排丢失来源。标题、摘要可能缺失，PDF URL 不代表已获取全文。`top_k=30` 是每个来源的限制，不是最终全局排名数量。来源名次不是 BM25/SBERT 名次。

以下为拟议的 Person 2 → Person 3 接口，尚未与 Naila 确认：

```python
{
    "claim_id": "example-claim",
    "claim": "原始 claim",
    "papers_by_id": {"paper-A": "对应的完整论文条目"},
    "bm25_ranking": ["paper-A", "paper-B"],
    "sbert_ranking": ["paper-B", "paper-A"],
    "latency_seconds": {"bm25": 0.0, "sbert": 0.0},
}
```

数组从最相关到最不相关排序，第一项为第 1 名；两份列表使用同一 ID 空间，不包含重复 ID。需确认候选集、截断深度、缺失分支处理及耗时范围。线上使用 Person 1 的 identity；SciFact 使用固定 corpus 文档 ID，评估标签必须与其一致。跨运行对齐时保留 aliases，不能假定首选 identity 永远不变。

Person 3 输出拟保留：claim ID、论文 ID、最终名次、RRF 分数、两个分支原始名次、可选重排分数、完整论文引用、各阶段耗时及配置。正式 schema 在实现阶段确定。

## 分阶段实施与验收

| 阶段 | 实施内容 | 验收标准 | 依赖 | 状态 |
|---|---|---|---|---|
| P0 | 明确任务、分支、接口与开发记录 | 计划文件已提交，未确认事项显式列出 | 无 | 已完成 |
| P1 | 定义输入输出结构，准备固定模拟输入 | 两份排名可关联同一论文；保留原始 claim 和来源；明确无效输入处理 | 无，接口暂用拟议版本 | 本地实现并验证，待团队确认接口 |
| P2 | 实现 RRF | 手工可核算的融合结果正确；空列表、单分支、重复 ID、并列分数有明确行为；结果确定且可截断 | P1 | 未开始 |
| P3 | 实现 Recall@K、MRR、nDCG@K 与计时记录 | 固定标签样例的结果可手算验证；处理无相关标签、空结果、K 边界；记录指标定义 | P1 | 未开始 |
| P4 | 加入可选 Cross-Encoder | 可关闭；只重排指定前 N 篇；保留来源；明确缺失文本、模型不可用和截断处理 | P2，模型选择待定 | 未开始 |
| P5 | 接入固定 SciFact 数据与 Person 2 输出 | corpus/claim/qrels ID 一致；记录数据版本、split、检索与模型配置；无测试集调参 | P2/P3，Person 2 与共享数据 | 未开始 |
| P6 | 比较实验与消融 | 同一数据与候选条件比较 BM25、SBERT、RRF、RRF+Cross-Encoder；保存配置、逐 claim 结果及汇总 | P4/P5 | 未开始 |
| P7 | 在线集成及下游交接 | 用 Person 1 实际输出完成联调；排序论文可追溯来源并交给 Person 4；故障与空结果有记录 | Person 2 可用，P4 | 未开始 |

### 实现时的具体决策

- RRF：`score(d) = sum(1 / (rrf_k + rank(d)))`，从 1 开始计数；未出现的分支贡献 0。拟以 `rrf_k=60` 为起点，明确它与结果条数 K 不同；参数只在开发/验证集上选择。
- Recall@K：top K 中相关文档数除以该 claim 的全部相关文档数。
- MRR：首个相关文档排名的倒数；若做截断，明确命名为 MRR@K。
- nDCG@K：记录 gain、折扣和 IDCG 定义；初始拟用二值相关性。无相关标签的 claim 如何统计必须在实验前固定并报告数量。
- 初始拟报告 K = 5、10、20；若数据或候选深度不支持，调整并记录原因。
- 延迟分别记录检索、RRF、重排、总流程；并发检索时间不能简单相加。区分模型加载、冷启动与预热后的推理，报告运行次数、设备和聚合方式。
- Cross-Encoder 模型尚未选择。选型后固定模型名称、revision、输入长度、批量大小、重排深度及运行设备；模型下载需显式执行并记录。

## 建议文件结构

以下是计划路径，尚未创建或承诺为最终公共接口：

```text
ranking_evaluation/
  DEVELOPMENT_PLAN.md
  README.md                 # 安装、调用与复现实验
  schemas.py                # 输入输出约定
  rrf.py                    # 排名融合
  metrics.py                # 检索指标
  reranker.py               # 可选 Cross-Encoder
  evaluate.py               # 固定 benchmark 运行入口
  tests/                    # 手算样例和关键边界验证
  experiments/              # 配置、结果索引与实验报告
```

大数据、模型权重、密钥、缓存不纳入 Git；保留可复现的下载说明、版本与必要哈希。实验结果的实际保存路径和 Git 跟踪策略在首次实验前确定。

## 当前可以立刻做什么

P1 已完成本地实现，下一步执行 P2：用模拟 BM25/SBERT 排名实现并验证 RRF，再做 P3 指标。不等待 Person 2 完工，不将模拟结果写成真实实验结论。

需要与 Naila 确认：论文 ID、排名格式、截断深度、耗时字段；与共享数据负责人确认 SciFact 版本、split 和相关性标签映射。此处仅记录协作事项，尚未发送消息，也未假定对方同意。

## 持续留痕规则

每个有实质进展的工作批次都更新本文件：

1. 更新阶段状态和待办事项；记录接口或计划变更的原因。
2. 在下面追加日期、修改文件、验证命令与实际结果、限制和下一步。
3. 功能改动配套必要验证；未运行、失败、跳过分别写明，不将他人历史测试结果视为本次验证。
4. 用清晰的 Git commit 保存工作，并推送到 `yuyue`；推送失败时记录原因，不声称完成。
5. 实验记录数据/split、代码提交、模型版本、参数、设备、命令、随机种子（如适用）、耗时与输出路径。报告区分真实实验、模拟验证和历史记录。

Git 历史提供提交痕迹；本日志解释每次工作做了什么、如何验证及下一步。提交与推送事实以 Git 历史及工具返回为准。

## 工作日志

### 2026-11-04 — 分支准备与计划建立

- 已将已有个人分支 `yuyue` 从 `8956098` 快进至 Person 1 的 `259d2a9` 并成功推送；未更改 `main` 或 `raghavi` 的远程分支。
- 阅读 Person 1 的交接说明、论文及结果 schema，确认输出为未排序候选集合。
- 建立本文件，列出 P0–P7、拟议接口、验收标准与后续记录规则。
- 验证：基础分支推送后 `git status --short --branch` 显示 `yuyue...origin/yuyue`，工作区干净；计划文档提交前执行 `git diff --check`。
- 本次没有实现排名算法，没有运行模型、在线 API 或评估实验。Person 1 文档中的历史测试数量不作为本次验证结果。
- 下一步：P1 输入输出结构及模拟样例 → P2 RRF 与关键边界验证 → P3 指标。
- 文档首次检查发现两处 Markdown 换行尾空格，已清理；重新执行差异检查确认。

### 2026-11-04 — P1 接口与离线示例

- 新增 `schemas.py`，直接引用 Person 1 的 `RetrievalResult`、`RetrievedPaper`。实际拟议接口采用嵌套的 `candidates` 保留完整上游结果，替代上文最初草案的 `papers_by_id`，避免复制和丢失诊断数据；详见本模块 README。
- 新增 `examples/mock_input.py`，构造三篇明确标记为虚构的论文及两份人工排名；没有伪造真实论文或推理结果。
- 校验重复候选、重复排名、未知论文 ID、claim 不一致和无效耗时；允许空排名，未知耗时使用 `None`。`RankedPaper` 预留原始条目、融合分数、分支名次及可选重排分数。
- 新增 README，记录最小依赖和运行命令。项目 `.venv` 被 Git 忽略；首次依赖安装受网络限制失败，获得工具授权后安装成功。
- 验证环境：Python 3.13.1、Pydantic 2.14.2、pytest 9.2.0；Python 3.11 尚未验证。
- `PYTHONPATH=live_search/src .venv/bin/python -m pytest ranking_evaluation/tests -q`：**8 passed**。
- `PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.examples.mock_input`：成功输出 JSON；临时输出在 `/private/tmp/nlp-person3-mock-input.json`，不作为实验报告提交。
- `git diff --check`：通过。未运行在线搜索、模型、Person 1 完整测试或真实检索实验。
- 下一步：P2 实现 RRF，包括可手算结果、空输入、单分支、并列分数与截断验证。Person 2 接口仍待 Naila 确认。
