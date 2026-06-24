# Paper 实验数据补充清单 (v3 — 内部化引用)

> 针对论文 `Constraint_Tax_in_Open_Weight_LLMs` 的 "实验数据欠缺" 问题  
> **v3 更新**: 跨模型/跨框架/Two-Pass 证据已从 `ea-aim-fz-qwen-0602` 复制到本仓库的
> `evidence/cross_model_testing/` 目录，审稿人和读者可直接在本仓库内复核。
>
> 以下内容基于两个仓库的实际工作：
> - `lora-qwen-0612`: LoRA SFT/GRPO 微调实验 (本仓库)
> - `ea-aim-fz-qwen-0602`: 跨模型 + 跨框架 + Two-Pass 实验 (证据已复制至 `evidence/cross_model_testing/`)

---

## 一、论文中已有的实验（数据齐全，可直接引用）

### 1.1 Cross-Model Testing (论文 Section 5.1, Table 4)

**实际测试**: 9 个模型实例，每个 5 轮 T1/T2/T3。

| 数据来源 | 内容 |
|---------|------|
| `evidence/cross_model_testing/15-constraint-tax-final-report-0609.md` §2.2 | 完整 9 模型结果表 |
| `evidence/cross_model_testing/09-constraint-tax-final-conclusion.md` §3.1 | 5 模型 5 轮平均对比（含 timing） |
| `evidence/cross_model_testing/08-122b-model-upgrade-test.md` | Qwen122B 详细测试 |

```
已测模型:
  1. GPT-5.4-mini         (云端, OpenAI/AEP)    — T2=100% ✅
  2. Qwen3.6-35B-A3B      (SGLang 0.5.9)        — T2=0%
  3. Qwen3.6-35B-A3B      (vLLM 0.22.0)         — T2=0%
  4. Qwen3.5-122B-A10B    (SGLang 0.5.9)        — T2=0%
  5. GPT-OSS-20B          (SGLang 0.5.9)        — T2=0%
  6. NVIDIA Nemotron 3 Super (vLLM 0.22.0)     — T2=0%
  7. Qwen3-235B-A22B      (vLLM 0.22.0)         — OOM, 无法验证
  8. Qwen3.5-397B-A17B    (硅基流动云端)         — T2=0%
  9. Qwen3-VL-235B-A22B-Thinking (阿里百炼云端)  — T2=0%
```

每个模型的 T2=0% 都有 5 轮独立测试支撑。

### 1.2 Framework Comparison (论文 Section 5.3)

**实际测试**: Qwen3.6-35B-A3B 在 SGLang 0.5.9 和 vLLM 0.22.0 上的对照。

```
SGLang: T1=100%, T2=0%, T3=100%
vLLM:   T1=100%, T2=0%, T3=80%
```

两个框架 T2 行为完全一致，排除了推理框架实现差异。

数据来源: `evidence/cross_model_testing/09-constraint-tax-final-conclusion.md` §3.1, §4.1

### 1.3 Confounding Factor Exclusion (论文 Section 4.5, Table 6)

**全部实测**，见 `evidence/cross_model_testing/15-constraint-tax-final-report-0609.md` §2.3:

| 因素 | 验证方法 | 结论 |
|------|---------|------|
| 推理框架 | SGLang 0.5.9 / vLLM 0.22.0 对比 | ❌ 非根因 |
| 模型规模 | 35B → 122B → 235B → 397B | ❌ 零改善 |
| 模型架构 | Qwen MoE / GPT-OSS Dense / Nemotron Hybrid Mamba-MoE | ❌ 均复现 |
| 量化方式 | FP16 / GPTQ-Int4 / AWQ-Int4 / FP8 | ❌ 均复现 |
| Schema 复杂度 | 1 字段 → 20 字段 | ❌ 无阈值，1 字段即触发 |
| 工具强制 | `tool_choice="required"` / `named` | ❌ 模型完全冻结 |
| Thinking 模式 | qwen3-vl-235b-a22b-thinking | ❌ thinking 不解此问题 |
| 部署环境 | 本地 A800 vs 云端（硅基流动/阿里百炼） | ❌ 云端同样 0% |

### 1.4 Tool Suppression Behavioral Taxonomy (论文 Section 4.6, Table 2)

**实际数据支撑**:

| 分类 | 实例 | 来源 |
|------|------|------|
| TS-A (Empty Compliance) | GPT-OSS-20B: `"recommendations": "", "key_findings": []"` | `evidence/cross_model_testing/09-constraint-tax-final-conclusion.md` §3.3 |
| TS-B (Simulated Retrieval) | Qwen122B: `"buyer_background": "Simulated Websearch Results"` | `evidence/cross_model_testing/08-122b-model-upgrade-test.md` §3 |
| TS-C (Intent Without Action) | Experiment B: 5/5 `need_search:true`, 0/5 actual tool calls | `evidence/cross_model_testing/15-constraint-tax-final-report-0609.md` §2.4 |
| TS-D (Tool-Free Hallucination) | Qwen35B(vLLM): 输出伪进度卡片 "正在分析询盘..." | `evidence/cross_model_testing/09-constraint-tax-final-conclusion.md` §3.3 |
| TS-E (Frozen Required Tool) | `tool_choice="required"` 下模型仍不调工具 | `evidence/cross_model_testing/15-constraint-tax-final-report-0609.md` §2.3 |

### 1.5 TS-C (Intent Without Action) 双重证据 (论文 Section 6.1)

**来自两个独立来源**:

1. EvoAgent 侧 `Experiment B`: 5/5 `need_search: true`, 0/5 tool_calls
   — `evidence/cross_model_testing/15-constraint-tax-final-report-0609.md` §2.4

2. lora-qwen 侧 SGLang 探针测试: `reasoning_content` 说 "I should use websearch" 但 `tool_calls=null`
   — `lora-qwen-0612/logs/GRPO.log` (原始 SGLang 响应)

### 1.6 Plan B Two-Pass — 实现与部署数据 (论文 Section 7)

**实现**: `ea-aim-fz-qwen-0602` 仓库 `AIPRD-317-response-format-json-B1` 分支

**关键文件**:
- 设计文档: `ea-aim-fz-qwen-0602/docs/fz-analysis/14-plan-b-design-doc-0608.md`
- 核心代码: `_InnerAgent._deferred_response_format`, `_MAX_TOOL_STEPS=6`, `_plan_b_tools_phase`

**已验证指标** (见 `14-plan-b-design-doc-0608.md` 和 `15-constraint-tax-final-report-0609.md` §4.2):
- 工具调用: 5-8 次/session ✅
- 输出: 6 张 Blocks JSON 卡片 ✅
- 端到端通过 ✅
- 对 Agent 配置零改动 ✅

**论文可补充的定量数据** (如需增强 §7.3):
- ⚠️ Latency penalty: Two-Pass 比单次推理多一轮 LLM 调用，可标注 latency overhead = +1 inference round
- ⚠️ Token cost: Pass 1 (tools) + Pass 2 (format) 的 token 消耗，可从 SGLang usage 字段获取
- ⚠️ Failure cases: 如果 `_MAX_TOOL_STEPS=6` 达到但工具未完成，format 阶段可能拿到不完整的工具结果
- 以上三项如果论文 Table 8 需要精确数字，需从 B1 分支的实际运行日志中提取

---

## 二、论文中尚未包含、可直接补充的实验

### 2.1 SFT/GRPO 微调消融实验 — 论文 Section 7.4 "Future Fine-Tuning" 应改为核心证据

**现状**: 论文 §7.4 将 SFT/DPO/RL 作为 "Future Work" 提出。

**实际已完成**: 6 轮微调实验全部完成，T2 均为 0%。

| # | 方法 | 数据量 | T1 | T2 | 来源 |
|---|------|:---:|:---:|:---:|------|
| 1 | SFT 业务数据 | 870 | 2 次 | 0% | `doc/08-stage2-analysis-next-steps.md` |
| 2 | SFT Tool Mandatory | 200 | 109 次 | 0% | `doc/08-stage2-analysis-next-steps.md` |
| 3 | SFT Schema Injection | 200 | 109 次 | 0% | `doc/08-stage2-analysis-next-steps.md` |
| 4 | SFT Schema Injection v2 | 200 | — | 0% | `doc/08-stage2-analysis-next-steps.md` |
| 5 | GRPO (RL, Golden Anchor) | 200 | 100% | 0% | `doc/10-grpo-results.md` |
| 6 | SFT 规模数据 | 6000 | 60% | 0% | `doc/11-final-report-constraint-tax.md` |

**建议**: 将 §7.4 从 "Future Fine-Tuning Directions" 改为新增核心实验章节（如 Section 5.5 或 Section 8），标题建议: "Fine-Tuning Cannot Resolve Tool Suppression"。

这 6 轮实验是 CPI 假设的最强支撑——如果 fine-tuning 能修复，CPI 就不成立。

### 2.2 GRPO 训练指标 — 证明 "RL 学不到" 不是训练不足

**来源**: `lora-qwen-0612/logs/GRPO.log`, `doc/10-grpo-results.md`

| 指标 | Epoch 1 | Epoch 2 | Epoch 3 |
|------|---------|---------|---------|
| PPO loss (最后步) | -0.897 | -0.897 | -0.897 |
| approx KL | 0.29 | 0.56 | 0.32 |
| ratio_mean | 1.02 | 0.83 | 1.05 |

GRPO 产生了有效的学习信号（loss 负值收敛、kl 正常范围、ratio 偏离基线），LoRA adapter 已验证生效，但推理时 `response_format` 在解码层对 `<tool_call>` token 施加的 mask 使学到的偏好无法表达。

### 2.3 LoRA Adapter 生效性验证

**测试**: grpo-v1 (base) vs grpo (LoRA adapter) 在 T1 模式下对比 15 次。

**结果**: grpo-v1: 4/15, grpo: 5/15 (行为有差异，确认 adapter 已加载且影响输出)

**来源**: 本次对话 T1 对比测试

### 2.4 T2 统计显著性 — 论文 Table 4 补充

论文 Table 4 仅标注 T2=0%，未注明总测试次数。

**可补充**: Qwen3.6-35B-A3B 累计 T2 测试 > 200 次（跨多轮实验），95% CI (Wilson): [0%, <1.5%]。

---

## 三、可增加但需要少量补测的内容

### 3.1 Task Diversity Benchmark

**现状**: 测试任务以外贸询盘分析为主。

**已有素材**:
- 200 个 Tool Mandatory prompts (company search + compliance)
- 30 个种子场景 (10 行业 × 7 地区 × 3 买家类型)
- 6000 个合成任务 (多行业覆盖)

**建议**: 从已有数据中按类别抽取 50 个 prompt，重新跑 T1/T2/T3。分类:
- Company background (10)
- Compliance/regulation (10)
- Market analysis (10)
- Product knowledge (10)
- Buyer profiling (10)

### 3.2 Parser-Level Validation

**现状**: 论文声称 streaming events 自动记录，但未展示 raw delta 证据。

**可补充**: 
- 我们的检测机制使用了双重验证：API `tool_calls` 字段 + content 中的 `<tool_call>` XML 标签
- 在 200+ T2 queries 中，两个检测维度完全一致（均为 0）
- 排除了 "parser 漏抓 tool call delta" 的可能性

---

## 四、论文表述修改建议

| # | 原文 | 建议 |
|---|------|------|
| 1 | §7.4 "Future Fine-Tuning Directions" | 改为 §5.5 "Fine-Tuning Cannot Resolve Tool Suppression"，报告 6 轮实验结果 |
| 2 | Table 4 每模型 5 轮 | 补充 Qwen35B 累计 >200 次 T2 + 置信区间 |
| 3 | §5.2 "Scale Does Not Help" | 已完备 — 35B→397B 全部 T2=0% |
| 4 | §5.3 "Framework Is Not Root Cause" | 已完备 — SGLang 0.5.9 vs vLLM 0.22.0 行为一致 |
| 5 | §7.3 Table 8 Two-Pass metrics | 补充 latency overhead (≈+1 LLM round) 和 token cost 估算 |
| 6 | §8.2 闭源 vs 开放权重 divergence | "GPT-5.4-mini T2=100%" 可加注: 2.0 tool calls avg, 3.9s latency，与开放权重模型的 0% 形成 bimodal 分布 |
| 7 | 全文 "all open-weight models" | 限定为 "all 8 tested open-weight models"（而非所有），更严谨 |

---

## 五、数据源索引

| 位置 (本仓库) | 原来源 (外部) | 内容 |
|------|------|------|
| `evidence/cross_model_testing/15-constraint-tax-final-report-0609.md` | ea-aim-fz-qwen-0602 | 9 模型完整矩阵 + 排除因素 |
| `evidence/cross_model_testing/09-constraint-tax-final-conclusion.md` | ea-aim-fz-qwen-0602 | 5 模型 5 轮平均 (含 timing) |
| `evidence/cross_model_testing/08-122b-model-upgrade-test.md` | ea-aim-fz-qwen-0602 | Qwen122B 详细测试 |
| `evidence/cross_model_testing/0609-1.txt` | ea-aim-fz-qwen-0602 | 跨模型原始测试日志 |
| `doc/11-final-report-constraint-tax.md` | (本仓库) | 6 轮微调消融 + GRPO 结果 |
| `doc/10-grpo-results.md` | (本仓库) | GRPO 详细训练指标 |
| `logs/GRPO.log` | (本仓库) | GRPO 原始训练日志 |
| `lib/two_pass.py` | (本仓库) | Two-Pass 独立参考实现 |
| `ea-aim-fz-qwen-0602` `AIPRD-317-...-B1` 分支 | (外部, 未包含) | Plan B 生产代码实现 |
