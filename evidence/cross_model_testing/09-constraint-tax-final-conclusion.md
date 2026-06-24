# Constraint Tax 跨模型、跨框架最终验证报告

**日期**: 2026-06-05
**前提**: Qwen35B 在 SGLang 上 Tool Calling 与 Structured Output 互斥（见 `06-qwen-toolcalling-formatoutput.md`）

---

## 1. 验证目标

确认 Tool Calling + Structured Output 互斥问题的根因归属：
- **假设 1**: SGLang 推理框架的 `response_format` 实现方式导致
- **假设 2**: 模型训练层面的系统性偏差（Constraint Tax）
- **假设 3**: 单一模型（Qwen35B）的能力不足

---

## 2. 测试矩阵

| # | 模型 | 框架 | Tool Parser | 测试方式 |
|---|------|------|------------|---------|
| 1 | Qwen3.6-35B-A3B | SGLang 0.5.9 | qwen3_coder | 直接 API + EvoAgent |
| 2 | Qwen3.5-122B-A10B-GPTQ-Int4 | SGLang 0.5.9 | qwen3_coder | 直接 API |
| 3 | GPT-OSS-20B | SGLang 0.5.9 | gpt-oss | 直接 API |
| 4 | Qwen3.6-35B-A3B | vLLM 0.22.0 | qwen3_coder / hermes | 直接 API + EvoAgent |
| 5 | GPT-5.4-mini | OpenAI 云端（AEP） | 内置 | 直接 API |

所有测试统一使用 3 组对照：

| 测试组 | tools | response_format | 验证目标 |
|--------|-------|-----------------|---------|
| A | ✅ ON | ❌ OFF | 模型能否调用工具（基线） |
| B | ✅ ON | ✅ ON | **关键：能否同时调工具 + 输出 JSON** |
| C | ❌ OFF | ✅ ON | 能否输出合法 JSON（基线） |

测试脚本：`tests/fz-qwen-test/test_122b_tool_response_format.py`、`test_gptoss_tool_rfmt.py`、`test_vllm_qwen35b.py`

---

## 3. 测试结果

### 3.1 所有模型 5 轮平均对比

| 模型 | 框架 | A: tools+rfmt=OFF | B: tools+rfmt=ON | C: rfmt=ON |
|------|------|:--:|:--:|:--:|
| **GPT-5.4-mini** | OpenAI 云端 | ✅ 2.0 次, 3.9s, 100% | ✅ **2.0 次, 3.9s, 100%** | 1084 字, 8.6s |
| Qwen35B-A3B | SGLang | ✅ 2.0 次, ~2s, 100% | ❌ 0 次, 0% | 6 cards, ~12s |
| Qwen35B-A3B | vLLM | ✅ 2.2 次, 3.7s, 100% | ❌ 0 次, 0% | 660 字, 9.2s |
| Qwen122B-A10B | SGLang | ✅ 2.0 次, 2.4s, 100% | ❌ 0 次, 0% | 345 字, 30.3s |
| GPT-OSS-20B | SGLang | ⚠️ 1.0 次, 0.7s, 100% | ❌ 0 次, 0% | 206 字, 1.2s |

### 3.2 EvoAgent 实测表现

| 模型 | 框架 | 实际行为 |
|------|------|---------|
| GPT-5.4-mini | OpenAI 云端 | skill_manager → websearch+fetchurl+KB → 6 张分析卡片。标准 3 步流程 |
| Qwen35B-A3B | SGLang（无注入） | 直接输出 1 张空壳卡片或占位文字，0 次工具调用 |
| Qwen35B-A3B | vLLM | 输出 "正在分析询盘..." 伪进度卡片，0 次工具调用 |

### 3.3 开放权重模型的 "敷衍" 行为

在多轮测试中观察到开放权重模型在 `response_format` 约束下的特征性 "敷衍" 输出：

- Qwen122B: `"buyer_background": "Simulated Websearch Results"` — 模型知道该搜索，但选择伪造
- Qwen35B(vLLM): `{"message":"正在分析询盘...", "description":"正在加载子技能指令并分析买家背景..."}` — 口头说在分析，实际没做
- GPT-OSS-20B: `"recommendations": "", "key_findings": []` — 用空值填充 schema

---

## 4. 根因分析

### 4.1 排除推理框架因素

SGLang 和 vLLM 是两个完全独立的推理框架，其 `response_format`（OpenAI `json_schema`）的实现路径不同：
- **SGLang**: 在 `stream_async` 中注入 `response_format`，LLM 在生成时受 schema 约束
- **vLLM**: 通过 `StructuredOutputsConfig` 在 engine 层处理

两个框架下模型行为完全一致（`rfmt=ON` 时 0 次工具调用），**排除推理框架实现差异**。

### 4.2 排除模型规模因素

35B（激活参数 ~3B）和 122B（激活参数 ~10B）的 MoE 模型表现完全一致。**排除模型规模不足**。

### 4.3 排除模型架构因素

Qwen MoE（Qwen3.5/3.6）和 GPT-OSS（Dense Transformer）两种架构表现完全一致。**排除架构差异**。

### 4.4 确定根因：训练层面的 Constraint Tax

该现象与 2026 年论文 "The Constraint Tax: Measuring Validity-Correctness Tradeoffs in Structured Outputs for Small Language Models" 描述的行为高度一致：

> Schema Validity ↑ → Task Accuracy ↓ → Tool Calling 被压制

当模型在训练中学习到 "有 JSON Schema 约束 = 直接生成 JSON" 的捷径时，推理阶段面对 `response_format` 约束会：
1. 评估路径 A（调工具获取数据 → 推理 → 填 JSON）—— 高成本
2. 评估路径 B（直接用参数知识填 JSON）—— 低成本
3. 选择路径 B

GPT-5.4-mini 云端模型能同时处理两者，说明 OpenAI 在 RLHF/instruction tuning 阶段对此做了针对性优化。开放权重模型（Qwen、GPT-OSS）尚未达到这一水平。

---

## 5. 结论

```
Constraint Tax 是开放权重模型的系统性训练偏差，与以下因素均无关：

  ❌ 推理框架（SGLang vs vLLM 结果一致）
  ❌ 模型规模（35B vs 122B 结果一致）
  ❌ 模型架构（Qwen MoE vs GPT-OSS Dense 结果一致）
  ❌ Tool Call Parser 配置（qwen3_coder / gpt-oss / hermes 结果一致）

  ✅ 唯一可行方案：GPT-5.4-mini 云端 API
```

### 对 EvoAgent 各方向的影响

| 方向 | 结论 |
|------|------|
| A（Code-Level Research Enhancement） | ✅ 必需且有效。代码层绕过模型限制，预注入研究数据 |
| B（Two-Agent Separation） | ✅ 必需且有效。Research/Format 分离，各解除约束 |
| C（模型升级） | ❌ 不成立。更大模型不能解决此问题 |
| 更换推理框架（vLLM） | ❌ 不成立。vLLM 与 SGLang 结果一致 | (continued)
| D（br-2 回归） | 不推荐 |

---

## 6. 附录：测试命令参考

### SGLang 启动
```bash
no_proxy="$no_proxy,0.0.0.0" python -m sglang.launch_server \
  --model-path /path/to/model --served-model-name xxx \
  --port 8082 --host 0.0.0.0 --tp-size 2 \
  --mem-fraction-static 0.85 --max-total-tokens 128144 \
  --max-running-requests 64 --chunked-prefill-size 8192 \
  --enable-flashinfer --log-level warning \
  --reasoning-parser <qwen3|gpt-oss> \
  --tool-call-parser <qwen3_coder|gpt-oss> \
  --trust-remote-code
```

### vLLM 启动
```bash
vllm serve /path/to/model --served-model-name xxx \
  --port 8082 --host 0.0.0.0 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 128144 \
  --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --trust-remote-code
```
