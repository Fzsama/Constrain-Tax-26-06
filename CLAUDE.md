# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## 项目概述

Constraint Tax 研究项目——探究 Qwen3.6-35B-A3B 在 `response_format` (JSON Schema) 下工具调用被压制的根因和解决方案。经 7 轮实验（4 SFT + 1 GRPO + 1 规模数据 + 1 A2）确认根因为 xgrammar FSM 解码层硬约束，最终通过 A2 方案（JSON 内嵌 tool_calls）突破 T2=95%。

## 运行环境

```bash
# 训练环境
conda activate lora_qwen_0612

# SGLang 推理环境
conda activate sglang_059

# vLLM 推理环境 (需设置 LD_LIBRARY_PATH)
export LD_LIBRARY_PATH="/root/miniforge3/envs/vllm_0605/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
conda activate vllm_0605
```

## 核心文档

| 文档 | 内容 |
|------|------|
| `doc/11-final-report-constraint-tax.md` | ★ 最终综合报告：7轮实验 + xgrammar/vLLM 机制溯源 + 解决方案 |
| `doc/10-grpo-results.md` | GRPO 实验完整报告 |
| `doc/13-a2-progress.md` | A2 方案进度 |
| `doc/14-appendix-test-design-and-tool-schema.md` | Paper 附录：测试设计 + 工具/Schema 定义 |
| `doc/16-a2-breakthrough-results.md` | ★ A2 突破报告：T2=95% |

## 关键脚本

| 脚本 | 功能 |
|------|------|
| `scripts/06_train_grpo.py` | GRPO 训练 (pure PyTorch + SGLang generation, Golden Anchor) |
| `scripts/08_token_mask_analysis.py` | Token 级 xgrammar FSM bitmask 分析 |
| `scripts/09_generate_a2_seed_data.py` | A2 格式种子数据生成 |
| `scripts/10_test_a2_model.py` | A2 模型 T1/T2/T3 测试 |
| `lib/two_pass.py` | Two-Pass 推理框架 |
| `tests/test_constraint_tax_lora.py` | T1/T2/T3 测试脚本 |
| `config.py` | 全局配置（路径、API、工具定义、超参数） |

## 训练数据格式

旧格式 (SFT 前 5 轮使用):
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", ...},
    {"role": "assistant", "content": "{...}"}
  ]
}
```

A2 格式 (当前方案):
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"_tool_calls\": [...], \"blocks\": [...]}"}
  ]
}
```

## 实验状态

| 方案 | 方法 | T2 Emission | T2 Valid JSON | 状态 |
|------|------|:---:|:---:|------|
| 原始 SFT | 870条业务数据 | 0% | 0% | ❌ |
| Tool Mandatory SFT | 200条极简 | 0% | 0% | ❌ |
| Schema Injection SFT | 200条 | 0% | 0% | ❌ |
| GRPO | 200条 + Golden Anchor | 0% | 0% | ❌ (学到但无法表达) |
| 规模 SFT | 6000条 | 0% | 0% | ❌ |
| A2 (JSON内嵌tool_call) | 8000条 | **95%** | **20%** | ✅ **突破 (emission rate)** |

> 注: T2 Emission Rate = raw content `_tool_calls` 模式匹配率; T2 Valid JSON Rate = `json.loads()` 可解析率

## 关键依赖

- **EvoAgent** (`/root/0420-fz/ea-aim-fz-qwen-0602/`) — Constraint Tax 发现地，跨模型测试，B1 Two-Pass 实现
  - **跨模型证据**: 已复制到 `evidence/cross_model_testing/` 目录
- **Unsloth** — QLoRA 训练框架 (Studio)
- **SGLang 0.5.9** — 推理服务，xgrammar guided decoding
- **vLLM 0.22.0** — 对比推理框架
- **GPT-5.4-mini** (AEP 网关) — Teacher 模型 / 唯一豁免模型
- **Qwen3.6-35B-A3B** (Qwopus) — Student 模型

## 已知问题

- `tokenizer_config.json` 中 `tokenizer_class: TokenizersBackend` 需改为 `Qwen2Tokenizer`
- vLLM 0.23.0 编译为 CUDA 13.0，driver 535 仅支持 CUDA 12.6，需设置 `LD_LIBRARY_PATH`
- Unsloth 与 transformers 5.x / vLLM 0.23.0 存在依赖冲突，已移除 unsloth 包
