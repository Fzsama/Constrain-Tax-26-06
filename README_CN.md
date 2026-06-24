# LoRA Constraint Tax — Qwen3.6-35B 工具调用行为纠正

[English](README.md)

> 针对已测 8 个开放权重模型在 `response_format`（JSON Schema）约束下工具调用被一致压制的可复现缺陷，
> 经 6 轮实验（SFT/GRPO）确认根因为 xgrammar 解码层硬约束，探索框架层和模型层绕过方案。

## 核心发现

**Constraint Tax**: 当 `tools` + `response_format` (JSON Schema) 同时启用时，开放权重模型系统性跳过工具调用（T2=0%）。

- **根因**: xgrammar 将 JSON Schema 编译为 FSM，`<tool_call>` token 在所有 FSM 状态下 bit=0 → logit=-inf
- **非训练问题**: SFT/GRPO 在权重层修改模型偏好，无法突破解码层硬 mask
- **跨框架验证**: SGLang 0.5.9 和 vLLM 0.22.0 使用相同的 xgrammar 库，行为完全一致
- **唯一豁免**: GPT-5.4-mini（闭源，实现方式未知）
- **局限性**: 跨模型 9-model 矩阵使用固定单任务 × 5 轮/条件；对 Qwen3.6-35B-A3B 额外进行了 200+ 次跨任务验证，其余模型仅单任务验证

详见 [doc/11-final-report-constraint-tax.md](doc/11-final-report-constraint-tax.md)

## 实验矩阵（7 轮）

| # | 方法 | 数据量 | T1 | T2 Emission | T2 Valid JSON | 结论 |
|:---:|------|:---:|:---:|:---:|:---:|------|
| 1 | SFT 业务数据 | 870 | 2 次 | 0% | 0% | 学到 hallucination |
| 2 | SFT Tool Mandatory | 200 | 109 次 | 0% | 0% | Tool dependency 已学会 |
| 3-4 | SFT Schema Injection | 200 | — | 0% | 0% | Mismatch 非根因 |
| 5 | **GRPO** (RL) | 200 | 100% | 0% | 0% | RL 也无法翻转 |
| 6 | SFT 规模数据 | 6000 | 60% | 0% | 0% | 数据量非瓶颈 |
| 7 | **A2 JSON内嵌** | **8000** | **95%** | **95%** | **20%** | ✅ **首次突破 (emission rate)** |

> **指标说明**: T2 Emission Rate = raw content 含 `_tool_calls` 模式的比例（模型"意图"调工具）。T2 Valid JSON Rate = `json.loads()` 成功解析且 `_tool_calls` 非空的比例（框架可直接执行）。A2 方案 T2 Emission Rate 达 95%，证明 FSM 已被突破；Valid JSON Rate 仅 20%，需要改进 JSON 格式合规（详见 [doc/16-a2-breakthrough-results.md](doc/16-a2-breakthrough-results.md) §2.2）。

详见 [doc/16-a2-breakthrough-results.md](doc/16-a2-breakthrough-results.md)

## 项目结构

```
lora-qwen-0612/
├── config.py                          # 全局配置（路径、API、工具定义、超参数）
├── README.md                           # English README
├── README_CN.md                        # 中文 README
├── CLAUDE.md
├── .env.example                       # 环境变量模板
├── .gitignore
│
├── doc/                               # 文档（17 篇）
│   ├── 00_through_07_*.md             # 早期设计文档 & GPT 分析（7篇）
│   ├── 08-stage2-analysis-next-steps.md       # SFT 4轮分析
│   ├── 09-grpo-technical-doc.md               # GRPO 技术文档
│   ├── 10-grpo-results.md                     # GRPO 实验报告
│   ├── 11-final-report-constraint-tax.md      # ★ 最终综合报告
│   ├── 12-paper-experiment-supplement-v2.md   # Paper 实验补充
│   ├── 13-a2-progress.md                      # A2 方案进度
│   ├── 14-appendix-test-design-and-tool-schema.md  # Paper 附录
│   ├── 15-handover-lifangzheng.md             # 交接文档
│   ├── 16-a2-breakthrough-results.md          # ★ A2 T2=95% 突破报告
│   └── reference/                             # 参考文档（Unsloth Studio 平台）
│
├── scripts/                           # 脚本（10+ 个）
│   ├── 01_collect_seed_traces.py             # Phase 1: 种子轨迹采集
│   ├── 01b_generate_synthetic_seeds.py       # 合成种子生成
│   ├── 01c_enrich_seeds.py                   # 种子富化
│   ├── 02_generate_training_data.py          # Phase 2: 规模生成
│   ├── 03_train_sft.py                       # Phase 3: SFT 训练
│   ├── 04_generate_tool_mandatory_data.py     # Tool Mandatory 数据生成
│   ├── 05_train_dpo.py                        # DPO 训练 (未完成)
│   ├── 06_train_grpo.py                       # ★ GRPO 训练 (pure PyTorch + SGLang)
│   ├── 07_test_grpo_direct.py                 # GRPO 直接模型测试
│   ├── 07_test_grpo_vllm.py                   # GRPO vLLM 测试
│   ├── 08_token_mask_analysis.py              # ★ Token 级 FSM bitmask 分析
│   ├── 09_generate_a2_seed_data.py            # A2 种子数据生成
│   ├── 10_test_a2_model.py                   # ★ A2 模型 T1/T2/T3 测试
│   ├── 11_generate_a2_v2_seed_data.py         # A2 v2 种子 (50:50)
│   └── lib/                                   # 脚本库
│       ├── trace_utils.py                     #   轨迹提取/格式转换
│       └── quality_check.py                   #   6维自动化质量校验
│
├── lib/
│   └── two_pass.py                     # Two-Pass 推理框架
│
├── tests/
│   └── test_constraint_tax_lora.py     # T1/T2/T3 测试脚本
│
├── evidence/
│   └── cross_model_testing/            # 跨模型/跨框架 证据文档
│
├── artifacts/                          # 实验产物（已提交）
│   ├── experiment_results_summary.json
│   ├── model_fingerprint.json
│   └── grpo_training_summary.json
│
├── data/                              # 数据目录
│   ├── seeds/inquiry_templates.py     #   30 个种子场景模板
│   ├── generated/                     #   Phase 2 规模生成数据（gitignored）
│   └── processed/                     #   训练/评估数据集（gitignored）
│
└── outputs/                           # 训练产出（gitignored）


## 环境

- **训练**: conda `lora_qwen_0612` (torch 2.10, transformers 5.5, PEFT)
- **推理**: SGLang 0.5.9 (`sglang_059`), vLLM 0.22.0/0.23.0 (`vllm_0605`)
- **GPU**: 2× NVIDIA A800-SXM4-80GB (或同等 80GB+ VRAM)
- **Python**: 3.12+ (见 [requirements.txt](requirements.txt))

### 推理服务启动

```bash
# SGLang (推荐 — 所有测试使用的推理框架)
conda activate sglang_059
python -m sglang.launch_server \
    --model-path /path/to/model \
    --served-model-name qwen-a2 \
    --port 8082 --host 0.0.0.0 \
    --tp-size 2 \
    --mem-fraction-static 0.85 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code

# vLLM (跨框架对比)
conda activate vllm_0605
export LD_LIBRARY_PATH="/path/to/env/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

## 关键脚本

```bash
# Token 级 FSM 分析
python scripts/08_token_mask_analysis.py

# A2 模型测试 (支持 CLI 参数覆盖 API 地址和模型名)
python scripts/10_test_a2_model.py --api http://localhost:8082/v1/chat/completions --model qwen-a2
# 或通过环境变量:
CT_API_URL=http://localhost:8082/v1/chat/completions CT_MODEL_NAME=qwen-a2 python scripts/10_test_a2_model.py

# GRPO 训练
python scripts/06_train_grpo.py --n_gen 4 --epochs 3 --batch 2 --grad_accum 8 --merge

# A2 种子生成
python scripts/09_generate_a2_seed_data.py

# Two-Pass 推理
python -c "from lib.two_pass import TwoPassInference; ..."
```

## HF 数据集

| 数据集 | 条数 | 格式 | 用途 |
|------|:---:|------|------|
| [FZSAMA/qwen-constraint-tax-training-data](https://huggingface.co/datasets/FZSAMA/qwen-constraint-tax-training-data) | 6000 | 传统 tool_call → JSON | SFT 第 6 轮 |
| [FZSAMA/qwen-a2-constraint-tax-data](https://huggingface.co/datasets/FZSAMA/qwen-a2-constraint-tax-data) | 8000 | A2 JSON 内嵌 `_tool_calls` | A2 第 7 轮 |

> 注: 原始合成数据也存在于 `data/processed/synthetic_6000/` 和 `data/processed/a2_synthetic_8000/`（gitignored）。

## 实验产物

| 产物 | 位置 | 说明 |
|------|------|------|
| 实验矩阵总表 | [`artifacts/experiment_results_summary.json`](artifacts/experiment_results_summary.json) | 7 轮完整结果 + 跨模型数据 |
| 模型指纹 | [`artifacts/model_fingerprint.json`](artifacts/model_fingerprint.json) | A2 模型 config + adapter 信息 |
| GRPO 训练摘要 | [`artifacts/grpo_training_summary.json`](artifacts/grpo_training_summary.json) | 3 epoch 训练指标 |
| GRPO 完整日志 | `logs/GRPO.log` (gitignored, 446KB) | GRPO 原始训练日志 |
| A2 模型权重 | `outputs/loraed_Qwopus3.6-35B-A3B-a2-0623/` (gitignored) | 26 safetensors + config |
| A2 种子数据 | [`data/processed/a2_seed_data.json`](data/processed/a2_seed_data.json) (gitignored) | 30 条 v1 种子 |
| A2 v2 种子数据 | [`data/processed/a2_v2_seed_data.json`](data/processed/a2_v2_seed_data.json) (gitignored) | 40 条 v2 (50:50) |
| 跨模型证据 | [`evidence/cross_model_testing/`](evidence/cross_model_testing/) | 9-model 矩阵 + 原始日志 |

