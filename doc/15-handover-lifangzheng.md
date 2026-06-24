# 李方正交接文档

所有核心代码见以下两个仓库：
- **/root/0420-fz/lora-qwen-0612**（Constraint Tax 研究主仓库）
- **/root/0420-fz/ea-aim-fz-qwen-0602**（生产 Agent 系统，Constraint Tax 发现地，B1 Two-Pass 实现）

---

# Constraint Tax 发现与验证

- 状态：已完成（6 轮实验，论文已投稿）

- 进度
    1. 在 EvoAgent 生产部署中首次发现 Tool Suppression 现象
    2. 完成 9 模型 × 2 框架 × 3 条件的跨模型受控实验
    3. 排除推理框架、模型规模、量化方式、Schema 复杂度等全部混淆因素
    4. 提出 Constraint Priority Inversion (CPI) 假说
    5. 完成 Tool Suppression 行为分类学（TS-A 到 TS-E）

1. 整体分析见：[Constraint Tax 最终验证报告](ea-aim-fz-qwen-0602/docs/fz-analysis/15-constraint-tax-final-report-0609.md)

2. 跨模型测试代码见：**/root/0420-fz/ea-aim-fz-qwen-0602/tests/fz-qwen-test/**

3. 论文见：**/root/0420-fz/CT-0617/**（含 PDF、Latex 源文件、README）

---

# Constraint Tax 根因溯源：xgrammar FSM Token Mask

- 状态：已完成

- 核心发现
    1. 追踪 SGLang 0.5.9 源码，确认完整调用链：`response_format` → `xgrammar.compile_json_schema()` → `GrammarMatcher.fill_next_token_bitmask()` → `apply_token_bitmask_inplace_triton()` → bit=0 → logit=-inf
    2. 追踪 vLLM 0.22.0 源码，确认使用**完全相同的 xgrammar 库**，机制 100% 一致
    3. 编写独立脚本直接调用 xgrammar API，逐状态提取 FSM bitmask，实验验证 `<tool_call>` (id=248058) 在所有状态下均被 mask

1. 详细分析见：[最终报告 §4](lora-qwen-0612/doc/11-final-report-constraint-tax.md)

2. Token 级分析脚本见：**/root/0420-fz/lora-qwen-0612/scripts/08_token_mask_analysis.py**

3. 源码分析见：
   - SGLang: `sglang/srt/constrained/xgrammar_backend.py`, `sglang/srt/constrained/triton_ops/bitmask_ops.py`
   - vLLM: `vllm/v1/structured_output/backend_xgrammar.py`, `vllm/v1/worker/gpu_model_runner.py`

---

# SFT 微调实验（4 轮）

- 状态：已完成（四轮全部 T2=0%）

- 进度
    1. 第 1 轮：870 条业务数据 → T1=2 次, T2=0%（学到 hallucination 而非 tool usage）
    2. 第 2 轮：200 条 Tool Mandatory 极简数据 → T1=109 次, T2=0%（学会 tool dependency，schema 仍压制）
    3. 第 3-4 轮：200 条 Schema Injection 数据 → T2=0%（排除 Training-Inference Mismatch）
    4. 第 6 轮：6000 条规模合成数据 → T1=60%, T2=0%（数据量不改善 T2）

1. 分析见：[Stage 2 分析](lora-qwen-0612/doc/08-stage2-analysis-next-steps.md)

2. 训练数据见：
   - Tool Mandatory: **/root/0420-fz/lora-qwen-0612/data/processed/tool_mandatory_dataset.json**
   - 6000 条规模数据: **HF: FZSAMA/qwen-constraint-tax-training-data**

3. 数据生成脚本见：**/root/0420-fz/lora-qwen-0612/scripts/04_generate_tool_mandatory_data.py**

---

# GRPO 强化学习实验

- 状态：已完成（loss 有效，T2 仍为 0%）

- 核心发现
    1. GRPO 训练产生有效学习信号（loss=-0.897, kl=0.29~0.56, ratio 偏离基线）
    2. LoRA adapter 已验证生效（grpo-v1 vs grpo 输出有差异）
    3. T2 仍为 0/70 (0%)——GRPO 在序列层学会偏好 `<tool_call>`，但无法突破 xgrammar FSM 的 token 级硬约束

1. 详细报告见：[GRPO 实验报告](lora-qwen-0612/doc/10-grpo-results.md)

2. 训练脚本见：**/root/0420-fz/lora-qwen-0612/scripts/06_train_grpo.py**

3. GRPO adapter 见：**/root/0420-fz/lora-qwen-0612/outputs/loraed_Qwopus3.6-35B-A3B-v1/adapter_grpo_v1/** (130MB)

---

# Plan B: Two-Pass 框架层绕过

- 状态：已实现并端到端验证通过

- 进度
    1. 设计并实现框架层透明 Two-Pass 执行方案
    2. Pass 1: tools=ON, response_format=OFF → 模型自由调工具
    3. Pass 2: tools=OFF, response_format=ON → 基于工具结果生成结构化 JSON
    4. 端到端验证：5-8 次工具调用 + 6 张 Blocks JSON 卡片

1. 设计文档见：[Plan B 设计文档](ea-aim-fz-qwen-0602/docs/fz-analysis/14-plan-b-design-doc-0608.md)

2. 代码见：**/root/0420-fz/ea-aim-fz-qwen-0602** 的 `AIPRD-317-response-format-json-B1` 分支

3. 核心改动：`_InnerAgent._deferred_response_format`, `_MAX_TOOL_STEPS=6`, `_plan_b_tools_phase`

4. 简化版 Two-Pass wrapper 见：**/root/0420-fz/lora-qwen-0612/lib/two_pass.py**

---

# A2 方案：JSON 内嵌 Tool Calls 绕过 FSM（已完成训练 + T2 突破验证）

- 状态：已完成训练 + T2=95% 验证通过，待框架层集成

- 进度
    1. 提出 A2 方案：将 tool calls 编码为 JSON `_tool_calls` 字段（数组格式），第一个 token 从 `<` 变为 `{`，FSM 允许
    2. xgrammar 验证通过：A2-JSON 格式 92/92 tokens ALL ACCEPTED by FSM
    3. 生成 30 条 A2 种子数据 + 8000 条 GPT-5.4-mini 合成训练数据
    4. 完成 Unsloth Studio SFT 训练
    5. ★ T2 测试突破：temp=0.3 + max_tokens=4096 条件下 A2-T2 = 19/20 = 95%（此前 6 轮实验 T2 均为 0%）
    6. T1 工具调用基线 95%，T3 无工具场景 5%——需增加 tools_optional 训练样本比例

1. 完整分析见：[A2 突破性结果文档](lora-qwen-0612/doc/16-a2-breakthrough-results.md)

2. 测试脚本见：**/root/0420-fz/lora-qwen-0612/scripts/10_test_a2_model.py**

3. 种子数据见：**/root/0420-fz/lora-qwen-0612/data/processed/a2_seed_data.json**

4. 8000 条合成数据见：**HF: FZSAMA/qwen-a2-constraint-tax-data**

5. 数据配方配置见：**/root/0420-fz/lora-qwen-0612/data/processed/a2_recipe_config.json**

6. 训练模型路径：**/root/0420-fz/lora-qwen-0612/outputs/loraed_Qwopus3.6-35B-A3B-a2-0623**

---

# 论文撰写

- 状态：论文已完成，根据审稿意见修改中

- 论文路径：**/root/0420-fz/CT-0617/**（main-en.tex、PDF、README）

- 待补充的实验见：[Paper 实验补充 v2](lora-qwen-0612/doc/12-paper-experiment-supplement-v2.md)

- 附录材料见：[测试设计与工具 Schema 定义](lora-qwen-0612/doc/14-appendix-test-design-and-tool-schema.md)

---

# 环境和配置

## 核心 conda 环境

| 环境名 | 用途 | 关键包 |
|------|------|------|
| `lora_qwen_0612` | 训练、数据生成 | torch 2.10, transformers 5.5, PEFT, Unsloth |
| `sglang_059` | SGLang 推理服务 | SGLang 0.5.9, xgrammar, flashinfer |
| `vllm_0605` | vLLM 推理服务 | vLLM 0.23.0, torch 2.11+cu126 |

## 已知环境问题

1. **tokenizer_config.json**: `tokenizer_class: TokenizersBackend` → 需改为 `Qwen2Tokenizer`（SGLang 启动前必须修复）
2. **vLLM CUDA 兼容**: vLLM 0.23.0 编译为 CUDA 13.0，driver 535 仅支持 12.6。设置 `export LD_LIBRARY_PATH=".../nvidia/cu13/lib:$LD_LIBRARY_PATH"` 可工作
3. **Unsloth 依赖冲突**: Unsloth 与 transformers 5.x 不兼容，如需用 vLLM+transformers 5.x 则移除 Unsloth
4. **SGLang LoRA**: 需要 patched config.json（补齐 `num_hidden_layers`, `intermediate_size` 等顶层属性）才能加载 Qwen3.5 MoE 的 LoRA adapter

## GPU 服务器

- **GPU**: 2× NVIDIA A800-SXM4-80GB
- **Driver**: 535.183.06, CUDA 12.6
- **SGLang 标准启动命令**见项目文档或 `doc/14-appendix-test-design-and-tool-schema.md §E`

---

# 文档索引

| 文档 | 路径 |
|------|------|
| 最终综合报告 | `lora-qwen-0612/doc/11-final-report-constraint-tax.md` |
| GRPO 实验报告 | `lora-qwen-0612/doc/10-grpo-results.md` |
| GRPO 技术文档 | `lora-qwen-0612/doc/09-grpo-technical-doc.md` |
| SFT 4轮分析 | `lora-qwen-0612/doc/08-stage2-analysis-next-steps.md` |
| A2 方案进度 | `lora-qwen-0612/doc/13-a2-progress.md` |
| Paper 实验补充 | `lora-qwen-0612/doc/12-paper-experiment-supplement-v2.md` |
| Paper 附录（测试+工具） | `lora-qwen-0612/doc/14-appendix-test-design-and-tool-schema.md` |
| Constraint Tax 跨模型验证 | `ea-aim-fz-qwen-0602/docs/fz-analysis/15-constraint-tax-final-report-0609.md` |
| Plan B Two-Pass 设计 | `ea-aim-fz-qwen-0602/docs/fz-analysis/14-plan-b-design-doc-0608.md` |

---

# HF 数据集

| 数据集 | URL |
|------|------|
| 6000 条 SFT 训练数据 | https://huggingface.co/datasets/FZSAMA/qwen-constraint-tax-training-data |
| 8000 条 A2 格式数据 | https://huggingface.co/datasets/FZSAMA/qwen-a2-constraint-tax-data |

---

# 联系人

- **李方正 (Fangzheng Li)** — 论文第一作者
- **张爱民 (Aimin Zhang)** — 对应作者
- **吕晨 (Chen Lv)** — 论文合作者
