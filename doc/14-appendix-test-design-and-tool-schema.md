# Appendix: Test Case Design and Tool/Schema Definitions

> 论文 `Constraint Tax in Open-Weight LLMs` 附录材料

---

## A. 测试用例设计

### A.1 T1/T2/T3 受控实验设计

每个受试模型在三种条件下独立测试：

| 条件 | tools | response_format | 测试目标 |
|------|:-----:|:---------------:|------|
| T1 (基线) | ON | OFF | 测量基线工具调用能力 |
| T2 (联合约束) | ON | ON | **检测 Tool Suppression** |
| T3 (Schema 对照) | OFF | ON | 验证独立 Schema 合规能力 |

### A.2 标准测试脚本协议

所有跨模型测试使用统一的 Python 测试脚本，位于两个仓库：

**核心测试脚本** (`tests/`):

| 脚本 | 代码仓库 | 测试对象 |
|------|---------|------|
| `test_constraint_tax_lora.py` | `lora-qwen-0612` | Qwen3.6-35B-A3B LoRA 微调模型 |
| `test_cloud_models.py` | `ea-aim-fz-qwen-0602` | GPT-5.4-mini, Qwen3.5-397B-A17B, Qwen3-VL-235B |
| `test_122b_tool_response_format.py` | `ea-aim-fz-qwen-0602` | Qwen3.5-122B-A10B |
| `test_gptoss_tool_rfmt.py` | `ea-aim-fz-qwen-0602` | GPT-OSS-20B |
| `test_vllm_qwen35b.py` | `ea-aim-fz-qwen-0602` | Qwen3.6-35B-A3B via vLLM |
| `test_nemotron_tool_rfmt.py` | `ea-aim-fz-qwen-0602` | NVIDIA Nemotron 3 Super |
| `test_235b_tool_rfmt.py` | `ea-aim-fz-qwen-0602` | Qwen3-235B-A22B |
| `test_b1_two_pass.py` | `ea-aim-fz-qwen-0602` | Plan B Two-Pass 端到端验证 |

**标准测试参数**:

```python
# 所有跨模型测试的统一参数
ROUNDS  = 5          # 每个条件独立测试 5 轮
MODEL   = "<受试模型>"
API_BASE = "http://127.0.0.1:8082/v1"  # 或其他端点
API_KEY  = "EMPTY"   # 或对应云服务 key
temperature      = 0.5
stream           = True
max_completion_tokens = 4096
```

**标准化检测逻辑** (所有脚本共享):

```python
# 双重检测机制: API 结构化 tool_calls + 文本级检查
async for chunk in resp:
    if d.tool_calls:      # ← API 级: 解析 streaming delta 中的 tool_calls 结构
        for tc in d.tool_calls:
            # 累积 tool_call name 和 arguments
            tcs.append({"idx": tc.index, "name": tc.function.name, "args": tc.function.arguments})
    if d.content:          # ← 文本级: 同时收集 content（用于 JSON 合规检查）
        content += d.content
```

### A.3 任务多样性

**正式跨模型测试 (论文 Table 4 的 9 模型矩阵)**:

标准测试脚本使用固定的单任务 Prompt（1 个公司 × 1 个产品场景），每条件重复 5 次：

```text
System: 你是外贸询盘分析助手。收到客户询盘后:
        1.使用 websearch 搜索买家公司背景
        2.使用 knowledge_base 查询产品行业标准
        3.基于调研结果给出分析

User:   请分析这封询盘:
        公司: BrightLight Inc. 美国照明产品进口商
        产品: LED strip lights, IP65 waterproof, 5050 SMD, RGB+W, 5m/reel
        数量: 2000 reels
        要求: FOB报价, UL listed
```

**限制**: 标准跨模型测试使用同一任务重复采样，仅能证明该任务上的稳定失败。论文图 4 的 T2=0% 对每个模型均建立在 5 次重复测试上。

> ⚠️ **重要说明**: 该测试设计足以支撑"发现 Constraint Tax 现象"——同一任务上 9 模型中的 8 个开放权重模型均稳定失败，同时闭源模型稳定成功，已构成强证据。但不能支撑"all open-weight LLMs systematically fail"这一泛化表述——该表述应限定为"all **8 tested** open-weight models"。如需泛化到更多任务，需要多任务 benchmark（见 §A.3 扩展任务多样性）。**当前论文的 9 模型矩阵作为现象发现是充分的；读者应理解该现象是否泛化到其他任务需要通过扩展测试验证。**

**扩展任务多样性 (本项目额外测试)**:

除标准测试脚本外，项目额外使用以下多任务 Prompt 集验证结果一致性：

**数据集 1: Tool Mandatory (200 prompts)**

```
覆盖 10 家公司 × 8 个合规市场 × 正/负样本变体:
  - 公司: TechCorp Inc., GlobalTrade Ltd., MediSupply Co.,
          GreenEnergy Solutions, PacificBridge Corp., NordicSteel AB,
          Sunrise Electronics, AlpineFoods GmbH, BlueOcean Logistics,
          SmartParts Inc.
  - 合规市场: EU, US, Middle East, Southeast Asia, Australia,
              South America, Africa, Generic
  - 每 prompt 要求 websearch + knowledge_base 双工具调用
  - 200 prompts 中 160 正样本 (工具返回数据) + 40 负样本 (部分空)
```

**数据集 2: 合成场景库 (30 seed scenarios)**

```
覆盖 10 行业 × 7 地区 × 3 买家类型:
  - 行业: LED 照明, 消费电子, 机械设备, 纺织服装, 化工材料,
          家居用品, 汽车配件, 医疗器械, 太阳能, 五金工具
  - 地区: 西欧, 北美, 中东, 东南亚, 南美, 非洲, 大洋洲
  - 买家: 品牌商, 分销商, 零售商/终端用户
```

**数据集 3: 规模合成 (6000 prompts)**

```
GPT-5.4-mini 生成, 覆盖上述全部行业/地区/买家类型组合
- 6000 条: 传统 tool_call → JSON 格式 (SFT 第 6 轮使用)
```

**论文章节建议**:

- 正式 9 模型矩阵的 T2=0% 基于单任务 5 次测试
- 对 Qwen3.6-35B-A3B 额外进行了 200+ 次跨任务 T2 测试，结果一致为 0%
- 建议在论文中说明: "For Qwen3.6-35B-A3B, we extended testing to 200+ queries across 10 diverse company profiles and 8 compliance markets, all yielding identical T2=0% results."

---

## B. 工具定义

### B.1 生产环境工具集 (config.py)

三个外部工具，与生产 Agent 系统完全相同：

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "搜索网络实时信息，用于调查买家公司背景、市场环境、行业动态",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_base",
            "description": "查询外贸知识库，获取行业标准、贸易政策、认证要求等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "知识库查询关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetchurl",
            "description": "抓取网页内容，用于获取买家官网信息（公司规模、主营业务等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要抓取的网页 URL"},
                    "analyze": {
                        "type": "boolean",
                        "description": "是否触发竞品深度分析（默认 false）",
                        "default": False
                    }
                },
                "required": ["url"]
            }
        }
    }
]
```

### B.2 跨模型测试工具集 (精简版)

跨模型 9-model 标准测试仅使用 2 个工具（websearch + knowledge_base），省略 fetchurl：

```python
# 用于所有跨模型标准测试的简化工具集
TOOLS = [
    {"type": "function", "function": {
        "name": "websearch",
        "description": "Search the web for real-time information about companies and markets",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "knowledge_base",
        "description": "Query foreign trade knowledge base for industry standards",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
]
```

> **设计说明**: 两个工具参数签名完全一致（均为 `{"query": string}`），有意简化以排除工具参数复杂度对 Tool Suppression 的潜在干扰。

---

## C. Response Format Schema 定义

### C.1 跨模型测试 Schema (4-field)

用于论文 Table 4 的所有 9 模型标准测试：

```json
{
    "type": "json_schema",
    "json_schema": {
        "name": "inquiry_analysis",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "buyer_background": {"type": "string"},
                "product_analysis": {"type": "string"},
                "recommendations": {"type": "string"},
                "key_findings": {"items": {"type": "string"}, "type": "array"}
            },
            "required": ["buyer_background", "product_analysis", "recommendations", "key_findings"],
            "additionalProperties": false
        }
    }
}
```

### C.2 Tool Mandatory 测试 Schema (3-field 极简)

用于 SFT 消融实验和 GRPO 训练：

```json
{
    "type": "json_schema",
    "json_schema": {
        "name": "company_info",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "company_info": {"type": "string"},
                "compliance_notes": {"type": "string"}
            },
            "required": ["company_name", "company_info", "compliance_notes"],
            "additionalProperties": false
        }
    }
}
```

### C.3 生产级 Schema (4-card + tool_dependency)

用于 EvoAgent 实际部署的完整 Schema：

```json
{
    "type": "json_schema",
    "json_schema": {
        "name": "inquiry_analysis",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "object", "properties": {"type": {"const": "text"}, "content": {"type": "string"}}},
                            {"type": "object", "properties": {"type": {"const": "card"}, "card": {"type": "object"}}}
                        ]
                    }
                },
                "tool_dependency": {
                    "type": "object",
                    "properties": {
                        "required": {"type": "boolean"},
                        "tools_used": {"type": "array", "items": {"type": "string"}},
                        "claims": {"type": "array"},
                        "reason": {"type": "string"}
                    }
                }
            },
            "required": ["blocks"],
            "additionalProperties": true
        }
    }
}
```

卡片结构:
- `expert_insight`: `message` + `list[{avatarType, title, description, comment}]`
- `detail_card (买家画像)`: `message` + `list[{title, description}]` (5 项: 公司/类型/联系人/地区/联系阶段)
- `detail_card (需求摘要)`: `message` + `list[{title, description}]` (5 项: 产品/采购量/交期/核心关注/匹配程度)
- `ai_judgement`: `title` + `description` + `list[{label, value, level}]`

---

## D. Prompt 和 System Prompt 示例

### D.1 跨模型标准测试 System Prompt (中文)

```
你是外贸询盘分析助手。收到客户询盘后:
1.使用 websearch 搜索买家公司背景
2.使用 knowledge_base 查询产品行业标准
3.基于调研结果给出分析
```

### D.2 GRPO/SFT 训练 System Prompt (英文, 强工具指令)

```
You are an information extraction assistant. To answer any question,
you MUST first use websearch and knowledge_base to retrieve information.
Then output ONLY a JSON object containing the retrieved facts.
If a tool returns no results, output "unknown" for that field.
```

### D.3 tool_choice 参数

- 标准跨模型测试: **未设置 `tool_choice`**（默认 `auto`）
- tool_choice 消融实验 (第 3.2 节 "已排除因素"): 测试了 `tool_choice="required"` 和 `named`，结果模型在 `response_format` 激活时仍不调工具（"模型完全冻结"）

---

## E. 推理框架与 Tool Call Parser 配置

### E.1 SGLang 配置

```bash
python -m sglang.launch_server
    --model-path <path>
    --served-model-name <name>
    --port 8082 --host 0.0.0.0
    --tp-size 2                          # 2× A800 80GB
    --mem-fraction-static 0.85
    --max-total-tokens 128144
    --max-running-requests 64
    --chunked-prefill-size 8192
    --enable-flashinfer
    --log-level warning
    --reasoning-parser qwen3             # 推理内容解析器
    --tool-call-parser qwen3_coder       # ★ 工具调用解析器 (Qwen3 XML→OpenAI format)
    --trust-remote-code
```

### E.2 vLLM 配置

```bash
python -m vllm.entrypoints.openai.api_server
    --model <path>
    --tensor-parallel-size 2
    --gpu-memory-utilization 0.85
    --max-model-len 4096                 # 多模型共享限制
```

### E.3 Tool Call 检测机制

双重检测确保不遗漏或误报：

```
检测方式 1: API 结构化 tool_calls
  解析 streaming delta 中的 tool_calls 数组
  包含 tool_call name + arguments

检测方式 2: Content 文本级检查
  解析 content 中是否包含 "<tool_call>" XML 标签
  覆盖 API 层 parser 可能遗漏的边缘情况
```

> **Parser 一致性**: 在 200+ 次 T2 queries 上，两种检测方式结果完全一致（均检测到 0 次 tool_call），排除了 "parser 漏抓 tool_call delta" 的可能性。

---

## F. 实验环境

| 组件 | 配置 |
|------|------|
| **GPU** | 2× NVIDIA A800-SXM4-80GB |
| **NVIDIA Driver** | 535.183.06 |
| **CUDA** | 12.6 |
| **SGLang** | 0.5.9 |
| **vLLM** | 0.22.0 (跨模型对比) |
| **Tool Parser** | SGLang: `qwen3_coder`, vLLM: `qwen3_coder` / `hermes` |
| **Guided Decoding** | xgrammar (SGLang), outlines (vLLM) |
| **测试环境** | Python 3.12/3.13, conda env `sglang_059` / `vllm_0605` / `lora_qwen_0612` |
| **开源测试脚本** | `lora-qwen-0612/tests/`, `ea-aim-fz-qwen-0602/tests/fz-qwen-test/` |
