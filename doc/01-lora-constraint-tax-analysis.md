# LoRA 微调解决 Constraint Tax — 方向分析与设计

> 2026-06-12 | 分析阶段，非最终实现代码

---

## 1. 问题重述

### 1.1 Constraint Tax 本质

```
正常行为（无 response_format）:
  用户询盘 → LLM 调 websearch/knowledge_base → 获取真实数据 → 回复

Constraint Tax 行为（有 response_format）:
  用户询盘 → LLM 直接输出 JSON → 伪造/空值填充 → 跳过工具
```

**关键诊断（实验 B 已证实）**：
- 模型"大脑"100% 知道该调工具（`need_search: true`）
- 但 `response_format` 在**执行层**压制了工具调用
- 这是**执行优先级排序错误**，不是能力缺失
- 训练数据中缺乏 `tools + schema 同时存在` 的联合场景覆盖

### 1.2 LoRA 微调目标

**不是教模型"该调哪个工具"**（能力已存在），而是纠正执行优先级：

```
有 Schema 约束时:
  旧行为: Schema 要 JSON → 直接填（跳过工具）
  新行为: Schema 要 JSON → 先调工具获取数据 → 基于真实数据填 JSON
```

---

## 2. 核心挑战

### 2.1 挑战一：训练数据从哪里来？

需要的是 **tools + response_format 同时存在、且模型正确执行"先工具后 JSON"** 的完整对话轨迹。

### 2.2 挑战二：训练数据用什么格式？

标准 SFT 格式是 `messages: [{role, content}]`，但工具调用场景需要：
- `assistant` 消息中的 `tool_calls` 字段
- `tool` 角色的工具返回消息

### 2.3 挑战三：训练-推理一致性

训练时 schema 在 system prompt 中，推理时 schema 通过 `response_format` API 参数传入。两者格式不同，LoRA 权重能否泛化？

---

## 3. 数据获取策略

### 3.1 方案概述：Teacher-Student 数据蒸馏

```
GPT-5.4-mini (Teacher)              Qwen3.6-35B (Student)
      │                                    │
  多样询盘场景                           LoRA SFT 训练
  + 真实工具定义                         │
  + JSON Schema 约束                     ▼
      │                         覆盖旧的"有 Schema → 跳工具"捷径
      ▼
  正确的"先工具 → 后 JSON"轨迹
      │
      ▼
  训练数据 (500-1000条)
```

### 3.2 数据生成流程

```
Step 1: 构造多样询盘场景（模板 + GPT 扩充）
  - 行业维度: LED照明、机械设备、纺织服装、消费电子、化工...
  - 地区维度: 北美、欧洲、中东、东南亚、南美、非洲
  - 买家类型: 品牌商、分销商、零售商、终端用户、代理商
  - 询盘复杂度: 简单RFQ、详细规格、多产品、附件+文字

Step 2: 用 GPT-5.4-mini 生成完整对话轨迹
  - 每次生成一条完整的多轮工具调用 + JSON 输出轨迹
  - GPT-5.4-mini 是唯一已验证的 Constraint Tax 豁免模型
  - 在 AEP 网关上调用，成本可控

Step 3: 质量校验
  - 必须包含 ≥1 次工具调用（在 JSON 输出之前）
  - 最终 JSON 必须通过 AgentResponse schema 校验
  - 工具调用参数必须合理（非空、非占位符）
  - 不合格的样本丢弃或修复

Step 4: 格式转换
  - 转为 messages 数组格式
  - 切分 train/eval (90/10)
```

### 3.3 训练数据格式设计

每个训练样本是一个**完整的多轮对话**，包含工具调用：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是贸探·跟单专家...\n\n## 可用工具\n- websearch: 搜索网络实时信息\n- knowledge_base: 查询外贸知识库\n- fetchurl: 抓取网页内容\n\n## 输出格式要求（最高优先级）\n你的最终回复必须是合法的 JSON，格式如下：\n{\"blocks\": [{\"type\": \"card\", \"card\": {...}}, ...]}\n禁止使用 ```json 围栏，禁止在 JSON 外输出文本。"
    },
    {
      "role": "user",
      "content": "请分析这封询盘:\n公司: TechTronics GmbH, 德国电子元器件分销商\n产品: USB-C Hub, 12-in-1, PD 100W, 4K@60Hz\n数量: 3000 units\n要求: DDP 报价, CE/RoHS 认证"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "websearch",
            "arguments": "{\"query\": \"TechTronics GmbH Germany electronics distributor\"}"
          }
        },
        {
          "id": "call_2",
          "type": "function",
          "function": {
            "name": "knowledge_base",
            "arguments": "{\"query\": \"USB-C Hub CE RoHS 认证出口德国要求\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "搜索结果: TechTronics GmbH 成立于 2012 年，总部位于慕尼黑，是专业的电子元器件分销商..."
    },
    {
      "role": "tool",
      "tool_call_id": "call_2",
      "content": "知识库: USB-C Hub 出口欧盟需 CE-EMC + CE-LVD + RoHS 2.0，建议做 GS 认证..."
    },
    {
      "role": "assistant",
      "content": "{\"blocks\":[{\"type\":\"card\",\"card\":{\"type\":\"expert_insight\",\"message\":\"AI 专家指点\",\"list\":[{\"avatarType\":\"business\",\"title\":\"业务专家\",\"description\":\"\",\"comment\":\"德国中型分销商，USB-C Hub 品类成熟，DDP 条款意味着他们对供应链稳定性要求极高。建议首轮彰显认证实力和交付能力。\"}]}},{\"type\":\"card\",\"card\":{\"type\":\"detail_card\",\"message\":\"买家画像\",\"list\":[{\"title\":\"公司\",\"description\":\"TechTronics GmbH\"},{\"title\":\"类型\",\"description\":\"电子元器件专业分销商\"},{\"title\":\"联系人\",\"description\":\"未提取\"},{\"title\":\"地区\",\"description\":\"德国慕尼黑\"},{\"title\":\"联系阶段\",\"description\":\"首次询价\"}]}}]}"
    }
  ]
}
```

**关键设计要点**：

| 设计点 | 说明 |
|--------|------|
| Schema 在 system prompt 中 | 训练时无法使用 `response_format` API 参数，将 JSON 格式要求写入 system prompt 末段 |
| `content: null` + `tool_calls` | assistant 消息在调工具时不输出文本（标准 OpenAI tool_call 格式） |
| `tool` 角色消息 | 工具返回结果，带 `tool_call_id` 与调用对应 |
| 最终 assistant 消息 | 直接输出 JSON 文本（无 markdown 围栏） |
| 多工具并行 | 首轮同时发出 websearch + knowledge_base（模拟生产 SOP） |

### 3.4 数据量估算

| 阶段 | 样本量 | 目标 |
|------|--------|------|
| 初始验证 | 200-500 条 | 验证方法可行性，观察工具率是否回升 |
| 稳定提升 | 1000-2000 条 | 达到工具率 ≥ 80% |
| 充分训练 | 3000+ 条 | 达到工具率 ≥ 95%，稳定泛化 |

**生成成本估算**（GPT-5.4-mini AEP 网关）：
- 每条轨迹 ~3-5K tokens (输入+输出)
- 500 条 ≈ 2-3M tokens ≈ 低成本
- 3000 条 ≈ 12-18M tokens ≈ 可控

### 3.5 训练数据多样性保证

为避免过拟合到特定场景，训练数据必须覆盖：

```
询盘场景分布（目标）:
├── LED 照明类: 15%
├── 消费电子类: 15%
├── 机械设备类: 10%
├── 纺织服装类: 10%
├── 化工材料类: 5%
├── 家居用品类: 10%
├── 汽车配件类: 5%
├── 包装印刷类: 5%
├── 医疗器械类: 5%
└── 其他行业: 20%

地区分布（目标）:
├── 北美 (US/CA): 20%
├── 欧洲 (DE/FR/UK/IT/ES...): 25%
├── 中东 (AE/SA/QA...): 15%
├── 东南亚 (IN/VN/TH/ID...): 15%
├── 南美 (BR/MX/AR...): 10%
├── 非洲 (NG/ZA/EG...): 10%
└── 大洋洲 (AU/NZ): 5%
```

---

## 4. 模型训练策略

### 4.1 训练方法选择

**当前阶段：SFT（Supervised Fine-Tuning）**

理由：
- DPO 需要构造 chosen/rejected 对话对，增加数据构造复杂度
- SFT 直接教模型正确行为序列，更直接
- 如果 SFT 效果不够，再追加 DPO 强化偏好

**训练方式：QLoRA（4-bit 量化 + LoRA）**

理由：
- Qwen3.6-35B-A3B 全量 ~70GB（FP16），单卡装不下
- 2×A800-80GB 使用 QLoRA 训练绰绰有余
- 参考平台已验证 QLoRA + Unsloth 在 A800 上可用

### 4.2 超参数设计

```python
# LoRA 配置
lora_r = 32              # 比默认 16 高，行为纠正需要更大秩
lora_alpha = 64          # alpha = 2×r
lora_dropout = 0.05      # 少量 dropout 防止过拟合
target_modules = [        # 覆盖所有线性层
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练配置
training_type = "qlora"  # 4-bit 量化 LoRA
learning_rate = 2e-4     # QLoRA 标准学习率
batch_size = 4           # 每 GPU batch（工具调用序列较长，需降低）
grad_accum = 8           # 梯度累积 → 有效 batch = 4 × 2GPU × 8 = 64
epochs = 3               # 初始验证 3 epoch，数据扩充后可降为 2
max_seq_len = 4096       # 工具调用轨迹较长，需要更大上下文
lr_scheduler = "cosine"  # 余弦退火
warmup_ratio = 0.05      # 5% warmup
optimizer = "adamw_8bit" # 8-bit AdamW 节省显存
```

### 4.3 Loss Masking 策略

只对 assistant 角色的 token 计算 loss：

```python
def tokenize_with_tool_calls(sample, tokenizer, max_length):
    """
    关键：assistant 消息有两种形式：
    1. content=null + tool_calls → 应该计算 loss（教模型"调工具"）
    2. content=JSON → 应该计算 loss（教模型"输出正确 JSON"）
    
    system / user / tool 消息 → label = -100（不计算 loss）
    """
    msgs = sample["messages"]
    
    # 使用 Qwen chat_template 编码
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    
    # 获取 tokenized
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    labels = [-100] * len(full_ids)
    
    # 只对 assistant 部分计算 loss
    # 通过 find_assistant_spans 定位每个 assistant 消息的 token span
    assistant_spans = find_assistant_spans(msgs, tokenizer)
    for start, end in assistant_spans:
        for i in range(start, min(end, max_length)):
            labels[i] = full_ids[i]
    
    return {
        "input_ids": full_ids[:max_length],
        "attention_mask": [1] * len(full_ids[:max_length]),
        "labels": labels[:max_length],
    }
```

**⚠️ 关键差异 vs 参考平台**：
- 参考平台的 `tokenize_sample` 只识别 `role == "assistant"` 的纯文本消息
- 我们的数据中 assistant 消息有两种形式：`tool_calls` 和 `content`
- 需要在 tokenize 后定位 assistant 部分的 token span
- **或者更简单的方法**：利用 Qwen chat_template 自然产生的 token 序列，通过 token ID 边界来定位 assistant 区域

### 4.4 Chat Template 兼容性验证

Qwen3.6-35B-A3B 的 chat_template 原生支持 tool_calls 格式。验证方法：

```python
# 在训练前验证 tokenizer 能正确编码 tool_calls 消息
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

test_msgs = [
    {"role": "system", "content": "You are an agent. Tools: websearch"},
    {"role": "user", "content": "Search for something"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "websearch", "arguments": '{"query": "test"}'}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "Results: ..."},
    {"role": "assistant", "content": '{"blocks": []}'},
]

text = tokenizer.apply_chat_template(test_msgs, tokenize=False)
print(text)  # 人工检查格式
ids = tokenizer.encode(text)
print(f"Token count: {len(ids)}")  # 确保 < max_seq_len
```

### 4.5 环境准备

当前环境状态：
- 模型路径：`/root/.cache/modelscope/hub/models/Jackrong/Qwopus3___6-35B-A3B-v1/`
- GPU：3× NVIDIA A800-SXM4-80GB（训练用 2 张即可）
- Python 环境：需**新建** conda env（现有 `us_` 环境无 unsloth，speed-test 环境有 CUDA 兼容问题）

```bash
# 新建训练环境
conda create -n lora_qwen_0612 python=3.11 -y
conda activate lora_qwen_0612

# 安装依赖
pip install unsloth
pip install torch torchvision torchaudio  # unsloth 会自动拉正确版本
pip install datasets trl peft accelerate
pip install bitsandbytes  # QLoRA 4-bit 量化依赖
pip install xformers --index-url https://download.pytorch.org/whl/cu124  # 可选，加速 attention
```

---

## 5. 训练-推理一致性问题（关键风险）

### 5.1 问题描述

| | 训练时 | 推理时 |
|---|---|---|
| Schema 传递方式 | 写在 system prompt 里 | 通过 `response_format` API 参数 |
| 模型看到的格式 | 自然语言：`你的输出必须是 JSON 格式：{"blocks": [...]}` | OpenAI json_schema：`{"type": "json_schema", "json_schema": {...}}` |
| 格式注入位置 | system prompt 尾部 | 由推理框架注入到 prompt |

**风险**：LoRA 学会的是"system prompt 中有 JSON 格式要求 → 调工具"，但推理时 schema 由 API 参数注入，格式不同可能导致 LoRA 不生效。

### 5.2 缓解策略

#### 策略 A：模拟推理框架的 Schema 注入格式（推荐）

SGLang/vLLM 在收到 `response_format` 时，会将 schema 以特定格式追加到 prompt。我们可以：
1. 实际调用一次 SGLang API，抓取它实际发给模型的完整 prompt
2. 在训练数据的 system prompt 中**复制相同的格式**

这样训练和推理的 prompt 格式一致，LoRA 权重可以直接泛化。

```python
# 在推理框架上发一条测试请求，截获实际 prompt
# 假设框架注入的格式类似：
"""
...
# Output Format
You MUST respond with a JSON object that matches the following JSON Schema:
{"type": "object", "properties": {"blocks": ...}, "required": ["blocks"]}
Do NOT include ```json fences or any text outside the JSON object.
"""
```

#### 策略 B：双阶段训练（精准打击）

```
阶段 1: 用 system prompt 中的 schema 训练 → 建立"有 Schema → 调工具"的联结
阶段 2: 用推理框架实际注入的格式训练 → 将联结迁移到 response_format 场景
```

阶段 2 的数据较少（因为需要实际调用 API 截获 prompt），但可以直接对齐推理格式。

#### 策略 C：Plan B 兼容设计（保底）

如果 LoRA 在 `response_format` API 参数下仍不生效，可以：
1. **推理时不传 `response_format`**，改为将 schema 放在 system prompt 中（和训练时一致）
2. 后续通过 `StreamingJSONParser` 或后处理校验 JSON 合法性
3. 这本质上是用训练数据格式替代 API 参数——效果等价，但需要微调 agent.py

### 5.3 验证方法

训练完成后，用现有的 T1/T2/T3 测试脚本对比：

```
T1 (tools=ON, rfmt=OFF):  工具率从 100% → 保持 100% ✅（验证 LoRA 没有破坏工具能力）
T2 (tools=ON, rfmt=ON):   工具率从 0% → 目标 ≥ 80%（验证 Constraint Tax 被纠正）
T3 (tools=OFF, rfmt=ON):  JSON 率保持 ≥ 90%（验证 JSON 输出能力没有退化）
```

---

## 6. 项目结构设计

```
/root/0420-fz/lora-qwen-0612/
│
├── doc/                                # 文档
│   ├── 01-lora-constraint-tax-analysis.md   # ← 本文档
│   ├── 02-data-generation-design.md         # 数据生成详细设计（待写）
│   ├── 03-training-log.md                   # 训练记录（待写）
│   └── reference/
│       └── lora-plantform/                  # 参考平台（已有）
│           └── fz_lora_studio_/
│
├── data/                               # 数据目录
│   ├── raw/                            # 原始生成数据
│   │   └── gpt_traces/                # GPT-5.4-mini 生成的轨迹
│   ├── processed/                      # 处理后的训练数据
│   │   ├── train.json                 # SFT 训练集
│   │   └── eval.json                  # 评估集
│   └── templates/                      # 询盘场景模板
│       └── inquiry_templates.json     # 多样询盘模板库
│
├── scripts/                            # 脚本
│   ├── 01_generate_training_data.py   # 数据生成脚本（调 GPT-5.4-mini）
│   ├── 02_validate_data.py            # 数据质量校验
│   ├── 03_train_sft.py                # SFT 训练脚本
│   └── 04_evaluate.py                 # 评估脚本（T1/T2/T3 测试）
│
├── outputs/                            # 训练产出
│   └── loraed_Qwopus3.6-35B-A3B/      # LoRA adapter 输出
│       ├── adapter_sft_500/           # 阶段 1 adapter
│       ├── adapter_sft_1000/          # 阶段 2 adapter
│       └── ...
│
└── config.py                           # 全局配置（路径、超参数）
```

---

## 7. 关键决策点

### 7.1 待验证假设

| # | 假设 | 验证方法 | 风险 |
|---|------|---------|------|
| H1 | Qwen3.6-35B 的 chat_template 支持 tool_calls 格式 | 用 tokenizer 编码一条含 tool_calls 的消息，检查输出 | 低：Qwen 官方文档确认支持 |
| H2 | SFTTrainer 可以处理含 tool_calls 的多轮对话 | 用 10 条样本跑一个 epoch，观察 loss 下降 | 中：SFTTrainer 主要面向纯文本，需验证 |
| H3 | LoRA 在 system prompt 中学到的行为可以泛化到 response_format API 参数 | 训练后用 T1/T2/T3 测试框架对比 | **高：这是整个方案最大的不确定性** |
| H4 | 500 条数据足以看到工具率回升 | 先训 500 条，跑 T2 测试 | 中：可能需要更多数据 |

### 7.2 止损线

| 里程碑 | 成功标准 | 如果失败 |
|--------|---------|---------|
| 数据生成 | 生成 500 条合格轨迹 | 改用真实 EvoAgent 会话提取 + 人工审核 |
| 初始训练 (500条) | T2 工具率 ≥ 30% | 检查数据质量、尝试 DPO、调整超参 |
| 扩充训练 (1000条) | T2 工具率 ≥ 60% | 采用策略 C（schema 放 system prompt，不用 API 参数） |
| 最终训练 (2000条) | T2 工具率 ≥ 80% | 放弃 LoRA，确认 Plan B 作为生产方案 |

---

## 8. 下一步行动

### 第一阶段：环境+验证（1-2 天）

1. **创建 conda 环境** — `lora_qwen_0612`，安装 unsloth + trl + peft
2. **验证 chat_template 兼容性** — 确认 Qwen3.6 tokenizer 正确处理 tool_calls 消息
3. **10 样本 SFT 冒烟测试** — 用最小数据集跑通训练流程
4. **截获推理框架实际 prompt** — 调一次 SGLang/vLLM API，获取框架注入 schema 后的完整 prompt

### 第二阶段：数据生成（1-2 天）

5. **设计询盘场景模板库** — 覆盖 10+ 行业 × 7+ 地区 × 3+ 买家类型
6. **实现数据生成脚本** — 调 GPT-5.4-mini 批量生成训练轨迹
7. **数据质量校验** — 自动检查工具调用率和 JSON 合法性

### 第三阶段：训练+评估（2-3 天）

8. **SFT 训练（500 条）** — 初始验证
9. **T1/T2/T3 评估** — 测量工具率和 JSON 率
10. **根据结果决定** — 扩充数据 / 调整策略 / 启用 DPO

---

## 附录 A：与参考平台的差异

| 维度 | 参考平台 (fz_lora_studio) | 本项目 |
|------|--------------------------|--------|
| 数据格式 | 纯文本 Q&A (`messages: [{user, assistant}]`) | **多轮工具调用** (`messages` 含 `tool_calls` + `tool` 角色) |
| 训练目标 | 通用问答能力 | **执行优先级纠正**（有 Schema 时先调工具） |
| 可视化 | Web UI + 实时图表 | **纯脚本**，无 Web 界面 |
| 评估方式 | Loss 曲线 | **T1/T2/T3 工具率 + JSON 率** |
| 基座模型 | Qwen3.6-35B-A3B | 同 |
| 框架 | Unsloth + SFTTrainer | 同 |
| GPU | 2×A800-80GB | 同（可用第 3 张做更大 batch） |

## 附录 B：生产工具定义（训练数据中使用）

```python
TOOLS_FOR_TRAINING = [
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "Search the web for real-time information about companies, markets, and industries",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_base",
            "description": "Query foreign trade knowledge base for industry standards, trade policies, and certification requirements",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Knowledge base query"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetchurl",
            "description": "Fetch and extract content from a URL (company website, product page, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "analyze": {"type": "boolean", "default": False}
                },
                "required": ["url"]
            }
        }
    },
]
```

## 附录 C：AgentResponse Schema 精简版（训练 system prompt 中使用）

为控制训练数据的 system prompt 长度，使用精简版 schema 描述（而非完整的 167 行 Pydantic 定义）：

```
## 输出格式（最高优先级）

你的最终回复必须是合法的 JSON 对象：
{"blocks": [<block>, ...]}

每个 block 是以下之一：
- text block:  {"type":"text","content":"<markdown>"}
- card block:  {"type":"card","card":{"type":"<卡片类型>","message":"...","list":[...]}}

卡片类型: expert_insight | detail_card | ai_judgement | highlight_block | collapsible | inquiry_collapsible | suggested_questions | radio_select | checkbox_select | upload_request | product_images | report_preview

硬约束：
- 禁止 ```json 围栏
- 禁止在 JSON 外输出文本
- blocks 按顺序渲染
```
