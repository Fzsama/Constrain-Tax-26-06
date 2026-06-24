# Unsloth Studio Data Recipe 设计指南

> 配合 `data/processed/seed_for_unsloth_studio.json`（30条结构化种子）使用

## 输入文件

`seed_for_unsloth_studio.json` — 30 条种子，可用字段：

| 字段 | 说明 | Recipe 引用 |
|------|------|------------|
| `seed_id` | 种子编号 | `{{ seed_id }}` |
| `industry` | 行业 | `{{ industry }}` |
| `region` | 地区 | `{{ region }}` |
| `country` | 国家 | `{{ country }}` |
| `buyer_type` | 买家类型 | `{{ buyer_type }}` |
| `complexity` | 复杂度 | `{{ complexity }}` |
| `company` | 公司名 | `{{ company }}` |
| `product` | 产品 | `{{ product }}` |
| `quantity` | 数量 | `{{ quantity }}` |
| `inquiry_full` | 完整询盘文本 | `{{ inquiry_full }}` |

---

## GPT 诊断的核心问题（Recipe 要解决的）

1. **Tool → Reasoning 无因果绑定**：final JSON 不引用 tool 结果 → 模型学到"工具是装饰"
2. **Tool 结果质量低**：KB 空结果、websearch 返回无关页面 → 反向强化"不调工具更好"
3. **Schema 过重**：6 张卡片中有非必要卡片 → 模型学"拼装卡片"而非"推理"
4. **无 evidence grounding**：claim 是断言，不是基于证据

---

## Recipe 设计

### Block 1: Seed

```
类型: Seed
数据源: 本地 JSON → seed_for_unsloth_studio.json
```

### Block 2: Sampler（场景多样性增强）

手动添加 2 个 sampler 列，为每个种子产生变体：

```
列名: variation_type
值: ["same_industry_new_region", "same_region_new_industry", "same_industry_new_buyer"]
```

```
列名: tool_strategy
值: ["search_then_analyze", "verify_then_conclude", "cross_reference_sources"]
```

### Block 3: LLM — 生成训练对话（核心）

```
模型: GPT-5.4-mini（API）或本地 Qwen2.5-72B-Instruct
Temperature: 0.7
Max tokens: 8192
```

**System Prompt:**

```
你是训练数据生成器。为一个外贸询盘场景生成完整的 AI 助手对话轨迹。

## 角色
外贸跟单专家 AI 助手。收到客户询盘后，必须先用工具搜索外部信息，
再基于工具返回的真实数据输出结构化分析报告。

## 可用工具
1. websearch(query) — 搜索买家公司背景、市场环境
2. knowledge_base(query) — 查询外贸知识库（行业标准、认证、关税）
3. fetchurl(url) — 抓取买家官网

## 输出格式
你的输出必须是 JSON：
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."}, ...]}

## 🔴 最关键规则（违反即为不合格）

### 规则1: Tool Grounding（强制因果绑定）
最终输出的每一条分析结论，必须能用工具返回数据证明。
如果工具没查到信息，标注"未查到：[工具名]返回无相关信息"。
禁止在没有工具证据时编造任何公司名、人名、业务判断。

### 规则2: Evidence Linking（显式证据链）
在 expert_insight.comment 和 ai_judgement.description 中，
必须引用工具证据。格式：
  "根据 websearch 结果，该公司成立于2012年..." ✅
  "该买家品牌定位高端..." ❌（无证据来源）

### 规则3: Empty Tool Handling（空结果处理）
- 工具返回空/错误 → 最终 JSON 中标注"信息不足"
- 禁止在工具无结果时仍然输出详细分析
- 禁止用模型知识编造补充信息

### 规则4: Tool-First Sequence（先工具后JSON）
必须先有 ≥1 个 tool_call，才能输出 final JSON。
禁止在任何 tool_call 之前输出 JSON。

### 规则5: No Schema Ornament（简化输出）
只输出以下卡片类型：
  expert_insight → detail_card(买家画像) → detail_card(需求摘要) → ai_judgement
去掉 suggested_questions。
```

**User Prompt:**

```
## 场景信息
- 行业: {{ industry }}
- 地区: {{ region }} ({{ country }})
- 买家类型: {{ buyer_type }}
- 复杂度: {{ complexity }}
- 策略模式: {{ tool_strategy }}

## 询盘内容
{{ inquiry_full }}

请生成完整的 AI 助手对话轨迹。记住：
1. 先调 websearch 查公司背景
2. 如果询盘中有官网，调 fetchurl 抓取
3. 调 knowledge_base 查目标国贸易政策
4. 基于工具返回的真实数据输出 JSON
5. 工具无结果时标注"未查到"而非编造
```

### Block 4: Validator — Python 代码校验

```python
import json

def validate(output: str) -> bool:
    """校验生成的对话轨迹是否符合 Constraint Tax 纠正要求。"""
    try:
        data = json.loads(output)
        msgs = data.get("messages", [])
    except:
        return False
    
    # 1. 必须有 tool_call
    tool_calls = []
    for msg in msgs:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])
    if not tool_calls:
        return False  # 无工具调用 → 不合格
    
    # 2. tool_call 必须在 JSON 输出之前
    first_tool_idx = -1
    first_json_idx = -1
    for i, msg in enumerate(msgs):
        if msg.get("role") == "assistant":
            if msg.get("tool_calls") and first_tool_idx == -1:
                first_tool_idx = i
            if not msg.get("tool_calls") and msg.get("content", "").strip().startswith("{"):
                if first_json_idx == -1:
                    first_json_idx = i
    if first_json_idx > 0 and first_tool_idx >= first_json_idx:
        return False  # JSON 在 tool 之前 → 不合格
    
    # 3. 最终 JSON 不能有 suggested_questions
    for msg in msgs:
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            content = msg.get("content", "")
            if "suggested_questions" in content:
                return False  # 包含不必要的卡片类型
    
    # 4. 检查是否在编造（禁止出现占位符/伪造标记）
    for msg in msgs:
        content = msg.get("content", "") or ""
        if "Simulated" in content and "Search" in content:
            return False
        if "websearch" in content.lower() and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            return False  # 把工具名写成字段值
    
    return True
```

### Block 5: Expression（后处理格式化）

```jinja2
{
  "messages": {{ output.messages | tojson }}
}
```

---

## 目标产出

- **目标数量**：500-1000 条（recipe 自动生成）
- **每条格式**：`{"messages": [...]}` — 标准 OpenAI messages 格式
- **质量要求**：validator 通过率 ≥ 80%

## 与现有数据对比

| 维度 | 现有 106 条 | Recipe 修复后 |
|------|-----------|-------------|
| Tool grounding | ❌ 5/6 卡片无 tool 引用 | ✅ 每张卡片有证据来源 |
| 空结果处理 | ❌ 空 KB 仍输出详细分析 | ✅ 标注"未查到" |
| Schema | ❌ 6 张卡片（含 suggested_questions） | ✅ 4 张核心卡片 |
| Evidence chain | ❌ claim → assertion | ✅ claim → tool evidence |
| 多样性 | 🟡 10 行业 × 7 地区 | ✅ Sampler 自动扩展 |

---

## 操作步骤

1. 将 `seed_for_unsloth_studio.json` 复制到 Unsloth Studio 机器
2. 在 Studio 中创建新 Data Recipe
3. 添加 Seed block → 选择本地 JSON 文件
4. 添加 Sampler block → 配置 `variation_type` 和 `tool_strategy`
5. 添加 LLM block → 粘贴上述 System/User Prompt，配置 GPT-5.4-mini API
6. 添加 Validator block → 粘贴 Python 校验代码
7. Preview 测试 3-5 条输出
8. 确认质量后 Run 全量生成（500-1000 条）
9. 导出结果 JSON → 传回本项目的 `data/generated_unsloth/`
