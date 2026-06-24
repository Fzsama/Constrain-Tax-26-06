# Unsloth Studio 数据配方操作流程

> 配合 `data/processed/synthetic_seeds_for_studio.json`（37条种子）使用

---

## 一、种子数据特征速览

给 Studio 的 37 条数据，每条结构：

```
{
  "meta": { "industry": "LED 照明", "region": "西欧", "requires_tools": "true", ... },
  "messages": [
    {"role":"system",  "content":"你是贸探·跟单专家..."},
    {"role":"user",    "content":"请分析这封询盘: ..."},
    {"role":"assistant","content":"","tool_calls":[
      {"id":"call_1","type":"function","function":{"name":"websearch","arguments":"{...}"}},
      {"id":"call_2","type":"function","function":{"name":"knowledge_base","arguments":"{...}"}}
    ]},
    {"role":"tool",    "tool_call_id":"call_1","name":"websearch","content":"搜索结果..."},
    {"role":"tool",    "tool_call_id":"call_2","name":"knowledge_base","content":"知识库结果..."},
    {"role":"assistant","content":"{\"blocks\":[...],\"tool_dependency\":{...}}"}
  ]
}
```

**关键特征**（Recipe 要模仿的）：
- assistant 先输出 tool_calls，不输出 content
- tool 返回后，assistant 才输出最终 JSON
- 最终 JSON 含 `blocks`（4张卡片）+ `tool_dependency`（required + claims + evidence）

---

## 二、Recipe 搭建步骤

### Step 1：导入种子数据

1. 打开 Unsloth Studio → **Data Recipe** → **New Recipe**
2. 添加 **Seed** block
3. 数据源选择 **本地 JSON 文件** → 上传 `synthetic_seeds_for_studio.json`
4. Studio 会自动解析 JSON 结构，识别出 `meta` 和 `messages` 字段
5. 点击 **Preview** 确认字段识别正确

### Step 2：添加 Category block（多样性控制）

添加 **4 个 Category** block，用于控制生成数据的多样性。每个 Category 的 `Field name` 可在 LLM prompt 中用 `{{ field_name }}` 引用。

#### Category 1: industry

```
Field name: industry
Values（每行一个，回车添加）:
  LED 照明
  消费电子
  机械设备
  纺织服装
  化工材料
  家居用品
  汽车配件
  医疗器械
  太阳能
  五金工具
Keep out of final dataset: ✅（不导出到最终数据）
```

#### Category 2: region

```
Field name: region
Values:
  西欧
  北美
  中东
  东南亚
  南美
  非洲
  大洋洲
Keep out of final dataset: ✅
```

#### Category 3: buyer_type

```
Field name: buyer_type
Values:
  品牌商
  分销商
  零售商/终端用户
Keep out of final dataset: ✅
```

#### Category 4: required_mode

```
Field name: required_mode
Values:
  tools_required
  tools_optional
Weights (optional): 75, 25
Keep out of final dataset: ✅
```

> **说明**：Category 按权重随机采样。4 个 Category 组合会产生 `10 × 7 × 3 × 2 = 420` 种场景变体。勾选 "Keep out of final dataset" 表示这些字段只用于生成过程（作为 prompt 变量），不写进最终导出的训练数据。

### Step 3：添加 Model preset（模型配置）

在 AI generation → Setup 中添加 **Model preset**：

```
Provider: OpenAI Compatible API
Base URL: https://aep.focusaim.com/llm_openai_aic_schedule/v1
API Key: (你的 AEP API Key)
Model: gpt-5.4-mini
```

> 这个是全局模型配置，后续 AI structured data block 引用它。

### Step 4：添加 AI structured data（核心生成）

选择 **AI structured data** block（不是 AI text——我们需要强制 JSON 输出）。

**基本设置**：
```
Field name: llm_structured_1
Model preset: (选择 Step 3 配置的 gpt-5.4-mini)
Keep out of final dataset: 不勾（这是我们要的输出）
```

**Response format**（严格约束输出结构，防止冗余字段和卡片格式漂移）：
```json
{
  "messages": [
    {
      "role": "system",
      "content": ""
    },
    {
      "role": "user",
      "content": ""
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "websearch",
            "arguments": "{\"query\":\"...\"}"
          }
        },
        {
          "id": "call_2",
          "type": "function",
          "function": {
            "name": "knowledge_base",
            "arguments": "{\"query\":\"...\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "",
      "tool_call_id": "call_1",
      "name": "websearch"
    },
    {
      "role": "tool",
      "content": "",
      "tool_call_id": "call_2",
      "name": "knowledge_base"
    },
    {
      "role": "assistant",
      "content": "{\"blocks\":[{\"type\":\"card\",\"card\":{\"type\":\"expert_insight\",\"message\":\"AI 专家指点\",\"list\":[{\"avatarType\":\"business\",\"title\":\"业务专家\",\"description\":\"\",\"comment\":\"...\"}]}},{\"type\":\"card\",\"card\":{\"type\":\"detail_card\",\"message\":\"买家画像\",\"list\":[{\"title\":\"公司\",\"description\":\"...\"},{\"title\":\"类型\",\"description\":\"...\"},{\"title\":\"联系人\",\"description\":\"...\"},{\"title\":\"地区\",\"description\":\"...\"},{\"title\":\"联系阶段\",\"description\":\"...\"}]}},{\"type\":\"card\",\"card\":{\"type\":\"detail_card\",\"message\":\"需求摘要\",\"list\":[{\"title\":\"产品\",\"description\":\"...\"},{\"title\":\"采购量\",\"description\":\"...\"},{\"title\":\"交期\",\"description\":\"...\"},{\"title\":\"核心关注\",\"description\":\"...\"},{\"title\":\"匹配程度\",\"description\":\"...\"}]}},{\"type\":\"card\",\"card\":{\"type\":\"ai_judgement\",\"title\":\"AI 关键判断\",\"description\":\"...\",\"list\":[{\"label\":\"买家意向\",\"value\":\"...\",\"level\":\"success\"},{\"label\":\"询盘真实性\",\"value\":\"...\",\"level\":\"success\"},{\"label\":\"交易风险\",\"value\":\"...\",\"level\":\"warning\"}]}}],\"tool_dependency\":{\"required\":true,\"tools_used\":[\"websearch\",\"knowledge_base\"],\"claims\":[{\"id\":\"claim_001\",\"claim\":\"...\",\"source\":\"websearch\",\"if_missing\":\"cannot_determine\"}],\"reason\":\"...\"}}"
    }
  ]
}
```

> **说明**：上面的 Schema 直接用实际数据示例做模板（而非描述性文字），Studio 会按这个结构严格约束。card 的 list 项固定用 `title`/`description`（detail_card）或 `label`/`value`/`level`（ai_judgement）。最后一个 assistant 的 content 是完整的最终 JSON 字符串（含 blocks + tool_dependency）。

**Prompt**（去掉 Jinja2 `{% if %}` 语法，改为直接指令）：
```
基于下面的输入，生成一个外贸 AI 助手的完整对话轨迹。

## 场景信息
- 行业: {{ industry }}
- 地区: {{ region }}
- 买家类型: {{ buyer_type }}
- 生成模式: {{ required_mode }}

## 种子参考格式
{{ messages }}

## 规则
1. 最终 JSON 只能包含 4 张卡片：expert_insight → detail_card(买家画像) → detail_card(需求摘要) → ai_judgement
2. 每张卡片的结论必须引用工具数据（"根据websearch结果..."）
3. 最终 JSON 必须包含 tool_dependency 字段
4. 禁止 suggested_questions
5. 卡片字段严格使用以下格式：
   - expert_insight: message + list[{avatarType, title, description, comment}]
   - detail_card: message + list[{title, description}]  (5项: 公司/类型/联系人/地区/联系阶段 或 产品/采购量/交期/核心关注/匹配程度)
   - ai_judgement: title + description + list[{label, value, level}]

## tool_dependency 结构
{
  "required": true,
  "tools_used": ["websearch", "knowledge_base"],
  "claims": [
    {"id":"claim_001","claim":"从websearch得出的具体结论","source":"websearch","if_missing":"cannot_determine"},
    {"id":"claim_002","claim":"从knowledge_base得出的具体结论","source":"knowledge_base","if_missing":"miss_critical_info"}
  ],
  "reason": "为什么需要工具来完成此分析"
}

## 当前模式: {{ required_mode }}
如果 required_mode 是 tools_required：先调用 websearch + knowledge_base 获取外部信息，再基于工具返回数据输出 JSON。tool_dependency.required=true。
如果 required_mode 是 tools_optional：不要调用任何工具。直接输出 JSON。tool_dependency.required=false，tools_used=[]，claims=[]，reason 中说明询盘已包含完整信息。
```

> **注意**：`{{ messages }}` 会引用种子数据中的完整对话轨迹作为 few-shot 参考。如果 Studio 提示 `{{ messages }}` 不可用，可以用 `{{ meta }}` 代替。

### Step 4.5：添加 Expression block（类型归一化）⚠️ 必须

AI structured data 的输出类型可能不一致（有的 `{"messages":[...]}`，有的裸 `[...]`），导致 JSONL 写入报错 `cannot mix list and non-list`。加 Expression block 统一格式：

**Block 类型**: Expression (Jinja2)

```
Field name: final_output
Drop: false
Keep out of final dataset: false
```

**表达式**:
```jinja2
{% set obj = llm_structured_1 | default('{}') %}
{% if obj is mapping and obj.messages %}
{{ obj | tojson }}
{% elif obj is iterable and not obj is string %}
{"messages": {{ obj | tojson }}}
{% else %}
{{ obj | tojson }}
{% endif %}
```

> 这个 Expression 做了三件事：① 如果输出已经是 `{"messages":[...]}` 结构保持原样；② 如果输出是裸 `[...]` 数组，自动包装成 `{"messages":[...]}`；③ 兜底处理其他格式。输出统一为字符串类型，不会触发 "cannot mix list and non-list" 错误。

同时在 AI structured data block 上勾选 **Keep out of final dataset**——让 Expression block 的 `final_output` 作为最终导出列。

### Step 5：连接各 Block + 导出

```
Model preset(GPT-5.4-mini)
         ↓
Seed(37条) → Category×4 → AI structured data → Expression(类型归一化) → Export JSON
```

> 不需要 Validator（Python check 是检查 Python 代码的，不适配 JSON 校验）和 Expression（AI structured data 输出本身就是 `{"messages": [...]}` 格式）。质量过滤在 Studio 外完成——见下方"导出后校验"。

---

## 三、运行配置

| 参数 | 建议值 | 说明 |
|------|--------|------|
| Category 变体数 | 420 | 10行业×7地区×3买家×2模式 |
| AI structured data 目标数 | 500-1000 | 最终训练数据量 |
| Temperature | 0.7 | 多样性，首次生成 |
| Max Tokens | 8192 | 轨迹较长，需要足够空间 |
| Preview 数量 | 5 | 先验证格式正确 |
| Validator 通过率目标 | ≥ 70% | 不合格自动丢弃 |

### 运行步骤

1. **Preview** — 生成 5 条预览，人工检查质量
2. 检查预览样本：
   - tool_call 是否在 JSON 之前
   - tool 结果是否逼真（不是 "Simulated Results"）
   - 最终 JSON 是否有 tool_dependency
   - Response format 是否生效（输出是否严格符合 messages Schema）
3. **修正问题** — 根据预览结果调整 Prompt 或 Response format
4. **Run** — 全量生成 500-1000 条
5. **Export** — 导出为 JSON，传回本项目的 `data/generated_unsloth/`

---

## 四、预期产出

| 指标 | 目标 |
|------|------|
| 总样本数 | 500-1000 |
| required=true | ~75% (375-750) |
| required=false | ~25% (125-250) |
| Validator 通过率 | ≥ 70% |
| 每条含 tool_dependency | 100% |
| 每条含 evidence 引用 | ≥ 80% |

## 五、导出后质量校验

Studio 导出后，用本项目的 `quality_check.py` 做二次过滤：

```bash
cd /root/0420-fz/lora-qwen-0612

# 校验 Studio 导出的数据
python -c "
import json, sys; sys.path.insert(0, '.')
from scripts.lib.quality_check import validate_batch, generate_report

with open('data/generated_unsloth/studio_output.json') as f:
    data = json.load(f)

# 如果 Studio 导出的是数组 [{messages: [...]}, ...]
samples = []
for item in data:
    msgs = item.get('messages') or item.get('llm_structured_1', {}).get('messages')
    if msgs:
        samples.append({'messages': msgs})

result = validate_batch(samples)
print(generate_report(result))

# 保存合格的
passed = [s for s, r in zip(samples, result['results']) if r['passed']]
with open('data/processed/training_data_v4.json', 'w') as f:
    json.dump(passed, f, ensure_ascii=False, indent=2)
print(f'Saved {len(passed)}/{len(samples)} passed samples')
"
```

## 六、导出后合并到项目

```bash
# 将 Studio 导出的 JSON 放入项目
cp /path/to/exported/data.json data/generated_unsloth/studio_output.json

# 合并种子 + Studio 生成的数据
python -c "
import json
with open('data/processed/synthetic_seeds_for_studio.json') as f:
    seeds = json.load(f)
with open('data/generated_unsloth/studio_output.json') as f:
    generated = json.load(f)
all_data = seeds + generated
print(f'Total: {len(all_data)} samples')
with open('data/processed/training_data_v4.json', 'w') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
"
```
