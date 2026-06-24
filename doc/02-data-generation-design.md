# 训练数据生成方案 — 种子提取 + 规模生成

> 2026-06-12 | 数据获取流程详细设计

---

## 1. 整体流程

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: 种子数据获取 (20-30条)                          │
│                                                         │
│  询盘场景模板库 ──→ EvoAgent + GPT-5.4-mini ──→ 完整轨迹 │
│  (手写30个场景)      (调真实工具+输出JSON)     (tool_calls│
│                                                  + JSON) │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: 规模生成 (500-1000条)                           │
│                                                         │
│  种子轨迹(few-shot) + 场景变体 ──→ GPT-5.4-mini ──→ 更多轨迹│
│  (作为示例注入prompt)   (200+场景)    (批量生成)          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: 质量校验 + 格式转换                             │
│                                                         │
│  全部轨迹 ──→ 校验(工具率/JOSN合法性/参数合理性)          │
│           ──→ 格式转换(messages数组)                     │
│           ──→ train/eval split                          │
└─────────────────────────────────────────────────────────┘
```

---

## 1.5 正负样例策略

### 1.5.1 训练方式决定数据需求

| 训练方式 | 需要正样例 | 需要负样例 | 说明 |
|---------|:---:|:---:|------|
| **SFT**（当前阶段） | ✅ | ❌ | 只教正确行为序列，不需要反面教材 |
| **DPO**（可选第二阶段） | ✅ | ✅ | 需要 chosen/rejected 对比对 |

### 1.5.2 SFT 阶段：只要正样例

SFT 通过 teacher forcing 训练——模型只在 assistant 消息上计算 loss，学习"在这个上下文中，assistant 应该输出什么"。只要训练数据中的 assistant 行为是正确的（先调工具 → 后输出 JSON），模型就会学习这个行为模式。

**当前设计已覆盖**：GPT-5.4-mini 生成的轨迹天然是正样例。

### 1.5.3 DPO 阶段：需要负样例（对比对）

DPO 需要成对数据：**chosen（正确的）vs rejected（错误的）**，让模型学习偏好排序。

```
chosen:   用户询盘 → assistant 调 websearch → 获取真实数据 → 输出 JSON ✅
rejected: 用户询盘 → assistant 直接输出 JSON（编造/空值填充）❌
```

**负样例来源**：同一询盘场景，用 **Qwen3.6-35B 不加 Plan B** 运行，捕获 Constraint Tax 行为。

### 1.5.4 负样例采集策略

```
Phase 1.5: 负样例采集（为 DPO 准备，SFT 阶段可先跳过）

对每个种子场景，同时采集两条轨迹:

  GPT-5.4-mini (positive)              Qwen35B 不加 Plan B (negative)
  ┌─────────────────────┐              ┌──────────────────────┐
  │ System: tools+schema │              │ System: tools+schema  │
  │ User: 询盘           │              │ User: 询盘            │
  │ Assistant:           │              │ Assistant:            │
  │   tool_call: websearch │            │   content: {"blocks": │
  │   tool_call: kb       │             │     [空值/伪造数据]} │
  │   ...                 │              │   (跳过工具调用)    │
  │   content: {合法JSON} │              └──────────────────────┘
  └─────────────────────┘
```

**关键**：两条轨迹使用**完全相同的 system prompt 和 user message**，唯一的变量是模型。

**采集条件**：
- Qwen35B 使用 `response_format` API 参数（触发 Constraint Tax）
- **不启用 Plan B**（不用 `_deferred_response_format`）
- `tools=ON`（工具可用但被压制）
- 如果 Qwen 偶尔调了工具（罕见但可能），该样本不适用 DPO（没有形成对比）

### 1.5.5 DPO 数据格式

```json
{
  "prompt": [
    {"role": "system", "content": "你是贸探·跟单专家...\n## 可用工具\n- websearch..."},
    {"role": "user", "content": "请分析这封询盘: ..."}
  ],
  "chosen": [
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", ...}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "搜索结果: ..."},
    {"role": "assistant", "content": "{\"blocks\":[...]}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "{\"blocks\":[{\"type\":\"card\",\"card\":{\"type\":\"expert_insight\",\"message\":\"AI 专家指点\",\"list\":[{\"comment\":\"该买家...\"}]}}]}"}
  ]
}
```

**DPO 数据构造规则**：
- `prompt` = system prompt + user message（两个模型共享相同前缀）
- `chosen` = GPT-5.4-mini 的完整回复（含 tool_calls + tool_results + final JSON）
- `rejected` = Qwen35B 的完整回复（直接输出 JSON，无工具调用）
- 只有 Qwen 确实没调工具时才纳入 DPO 训练集

---

## 2. Phase 1: 种子数据获取

### 2.1 为什么用 EvoAgent 而非直接调 API？

| 方式 | 优点 | 缺点 |
|------|------|------|
| 直接调 OpenAI API | 简单 | 工具调用是"模拟"的（模型输出 tool_call 但不会真的执行），tool_result 需要手动编造，不真实 |
| **通过 EvoAgent 调用** | 工具真实执行（websearch 真实搜索、knowledge_base 真实查询），tool_result 是真实数据 | 需要 EvoAgent 运行环境 |

**结论**：使用 EvoAgent 的 inquiry-reply-agent + GPT-5.4-mini 跑真实询盘场景，捕获完整对话轨迹。GPT-5.4-mini 是唯一 Constraint Tax 豁免模型，在 `tools + response_format` 双重约束下仍能正确"先调工具、后输出 JSON"。

### 2.2 种子场景设计

手写 30 个多样询盘场景，覆盖以下维度：

```
行业 (10类, 每类3个):
├── LED 照明:      灯泡、灯带、工矿灯
├── 消费电子:      USB Hub、蓝牙耳机、充电器
├── 机械设备:      注塑机、CNC、包装机
├── 纺织服装:      T恤、工装、面料
├── 化工材料:      ABS粒子、涂料、胶粘剂
├── 家居用品:      厨具、收纳、浴室五金
├── 汽车配件:      刹车片、车灯、滤清器
├── 医疗器械:      口罩、手套、血压计
├── 太阳能:        光伏板、逆变器、储能电池
└── 五金工具:      钻头、扳手、紧固件

地区 (7类, 均匀分布):
├── 北美:     美国、加拿大
├── 西欧:     德国、法国、英国、意大利
├── 中东:     UAE、沙特
├── 东南亚:   印度、越南、泰国
├── 南美:     巴西、墨西哥
├── 非洲:     尼日利亚、南非
└── 大洋洲:   澳大利亚

买家类型 (3类):
├── 品牌商:   自有品牌、要求ODM
├── 分销商:   多品类采购、重价格
└── 零售商:   小批量、快交期

询盘复杂度 (3级):
├── 简单 RFQ:        只问价格和MOQ
├── 标准询盘:        产品规格+数量+认证要求
└── 复杂询盘:        多产品线+技术参数+附件+特殊条款
```

### 2.3 种子场景示例

```python
SEED_INQUIRIES = [
    {
        "id": "seed_001",
        "industry": "LED 照明",
        "region": "德国",
        "buyer_type": "品牌商",
        "complexity": "标准询盘",
        "user_message": """请分析这封询盘:

公司: LichtDesign GmbH, 德国高端照明品牌
产品: LED Strip Lights, 24V, CRI>90, 2700K-6500K tunable white, 10mm PCB
数量: 5000 meters (1000 rolls × 5m)
要求: DDP 汉堡港, CE/ROHS/REACH, 希望OEM包装

客户签名里有官网: www.lichtdesign.de""",
    },
    {
        "id": "seed_002",
        "industry": "消费电子",
        "region": "美国",
        "buyer_type": "分销商",
        "complexity": "复杂询盘",
        "user_message": """分析这个美国客户的询盘:

Hi, I'm John from TechDist Inc. (San Jose, CA). We're interested in your USB-C hubs.
We need:
- 12-in-1 model: 2×HDMI, 1×DP, 3×USB-A, 2×USB-C, SD/TF, RJ45, 3.5mm, PD 100W pass-through
- 8-in-1 model: 1×HDMI, 2×USB-A, 1×USB-C, SD/TF, RJ45, PD 100W
Qty: 3000 units each model
Target price: under $25 for 8-in-1, under $35 for 12-in-1
Need UL/FCC certification. Trial order first, then monthly container.
Payment: Net 30 after delivery.

公司网站: www.techdist.com""",
    },
    # ... 共 30 个
]
```

### 2.4 种子数据采集脚本设计

```python
# scripts/01_collect_seed_traces.py

"""
用 EvoAgent inquiry-reply-agent + GPT-5.4-mini 采集种子训练轨迹。

工作流程:
1. 加载 EvoAgent 配置（inquiry-reply-agent, tools, response_format=blocks）
2. 对每个种子场景:
   a. 创建新 session
   b. 发送询盘消息
   c. 收集完整 conversation history（含 tool_calls + tool_results）
   d. 校验轨迹质量（有工具调用 + JSON 合法）
3. 保存轨迹到 data/raw/seed_traces/

依赖:
- EvoAgent 项目（core/agent.py, core/sub_agent.py）
- .env 配置（FFS, API keys）
- GPT-5.4-mini API（AEP 网关）

输出格式（每条轨迹一个 JSON 文件）:
{
  "meta": {
    "seed_id": "seed_001",
    "industry": "LED 照明",
    "region": "德国",
    "buyer_type": "品牌商",
    "model": "gpt-5.4-mini",
    "timestamp": "2026-06-12T..."
  },
  "messages": [
    {"role": "system", "content": "你是贸探·跟单专家..."},
    {"role": "user", "content": "请分析这封询盘: ..."},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "websearch", "content": "..."},
    {"role": "tool", "tool_call_id": "...", "name": "knowledge_base", "content": "..."},
    {"role": "assistant", "content": "{\"blocks\":[...]}"}
  ]
}
"""
```

**关键实现细节**：
- 使用 `EvoAgent.run_stream()` 收集 SSE 事件，重组为 conversation history
- 不启动 Web 服务，直接以脚本方式调用 `EvoAgent`
- 每次种子采集创建新 session，避免上下文污染
- 从 `_post_session` 保存前的 `self.history` 提取 messages

### 2.5 种子轨迹质量校验

每个种子轨迹必须满足：

| 检查项 | 标准 | 不通过处理 |
|--------|------|-----------|
| 工具调用数 ≥ 1 | assistant 消息中有 tool_calls | 丢弃，检查 GPT-5.4-mini 配置 |
| 工具调用在 JSON 之前 | tool_calls 的 index < 最终 assistant JSON 的 index | 丢弃（理论上 GPT-5.4-mini 不会出现） |
| 最终输出是合法 JSON | `json.loads(final_content)` 通过 | 尝试修复（trim 前后空白），仍失败则丢弃 |
| JSON 符合 AgentResponse schema | `blocks` 字段存在且为 list | 记录警告但仍保留 |
| 不是空值填充 | blocks 不为空，text 有实质内容 | 记录警告但仍保留 |

---

## 3. Phase 2: 规模生成

### 3.1 方法：Few-shot Prompting

将 Phase 1 采集的种子轨迹作为 few-shot 示例注入 prompt，引导 GPT-5.4-mini 生成更多样的训练轨迹。

```
┌──────────────────────────────────────────┐
│ GPT-5.4-mini 规模生成 Prompt 结构          │
│                                          │
│ System:                                  │
│   你是训练数据生成器。基于示例轨迹格式，    │
│   为新的询盘场景生成完整对话轨迹。          │
│   规则: 必须先调工具，再输出 JSON。        │
│                                          │
│ Few-shot 示例 (2-3条种子轨迹):             │
│   示例1: [完整的 messages 数组]            │
│   示例2: [完整的 messages 数组]            │
│                                          │
│ 新场景:                                   │
│   {新的询盘描述}                          │
│                                          │
│ 输出: 完整的 messages 数组                │
└──────────────────────────────────────────┘
```

### 3.2 场景变体生成

基于 30 个种子场景，为每个生成 5-10 个变体：

```python
# 变体策略
VARIATION_STRATEGIES = [
    # 同行业不同地区
    ("same_industry_diff_region", "LED照明 × 美国 → LED照明 × 沙特"),
    # 同地区不同行业
    ("same_region_diff_industry", "德国LED → 德国机械设备"),
    # 同产品不同买家类型
    ("same_product_diff_buyer", "USB Hub 分销商 → USB Hub 品牌商"),
    # 同买家不同复杂度
    ("same_buyer_diff_complexity", "标准询盘 → 简单RFQ"),
    # 换公司名/人名/产品型号
    ("surface_variation", "LichtDesign → LuminaTech, LED Strip → COB Strip"),
]
```

### 3.3 规模生成脚本设计

```python
# scripts/02_generate_training_data.py

"""
基于种子轨迹，使用 GPT-5.4-mini 规模生成训练数据。

工作流程:
1. 加载种子轨迹（Phase 1 产出）
2. 生成场景变体列表（200+ 个新场景）
3. 对每个场景变体:
   a. 选 2-3 条最相似的种子轨迹作为 few-shot
   b. 调 GPT-5.4-mini 生成完整轨迹
   c. 校验质量（同 Phase 1 校验标准）
   d. 合格 → 保存；不合格 → 重试一次或丢弃
4. 去重 + train/eval split
5. 输出 train.json 和 eval.json

关键参数:
- 每次生成 1 条轨迹（保证质量）
- temperature=0.7（增加多样性，但不要太高导致格式错误）
- max_tokens=8192（轨迹较长）
"""

# few-shot 选择策略：按行业+地区相似度选最接近的种子
def select_few_shots(seed_traces, scenario, n=3):
    """选与目标场景最相似的 n 条种子轨迹。"""
    scored = []
    for t in seed_traces:
        score = 0
        if t["meta"]["industry"] == scenario["industry"]:
            score += 3
        if t["meta"]["region"] == scenario["region"]:
            score += 2
        if t["meta"]["buyer_type"] == scenario["buyer_type"]:
            score += 1
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:n]]
```

### 3.4 工具调用结果的模拟

**关键问题**：GPT-5.4-mini 在批量生成时，`tool_result` 怎么填？

GPT-5.4-mini 不知道真实的 websearch 结果。但这不是问题——训练目标不是教模型"搜索结果是什么"，而是教模型"先搜再填 JSON"的执行顺序。

处理方式：
1. **tool_call 参数必须合理**（非空 query、格式正确的 JSON）
2. **tool_result 由 GPT-5.4-mini 基于询盘上下文编造**，但必须看起来像真实搜索结果
3. **最终 JSON 必须基于 tool_result 中的信息**，不能凭空编造

```python
# 规模生成 prompt 中的关键约束
"""
## tool_result 生成规则

1. websearch 结果：必须包含公司名称、所在地、主营业务、规模等关键信息
2. knowledge_base 结果：必须包含目标国家的认证要求、关税政策等行业信息
3. 最终 JSON 中的信息必须能从 tool_result 中找到依据
4. 禁止出现 "Simulated Websearch Results" 等占位符——必须生成实质内容
"""
```

---

## 4. Phase 3: 数据格式转换与校验

### 4.1 原始轨迹 → 训练数据

EvoAgent 的 history 格式与训练数据格式的差异：

```python
def convert_trace_to_training_data(trace_messages):
    """
    将 EvoAgent conversation history 转为 SFT 训练格式。

    EvoAgent history:
    [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": null, "tool_calls": [...]},
      {"role": "tool", "tool_call_id": "...", "name": "websearch", "content": "..."},
      {"role": "assistant", "content": "{\"blocks\":[...]}"},
      ...可能有更多轮次...
    ]

    训练数据格式（同上，但需要截断和清理）:
    - 去掉中间的系统消息（history 中可能有多次 system prompt 注入）
    - 保留第一条 system 消息（含工具定义和 schema 约束）
    - 保留 user → assistant(tool_calls) → tool → assistant(JSON) 的完整序列
    - 去掉 skill_manager 的内部调用（get_ref 等非外部工具调用）
    - 只保留外部工具：websearch, knowledge_base, fetchurl
    """
    # 1. 过滤内部工具
    EXTERNAL_TOOLS = {"websearch", "knowledge_base", "fetchurl"}
    filtered = []
    for msg in trace_messages:
        if msg.get("role") == "tool" and msg.get("name") not in EXTERNAL_TOOLS:
            # 跳过 skill_manager, update_user_profile 等内部工具
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # 过滤掉对内部工具的调用
            external_calls = [
                tc for tc in msg["tool_calls"]
                if tc.get("function", {}).get("name") in EXTERNAL_TOOLS
            ]
            if not external_calls:
                continue
            msg = {**msg, "tool_calls": external_calls}
        filtered.append(msg)

    # 2. 只保留第一条 system 消息
    system_idx = None
    for i, msg in enumerate(filtered):
        if msg.get("role") == "system":
            if system_idx is None:
                system_idx = i
            else:
                filtered[i] = None  # 标记删除
    filtered = [m for m in filtered if m is not None]

    # 3. 确保以 system 开头
    if filtered and filtered[0]["role"] != "system":
        filtered.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    return {"messages": filtered}
```

### 4.2 自动化质量校验

```python
def validate_training_sample(sample):
    """
    校验单条训练数据。返回 (passed, issues)。
    """
    msgs = sample["messages"]
    issues = []

    # 1. 结构校验
    roles = [m["role"] for m in msgs]
    if "system" not in roles:
        issues.append("缺少 system 消息")
    if roles.count("user") < 1:
        issues.append("缺少 user 消息")

    # 2. 工具调用校验（核心）
    tool_call_count = 0
    final_json_idx = -1
    for i, msg in enumerate(msgs):
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                tool_call_count += len(msg["tool_calls"])
            elif msg.get("content") and msg["content"].strip().startswith("{"):
                final_json_idx = i

    if tool_call_count == 0:
        issues.append("缺少工具调用")
    if final_json_idx == -1:
        issues.append("缺少最终 JSON 输出")

    # 3. 工具调用在 JSON 之前
    last_tool_call_idx = -1
    for i, msg in enumerate(msgs):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            last_tool_call_idx = i
    if last_tool_call_idx > final_json_idx > 0:
        issues.append("工具调用在 JSON 输出之后（执行顺序错误）")

    # 4. JSON 合法性
    if final_json_idx > 0:
        try:
            obj = json.loads(msgs[final_json_idx]["content"])
            if "blocks" not in obj:
                issues.append("JSON 缺少 blocks 字段")
        except json.JSONDecodeError:
            issues.append("最终输出不是合法 JSON")

    # 5. 工具调用参数非空
    for i, msg in enumerate(msgs):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                    if not any(v for v in args.values() if v):
                        issues.append(f"消息{i} 工具调用参数为空")
                except json.JSONDecodeError:
                    issues.append(f"消息{i} 工具调用参数格式错误")

    return len(issues) == 0, issues
```

### 4.3 数据统计报告

```
训练数据生成报告
================
种子轨迹:     28/30 条通过校验
规模生成:     523/600 条通过校验
总合格样本:   551 条

train split:  496 条 (90%)
eval split:    55 条 (10%)

行业分布:
  LED照明: 15.2%  消费电子: 14.7%  机械设备: 10.3%
  纺织服装: 9.8%   化工材料: 5.4%   家居用品: 10.1%
  汽车配件: 5.2%   医疗器械: 4.9%   太阳能:   9.6%
  五金工具: 5.1%   其他:     9.7%

地区分布:
  北美: 20.1%  西欧: 24.8%  中东: 14.9%
  东南亚: 15.2%  南美: 10.1%  非洲: 9.8%
  大洋洲: 5.1%

每条平均:
  工具调用次数: 2.1
  对话轮次:     5.8
  token 数:     1850
```

---

## 5. 实现计划

### 5.1 文件结构

```
data/
├── seeds/
│   ├── inquiry_templates.py       # 30个种子场景定义
│   └── seed_traces/               # Phase 1 产出
│       ├── seed_001.json
│       ├── seed_002.json
│       └── ...
├── generated/                      # Phase 2 产出
│   ├── gen_001.json
│   └── ...
├── processed/                      # Phase 3 产出
│   ├── train.json                 # 最终训练集
│   └── eval.json                  # 最终评估集
└── reports/
    └── generation_report.json     # 统计报告

scripts/
├── 01_collect_seed_traces.py      # Phase 1: EvoAgent 种子采集
├── 02_generate_training_data.py   # Phase 2: 规模生成
├── 03_validate_and_convert.py     # Phase 3: 校验+格式转换
└── lib/
    ├── seed_scenarios.py          # 种子场景模板库
    ├── trace_utils.py             # 轨迹提取/转换工具
    └── quality_check.py           # 质量校验函数
```

### 5.2 时间估算

| 步骤 | 内容 | 预估时间 |
|------|------|---------|
| 1 | 编写种子场景 (30个) | 30 min |
| 2 | 实现种子采集脚本 + 运行 | 1-2 h (含 debug) |
| 3 | 检查种子轨迹质量 | 30 min |
| 4 | 生成 200+ 场景变体 | 30 min |
| 5 | 实现规模生成脚本 | 1-2 h |
| 6 | 运行规模生成 (500-1000条) | 2-4 h (API 调用) |
| 7 | 质量校验 + 格式转换 | 30 min |
| **总计** | | **6-10 h** |

---

## 6. 环境准备命令

```bash
# 1. 创建 conda 环境
conda create -n lora_qwen_0612 python=3.11 -y

# 2. 激活环境
conda activate lora_qwen_0612

# 3. 安装 PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 Unsloth（会自动安装 trl, peft, transformers 等）
pip install unsloth

# 5. 安装其余依赖
pip install datasets accelerate bitsandbytes xformers --index-url https://download.pytorch.org/whl/cu124
pip install pydantic loguru httpx python-dotenv

# 6. 验证安装
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
from unsloth import FastLanguageModel
print('Unsloth OK')
"
```
