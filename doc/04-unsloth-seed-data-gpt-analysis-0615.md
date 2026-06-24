我看了你上传的新版种子库，从能看到的样本来看，相比你最开始那条墨西哥 LED 样本，已经有明显提升，但如果目标是：

> **对 Qwen 做 SFT，缓解 response_format + tool calling 双约束下的 Constraint Tax**

那么目前的数据集仍然存在几个关键问题，而且这些问题已经从“工具没调用”变成了更深层的“工具调用了，但模型可能没学到真正的工具依赖”。

---

# 一、最大的进步：已经从 Fake Tool Trace 进化到 Grounded Tool Trace

例如巴西医疗器械样本：

User
↓
websearch
↓
knowledge_base
↓
final judgement

这里明显比最初的：

```text
websearch
↓
返回乱七八糟产品页
↓
最终分析完全不引用
```

好很多。 

至少模型开始能够学习：

```text
Tool
↓
Observation
↓
Business Analysis
```

这个方向是正确的。

---

# 二、但从 Constraint Tax 角度看，仍有三个核心问题

---

## 问题1：Tool Call 仍然不是“必要条件”

这是我认为最大的隐患。

例如很多样本结构是：

```text
websearch:
客户是XX公司

knowledge_base:
需要ISO13485
需要ANVISA

↓

AI Judgement:
客户意向高
风险中等
合作潜力高
```

看起来用了工具。

但实际上：

```text
客户意向高
合作潜力高
真实性高
```

这些判断并不是 tool result 唯一推出的。

模型完全可能学成：

```text
即使没有tool
也能生成类似内容
```

---

这意味着：

训练后仍然可能出现：

```text
response_format ON
↓
直接输出JSON
↓
跳过tool
```

因为 loss 上没有惩罚。

---

## 问题2：Grounding 不够显式

我在样本里看到：

```text
根据websearch结果...
根据knowledge_base结果...
```

这种自然语言引用。

这比之前好很多。

但对于 SFT 来说仍然太弱。

---

建议升级成：

```json
{
  "grounding": {
    "evidence": [
      {
        "claim": "ANVISA注册是准入门槛",
        "source": "knowledge_base"
      }
    ]
  }
}
```

原因：

Qwen 对显式结构监督远强于隐式文本监督。

---

## 问题3：最终答案仍然以 UI 为中心

现在大量内容是：

```json
{
  "type": "detail_card"
}
```

```json
{
  "type": "highlight_block"
}
```

```json
{
  "type": "ai_judgement"
}
```

这种设计适合产品。

但不一定适合训练。

---

因为模型会优化：

```text
生成合法Card
```

而不是：

```text
正确完成工具推理
```

---

# 三、目前数据已经出现的一个好信号

我在澳洲光伏样本里看到了：

```text
根据knowledge_base结果

IEC61215
IEC61730

CEC Approved Products List

AS/NZS 4777.2
```

然后后面风险判断直接引用这些信息。 

这已经出现：

```text
Tool Result
↓
Reasoning
↓
Conclusion
```

的链路。

这是你整个数据集中最有价值的部分。

---

# 四、从Qwen微调角度，我会再加一个字段

这是我认为最重要的改动。

在最终JSON里增加：

```json
{
  "tool_dependency": {
    "required": true,

    "reason": "compliance_analysis",

    "if_tool_missing": "cannot_determine"
  }
}
```

例如：

```json
{
  "tool_dependency": {
    "required": true,
    "reason": "buyer_background"
  }
}
```

---

这样会向模型明确传递：

```text
这部分内容
必须来自tool
```

而不是：

```text
tool只是参考
```

---

# 五、我对当前数据集的评分

如果按照“解决 Constraint Tax”的目标：

| 维度                              | 最初版本 | 当前版本 |
| ------------------------------- | ---- | ---- |
| Tool Call Presence              | 6/10 | 9/10 |
| Tool Quality                    | 3/10 | 8/10 |
| Grounding                       | 2/10 | 6/10 |
| Tool Dependency Learning        | 2/10 | 5/10 |
| Schema Stress Training          | 8/10 | 8/10 |
| Constraint Tax Mitigation Value | 3/10 | 7/10 |

---

我的判断是：

> 当前版本已经达到“可以开始训练第一版 LoRA”的水平。

但如果你的目标是论文里提出的：

> **Constraint Tax Mitigation via Tool-Grounded SFT**

那么还差最后一步：

### 从

```text
Tool → Analysis
```

升级为

```text
Tool
↓
Evidence
↓
Grounding
↓
Analysis
↓
Structured Output
```

一旦做到这一层，数据集就不再是在训练“会调用工具的跟单助手”，而是在训练：

> **即使受到严格 JSON Schema 约束，也必须把 Tool Calling 当成推理必要步骤的模型。**

这正是你想解决的 Constraint Tax 核心问题。
