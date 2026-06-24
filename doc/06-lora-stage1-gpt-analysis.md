这个结果其实非常有价值，因为它基本已经把你们的假设空间收敛到很小了。

从研究角度看，我认为你们现在已经可以排除一大批曾经怀疑的因素：

| 假设                          | 当前证据    |
| --------------------------- | ------- |
| SGLang Tool Parser Bug      | ❌ 基本排除  |
| Qwen 不会 Tool Calling        | ❌ 排除    |
| LoRA 没学到业务格式                | ❌ 排除    |
| 数据量太少                       | ❌ 大概率排除 |
| Constraint Tax 完全来自模型能力不足   | ❌ 不成立   |
| Training-Inference Mismatch | ✅ 高度可疑  |
| Tool Dependency 信号不够强       | ✅ 高度可疑  |

---

# 我对当前结果的解读

你的实验结果：

| 测试 | 微调前     | 微调后     |
| -- | ------- | ------- |
| T1 | 100%    | 100%    |
| T2 | 0%      | 0%      |
| T3 | 80~100% | 80~100% |

而：

```text
输出内容明显变长
输出内容明显更专业
开始虚构真实公司信息
```

这其实说明：

> LoRA 生效了。

否则输出风格不会发生这么大变化。

---

# 关键发现

微调后模型学到了：

```text
询盘
↓
输出完整分析报告
```

但没有学到：

```text
询盘
↓
必须Tool
↓
才能输出分析报告
```

换句话说：

LoRA优化了：

```text
P(Output | User)
```

没有优化：

```text
P(ToolCall | User)
```

---

# 为什么会这样

看你们的数据：

训练样本是：

```text
System:
必须输出JSON

User

Assistant:
Tool Call

Tool

Assistant:
JSON
```

---

但推理时：

OpenAI兼容格式：

```python
response_format = schema
tools = [...]
```

实际上进入模型上下文可能更像：

```text
<System hidden>

<tool schema hidden>

<response_format hidden>

User
```

---

于是训练看到的是：

```text
你必须输出JSON
```

推理看到的是：

```text
<|response_format|>
```

或者框架内部特殊token。

---

LoRA无法建立映射。

---

# 为什么会出现“高质量幻觉”

这是非常典型的SFT行为。

模型观察到：

训练集中的模式是：

```text
Tool Result
↓
Business Analysis
```

---

但训练loss只监督：

```text
最终输出
```

---

Tool Call部分token非常少：

例如：

```json
{
  "name": "websearch",
  "arguments": ...
}
```

几十token。

---

Final JSON：

```json
{
  "blocks": [...]
}
```

可能1500 token。

---

于是训练时：

```text
loss贡献：

Tool Call:
≈ 5%

Final Output:
≈ 95%
```

---

LoRA自然优先学习：

```text
如何生成分析
```

而不是：

```text
何时调用工具
```

---

# 我觉得当前最大的发现

其实不是 Tool Dependency 不够强。

而是：

> Tool Call Decision 根本没有进入主要训练目标。

你现在的训练实际上在做：

```text
Tool Usage SFT
```

而不是：

```text
Tool Selection SFT
```

---

# 这和你论文里的 Constraint Tax 有什么关系

实际上非常有意思。

你们最初的假设是：

```text
response_format
↓
压制tool calling
```

---

而现在更像：

```text
response_format
↓
改变解码策略

模型发现：

直接输出JSON
比

Tool → JSON

概率更高
```

所以：

```text
argmax
↓
直接JSON
```

---

# 我认为下一阶段不要继续加业务数据

837条已经足够验证趋势。

如果再扩到：

```text
3000条
5000条
```

大概率还是：

```text
T2 = 0%
```

只是幻觉更逼真。

---

# 下一步最值得做的实验

我会做一个极小规模数据集。

只训练：

## Tool Mandatory Dataset

例如：

```json
User

Assistant:
tool_call

Tool:
{"company":"ABC"}

Assistant:
{
  "company":"ABC"
}
```

---

然后负样本：

```json
Tool:
{}
```

Assistant:

```json
{
  "company":"unknown"
}
```

---

不要业务分析。

不要卡片。

不要外贸。

不要风险判断。

---

只保留：

```text
Tool
↓
Field
```

依赖关系。

---

100~200条即可。

---

# 我对当前阶段的判断

如果按研究路线来看：

### 第一阶段（已完成）

证明：

```text
Schema + Tool
↓
Qwen出现Tool Collapse
```

---

### 第二阶段（已完成）

证明：

```text
普通SFT
↓
无法恢复Tool Calling
```

这是你刚得到的结果。

---

### 第三阶段（下一步）

验证：

```text
Tool Dependency SFT
↓
能否恢复Tool Calling
```

这里的关键已经不再是增加业务样本数量，而是改变监督目标。

---

如果最终发现：

即使是极端 Tool Dependency Dataset，

```text
T2仍然=0%
```

那么你的论文结论会变得非常强：

> Constraint Tax 并非知识缺失问题，也并非工具能力问题，而是模型在 response_format 约束下形成的解码策略偏置（decoding policy bias）。普通 SFT 无法纠正该偏置，需要专门的 Tool Selection/Policy Learning 或 RL 方法介入。

这个结论的学术价值其实比“多做点 SFT 解决了问题”要高得多。
