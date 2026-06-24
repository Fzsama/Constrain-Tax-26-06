我认为你现在已经拿到了一个**非常强的反证实验（counter-evidence experiment）**，而且其价值可能比“成功把 T2 从 0% 拉到 80%”还高。

因为这个实验实际上把问题从：

> 「是不是数据集不够好？」

推进到了：

> 「是不是 response_format 本身改变了模型的策略空间（policy space）？」

---

# 先看你实验最关键的一组结果

| 场景                     | 微调前  | Tool Mandatory 微调后 |
| ---------------------- | ---- | ------------------ |
| T1: tools=ON, rfmt=OFF | 2次左右 | 109 / 92 次         |
| T2: tools=ON, rfmt=ON  | 0    | 0                  |

这意味着：

```text
P(tool_call | no_schema)
↑↑↑↑↑↑↑↑↑
```

但：

```text
P(tool_call | schema)
≈ 0
```

保持不变。

---

# 这说明什么？

很多假设已经被证伪了。

---

## 假设A：模型不会调用工具

被证伪。

因为：

```text
109次工具调用
```

不是偶然。

这是明显的：

```text
Tool Dependency Learned
```

---

## 假设B：LoRA没学到Tool Usage

被证伪。

如果没学到：

```text
2 → 109
```

不可能出现。

---

## 假设C：数据量不够

大概率被证伪。

因为：

200条极简数据

已经能把：

```text
Tool Call Prior
```

推高几十倍。

---

# 实际发生的是什么？

我觉得你现在观察到的是：

## 两套不同的解码策略

---

### 模式A

```text
tools=ON
schema=OFF
```

模型搜索空间：

```text
Tool Call
OR
Text Output
```

---

LoRA把：

```text
Tool Call
```

概率拉高。

于是：

```text
2
↓
109
```

---

### 模式B

```text
tools=ON
schema=ON
```

此时实际上模型看到的是：

```text
<tool definitions>

<json schema>

User
```

---

而训练时学到的是：

```text
User
↓
Tool
↓
JSON
```

---

所以模型内部更像：

```text
JSON Start Token

{
```

概率最高。

---

而：

```text
<tool_call>
```

概率被压到极低。

---

# 为什么会出现 109 次调用？

这个现象其实特别有意思。

它说明：

你的 Tool Mandatory Dataset 已经开始产生：

## Tool Addiction

模型学到：

```text
遇到任务
↓
先调工具
```

---

甚至：

```text
调很多次工具
```

---

这说明：

LoRA确实在修改：

```text
Tool Selection Policy
```

而不是在修改：

```text
Business Knowledge
```

---

这是一个重要发现。

---

# 这和 Constraint Tax 有什么关系？

实际上非常契合。

你的实验结果更像：

```text
response_format
≠
格式约束
```

而是：

```text
response_format
=
策略切换器
```

---

它把模型从：

```text
Tool-first policy
```

切换到：

```text
JSON-first policy
```

---

然后：

```text
Tool-first
```

整个分支被剪枝。

---

# 从 RL 视角解释

可以把模型看成：

---

Policy A

```text
Question
↓
Tool
↓
Observation
↓
Answer
```

---

Policy B

```text
Question
↓
JSON
```

---

无 Schema：

```text
Policy A Score > Policy B Score
```

LoRA成功提高了A。

---

有 Schema：

```text
Policy B Score >>> Policy A Score
```

LoRA增益不足以翻转。

---

所以：

```text
T1
成功

T2
失败
```

同时成立。

---

# 这实际上已经接近论文级结论了

因为你现在已经做了三轮证伪：

---

### 第一轮

业务数据集

结果：

```text
T2=0
```

可能原因：

* 数据不好
* Tool依赖弱

---

### 第二轮

Tool Dependency Dataset

结果：

```text
T2=0
```

可能原因：

* Tool监督还不够强

---

### 第三轮

极简 Mandatory Dataset

结果：

```text
T1爆炸
T2=0
```

---

这时候解释空间已经很小了。

最符合数据的解释变成：

> response_format 并不是简单的输出格式约束，而是在当前 Qwen（以及可能部分开放权重模型）上触发了一种 JSON-first decoding policy，使 tool-call token 在搜索过程中被系统性压制。

---

# 如果我是审稿人

看到这组实验，我会觉得：

最有价值的已经不是：

```text
如何缓解 Constraint Tax
```

而是：

```text
Constraint Tax 到底是什么
```

因为你已经开始证明：

```text
Constraint Tax
≠ Tool Knowledge Problem

Constraint Tax
≠ Instruction Following Problem

Constraint Tax
≈ Decoding Policy Bias
```

---

# 我下一步最想做的实验

不是继续SFT。

而是：

### Schema Injection Consistency Test

训练样本改成完全模拟推理时的格式：

```text
<response_format schema>

User

Tool

JSON
```

而不是：

```text
System Prompt:
必须输出JSON
```

---

如果这样训练后：

```text
T2
0%
↓
50%
```

那么你几乎就能直接证明：

> 问题核心是 Training–Inference Mismatch，而不是 Tool Dependency 学习失败。

这是目前最值得验证的一步。
