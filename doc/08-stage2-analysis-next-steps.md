# Stage 2 分析：三轮证伪后的下一步

> 结合 GPT 两篇分析 + 两轮实测结果

---

## 三轮实验总结

| 轮次 | 数据集 | T1 (无Schema) | T2 (有Schema) | 结论 |
|:--:|------|:--:|:--:|------|
| 1 | 业务数据 837条 | 2次 | **0%** | LoRA生效但学到错误目标（更好的幻觉） |
| 2 | Tool Mandatory 200条 | **109次** | **0%** | Tool Dependency 已学会，但 Schema 仍压制 |

## 已证伪的假设

| 假设 | 证据 |
|------|------|
| 模型不会调工具 | ❌ T1=109次，不是偶然 |
| LoRA 没学到 Tool Usage | ❌ 200条极简数据把工具调用推高了 50 倍 |
| 数据量不够 | ❌ 200条已有显著效果 |
| Tool Dependency 信号不够强 | ❌ 极简数据只保留了 Tool→Field 因果链 |

## 核心发现

**`response_format` 不是格式约束，而是策略切换器：**

```
无 Schema:  Tool-first policy (LoRA 成功把 T1 拉到 109)
有 Schema:  JSON-first policy (LoRA 增益不足以翻转，T2 仍为 0)
```

同一个模型，同一个权重，在两种模式下走到了完全相反的行为极端——说明问题不在模型能力，在**解码层**：`response_format` 将 Tool Call token 的概率系统性压制了。

## 最可能的根因：Training-Inference Mismatch

训练数据中的 schema 约束以**自然语言**形式出现在 system prompt 中：
```
你必须输出 JSON 格式...
```

推理时 schema 通过 API 参数注入，在模型内部可能是完全不同的 token 表示：
```python
response_format = {"type": "json_schema", ...}
→ 框架内部可能渲染为 <|response_format|> 等特殊标记
```

SFT 从未见过这种 token 形式，因此无法迁移。

---

## 下一步：Schema Injection Consistency Test

**最值得做的实验**——验证 Training-Inference Mismatch 是根因。

### 实验设计

构造一个新的 200 条 Tool Mandatory 数据集，但**schema 约束的注入方式**改成模拟推理时的格式：

```
旧数据格式:
  System: "你必须输出 JSON: {company_name, company_info, compliance_notes}"
  User: "..."
  → Tool → JSON

新数据格式:
  System: "你是一个信息提取助手"  (不含 schema 描述)
  User: "根据以下 JSON Schema 输出: {company_name, company_info, compliance_notes}\n查询: ..."
  → Tool → JSON
```

或者更进一步——把 schema 以类似框架内部格式写入：
```
  <|response_format|>{"type":"json_schema","json_schema":{...}}<|/response_format|>
  User: "..."
  → Tool → JSON
```

### 预期

| 场景 | 如果 Mismatch 是根因 | 如果不是根因 |
|------|:--:|:--:|
| T2 | 从 0% → >0%，理想 ≥ 30% | 仍为 0% |

如果 T2 有任何回升，就能直接证明：**问题核心是 Training-Inference Mismatch**。
如果仍为 0%，则证明 Constraint Tax 是 `response_format` 触发的**不可逆解码策略偏置**，需要 RL/DPO/decoding-level 方法介入。

两种结果论文结论都很强。

### 操作

仅需修改 `scripts/04_generate_tool_mandatory_data.py` 中的 system prompt 构建方式，把自然语言的 "必须输出 JSON" 替换为框架风格的 schema 注入。200 条数据，生成后训练 + 测试，1-2 小时可出结果。
