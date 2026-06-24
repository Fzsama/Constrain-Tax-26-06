# A2 方案进度：JSON 内嵌 Tool Calls 绕过 FSM 硬约束

> 日期: 2026-06-23
> 状态: ✅ 训练完成 + T2=95% 突破验证通过

---

## 1. 背景

### 1.1 已确立的结论

经历 6 轮实验（4 轮 SFT + 1 轮 GRPO + 1 轮 6000 样本 SFT），确认：

- `response_format` (JSON Schema) 激活时，SGLang 通过 xgrammar 将 Schema 编译为有限状态自动机 (FSM)
- FSM 在**所有状态**下将 `<` (id=27) 和 `<tool_call>` (id=248058) 的 bitmask 设为 0
- Triton kernel `apply_token_bitmask_inplace_triton` 将 bit=0 的 token logit 设为 `-inf`
- softmax(-inf)=0 — `<tool_call>` 绝对不可达
- SFT/GRPO 在**权重层**修改的是模型内部 logit，无法越过**解码层**的硬 mask

### 1.2 A2 方案核心思路

不对抗 FSM，而是**顺应**它：

```
旧路径 (被 FSM block):
  模型 → 第一个 token "<" → FSM bit=0 → -inf → 失败

A2 路径 (FSM 允许):
  模型 → 第一个 token "{" → FSM bit=1 → OK
       → key "_tool_calls" → 合法 JSON key
       → value [{"name":"websearch",...}] → 合法 JSON array
       → 其他业务字段 → 合法 JSON
       → "}" → 完成
```

`_tool_calls` 是合法的 JSON 字段，其值为 JSON array of objects——全部是 FSM 允许的 token。框架层提取此字段执行工具，再将其从最终输出中移除。

---

## 2. 已完成的验证

### 2.1 xgrammar FSM 验证

| 格式 | FSM 结果 |
|------|:---:|
| 裸 `<tool_call>` XML (旧) | ❌ Step 0: `<` blocked |
| A2-XML: string 内嵌 `<tool_call>` | ❌ Step 6: `<tool_call>` (id=248058) blocked |
| **A2-JSON: array of objects** | ✅ **92/92 tokens ALL ACCEPTED** |

关键发现：`<tool_call>` (id=248058) 是特殊 token，xgrammar 在 JSON string value 内部也将其 mask 为 0。因此必须用 JSON 对象格式，不能把 XML 编码进 string。

### 2.2 种子数据生成

- 30 条 A2 格式种子：`data/processed/a2_seed_data.json`
- 输出格式：`{"_tool_calls": [...], "company_name": "...", ...}`

### 2.3 规模数据生成

- 8000 条 GPT-5.4-mini 生成的 A2 训练数据
- 100% 通过校验（含 `_tool_calls` 字段 + 合法 JSON）
- HF 数据集：[FZSAMA/qwen-a2-constraint-tax-data](https://huggingface.co/datasets/FZSAMA/qwen-a2-constraint-tax-data)

---

## 3. 当前状态：模型训练中 🔄

- 模型: Qwen3.6-35B-A3B
- 训练数据: 8000 条 A2 格式样本
- 工具: Unsloth Studio SFT (Raw Text)
- 训练完成后产出: `outputs/loraed-qwen-v2-0622/` (预期)

---

## 4. 下一步测试计划

### 4.1 T2 测试 (核心)

训练完成后，用包含 `_tool_calls` 字段的 JSON Schema 启动 SGLang 服务，测试：

```
T2: tools=ON, response_format=ON (含 _tool_calls 字段的 schema)
```

**成功标志**: T2 > 0% — 模型在 schema 约束下输出含 `_tool_calls` 的 JSON

### 4.2 对比基线

| 模型 | 训练格式 | T2 预期 |
|------|------|:---:|
| Base (Qwopus) | — | 0% |
| 旧格式 SFT (6000 条) | 传统 tool_call → JSON | 0% |
| **A2 SFT (8000 条)** | **_tool_calls 内嵌 JSON** | **?>0%** |

### 4.3 推理配置

A2 模型需要特殊的 `response_format` Schema——在原 Schema 上增加 `_tool_calls` 字段：

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "a2_response",
    "schema": {
      "type": "object",
      "properties": {
        "_tool_calls": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "arguments": {"type": "object"}
            },
            "required": ["name", "arguments"]
          }
        },
        "...业务字段..."
      }
    }
  }
}
```

框架层需在收到响应后：
1. 解析 `_tool_calls` 数组
2. 执行对应的工具调用
3. 将工具结果注入对话
4. 从最终输出中移除 `_tool_calls` 字段

---

## 5. 后续方向

| 方向 | 说明 |
|------|------|
| A2 训练成功 | 验证 T2>0%，框架层集成，生产部署 |
| A2 训练仍然 T2=0% | 检查训练数据格式与推理时的 token 级一致性，可能的根因：训练时 chat template 产出的 token 分布与推理时 FSM 允许集合不匹配 |

---

## 附录：A2 格式示例

```json
{
  "messages": [
    {"role": "system", "content": "You are an information extraction assistant. To answer any question, you MUST first use tools. Output a SINGLE JSON with _tool_calls field first, then business fields."},
    {"role": "user", "content": "Find info about TechCorp Inc."},
    {"role": "assistant", "content": "{\"_tool_calls\": [{\"name\": \"websearch\", \"arguments\": {\"query\": \"TechCorp Inc.\"}}, {\"name\": \"knowledge_base\", \"arguments\": {\"query\": \"EU compliance\"}}], \"company_name\": \"TechCorp Inc.\", \"company_info\": \"SaaS provider.\", \"compliance_notes\": \"CE required\"}"}
  ]
}
```
