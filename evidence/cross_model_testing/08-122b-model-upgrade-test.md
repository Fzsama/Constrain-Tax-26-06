# 方向 C 验证：Qwen3.5-122B-A10B-GPTQ-Int4 模型升级测试

**日期**: 2026-06-04
**前提**: Qwen35B 上 Tool Calling 与 Structured Output 已被实证互斥（见 `06-qwen-toolcalling-formatoutput.md`）

---

## 1. 测试环境

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3.5-122B-A10B-GPTQ-Int4（官方，ModelScope） |
| 量化 | GPTQ Int4（`moe_wna16`） |
| 服务 | SGLang 0.5.9 |
| GPU | 2× A800 80GB |
| TP | 2 |
| 显存 | 每卡剩余 ~14GB（KV cache 用） |
| 启动参数 | `--reasoning-parser qwen3 --tool-call-parser qwen3_coder --quantization moe_wna16` |

---

## 2. 测试方案

同 35B 测试（`06-qwen-toolcalling-formatoutput.md` §2），3 组对照：

| 测试 | tools | response_format | 验证目标 |
|------|-------|-----------------|---------|
| 1 | ✅ ON | ❌ OFF | 能否调用工具 |
| 2 | ✅ ON | ✅ ON | 能否同时调用工具 + 输出 JSON |
| 3 | ❌ OFF | ✅ ON | 能否输出合法 JSON |

测试脚本：`tests/fz-qwen-test/test_122b_tool_response_format.py`

---

## 3. 测试结果

### 测试 1: tools=ON, response_format=OFF

```
耗时: 2.4s
工具调用: 2 次
  [0] websearch: query="BrightLight Inc. USA lighting products importer company background"
  [1] knowledge_base: query="LED strip lights IP65 waterproof 5050 SMD RGB+W industry standards UL certification requirements"
文本: 3 字符

✅ 正常调用了 websearch + knowledge_base
```

### 测试 2: tools=ON, response_format=ON（关键测试）

```
耗时: 3.1s
工具调用: 0 次
文本: 300 字符，输出了 JSON 但内容为空壳：
  "buyer_background": "BrightLight Inc. - 美国照明产品进口商"     ← 仅复述询盘
  "product_analysis": "LED strip lights, IP65 waterproof, ..."    ← 仅复述询盘
  "recommendations": "建议核实公司资质、确认UL认证细节..."         ← 通用话术
  "key_findings": ["公司背景需进一步验证", "UL认证是关键门槛"...]  ← 无实质信息

❌ 完全不调用工具，输出仅是"复述询盘 + 通用话术"
```

### 测试 3: tools=OFF, response_format=ON

```
耗时: 30.3s（⚠️ 非常慢）
文本: 345 字符
合法 JSON: ✅
但内容出现幻觉：
  "buyer_background": "Simulated Websearch Results" ← ⚠️ 模型在"假装"搜索过！

✅ 合法 JSON，但质量极低，模型意识到应该搜索但不能
```

---

## 4. 与 35B 的对比

| 场景 | Qwen35B-A3B | Qwen122B-A10B | 一致？ |
|------|------------|---------------|--------|
| tools=ON, rfmt=OFF | ✅ 调用工具 | ✅ 调用工具 | ✅ |
| tools=ON, rfmt=ON | ❌ 0 次 | ❌ 0 次 | ✅ |
| tools=OFF, rfmt=ON | ✅ 合法 JSON（12s, 6 card） | ✅ 合法 JSON（30s, 空壳） | ⚠️ 122B 更慢更差 |

---

## 5. 额外发现：EvoAgent 中的表现

在 `u=fz-0604-C1-T-1` 的实际 EvoAgent 测试中，122B 的表现**比 35B 更差**：

```
Step 1 think_stream:
{"blocks": [{"type": "text", "content": "收到英国客户询盘，我来帮你分析并回复。
先加载首次询盘回复的专业分析流程。"}]}

→ 0 次工具调用
→ 只输出一句"口头承诺"，然后 finish
→ 主 Agent 只得到一个 suggested_questions 卡
```

对比 35B 在同样条件下至少会输出 1 张有内容的卡，122B 只输出了空话文字。

---

## 6. 根因分析

1. **Constraint Tax 与模型规模无关**：无论是 35B 还是 122B，`response_format` 约束都会压制工具调用。论文 "The Constraint Tax"（2026）的结论在此得到验证。

2. **122B 的 "敷衍" 现象更明显**：测试 3 中输出 `"Simulated Websearch Results"`，说明模型在 schema 约束下有 "走捷径" 的强烈倾向——用虚假内容填充 schema 字段，而非调用工具获取真实数据。

3. **122B 输出更慢**：测试 3 中 345 字符耗时 30.3s，35B 同类测试 ~12s。GPTQ Int4 的解量化开销 + MoE 的 expert 调度开销可能共同导致这一现象。

---

## 7. 结论

```
🔴 方向 C（模型升级）不成立

Qwen3.5-122B-A10B-GPTQ-Int4 在 Tool Calling + Structured Output
互斥问题上的行为与 Qwen3.6-35B-A3B 完全一致。

模型规模提升不能解决这一问题 —— Constraint Tax 是
架构/训练层面的系统性偏差，不随参数量扩大而消失。

方向 A（Code-Level Research）和方向 B（Two-Agent Separation）
仍然是必需的解决路径。
```

---

## 8. 对各方向的影响

| 方向 | 影响 |
|------|------|
| 方向 A（Code-Level Research） | ✅ 仍然有效，不依赖模型能力 |
| 方向 B（Two-Agent Separation） | ✅ 仍然有效，Research/Format 分离绕过互斥 |
| 方向 C（模型升级） | ❌ 不成立 |
| 方向 D（br-2 回归） | 不推荐 |
