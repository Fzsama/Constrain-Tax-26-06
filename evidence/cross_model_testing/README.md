# Cross-Model & Cross-Framework Testing Evidence

> 来源: `ea-aim-fz-qwen-0602` 仓库 (EvoAgent 生产系统)
> 说明: 以下文件为跨模型/跨框架/Two-Pass 实验的原始证据，
> 原存放于外部仓库，现已复制至本仓库以确保审稿人和读者可独立复核。

## 文档索引

| 文件 | 内容 | 对应论文部分 |
|------|------|-------------|
| `15-constraint-tax-final-report-0609.md` | 9 模型完整矩阵 + 排除因素验证 | Table 4, §5.2-5.3 |
| `09-constraint-tax-final-conclusion.md` | 5 模型 5 轮平均对比 (含 timing) | §5.1, §5.3 |
| `08-122b-model-upgrade-test.md` | Qwen122B 详细测试 | §5.2 |
| `0604-1.txt` | 原始测试日志 (GPT-5.4-mini, Qwen35B) | - |
| `0609-1.txt` | 原始测试日志 (跨模型对照) | - |

## 关于 Two-Pass (Plan B)

Two-Pass 框架层实现位于 `ea-aim-fz-qwen-0602` 仓库的 `AIPRD-317-response-format-json-B1` 分支:

- **核心代码**: `core/agent.py` — `_InnerAgent._deferred_response_format`, `_MAX_TOOL_STEPS=6`, `_plan_b_tools_phase`
- **设计文档**: `ea-aim-fz-qwen-0602/docs/fz-analysis/` 中的 Plan B 设计文档

由于 Two-Pass 实现与 EvoAgent 生产代码深度集成，无法独立提取为独立测试脚本。
本仓库提供 `lib/two_pass.py` 作为 Two-Pass 的独立参考实现。

## 关于跨模型测试脚本

跨模型测试在 `ea-aim-fz-qwen-0602` 中通过统一测试协议执行:
- 标准参数: `temperature=0.5, stream=true, max_completion_tokens=4096`
- 每模型 3 条件 (T1/T2/T3) × 5 轮
- 检测机制: API `tool_calls` 字段 + content 文本级双重检测

详细测试协议见 `doc/14-appendix-test-design-and-tool-schema.md`。
