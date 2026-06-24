# GRPO 训练脚本技术文档

> `scripts/06_train_grpo.py` — Group Relative Policy Optimization for Constraint Tax

---

## 1. 背景与目标

### 问题定义

**Constraint Tax**: 当 `response_format` (JSON Schema) API 参数被激活时，Qwen3.6-35B-A3B 模型系统性跳过工具调用，直接输出 JSON。经 4 轮 SFT 实验证实这是解码层面的策略偏置（decoding policy bias），无法通过 SFT 修复。

### GRPO 策略

GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出的 RL 方法，核心思路：
- 每个 prompt 生成 N 个 completions
- 组内归一化计算 advantage
- 用 PPO clipped loss 优化 policy

**为什么选 GRPO 而非 DPO**:
- DPO 是二元的 (chosen vs rejected)，需要明确的偏好排序
- GRPO 利用组内相对 advantage，自动放大罕见正样本（tool_call）的信号
- 当 T2≈0% 时，DPO 需要人工构造负样本；GRPO 的 group-relative 机制天然适合处理这种极端分布

### Golden Anchor 机制

由于 T2=0%（模型在 schema 下从不调工具），纯模型生成的 completions 全为负样本（直接 JSON），组内无 variance → advantage 全为零 → 无学习信号。

**解决方案**: 每个 prompt 的组内注入 1 个 ground-truth tool_call XML 作为 "golden anchor"（reward=+1），与 K 个模型生成的 direct JSON（reward=-1）混合，确保组内始终有 positive variance。

---

## 2. 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│  输入: Tool Mandatory 数据集 (200 prompts)                    │
│                                                             │
│  Step 1 — SGLang 生成                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 200 prompts × 4 completions = 800 requests           │   │
│  │ SGLang API: tools=ON, response_format=ON, temp=0.8   │   │
│  │ ThreadPoolExecutor(16 workers) 并发请求               │   │
│  │ → 预期 T2 行为: 大多数为 direct JSON (no tool_call)   │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  Step 2 — Golden Anchor + Reward Scoring                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 每组 = 4 model gens + 1 golden XML tool_call          │   │
│  │ reward: tool_call → +1.0, direct JSON → -1.0         │   │
│  │ advantage = (reward - μ_group) / σ_group              │   │
│  │ → 1000 training samples (800 gen + 200 golden)        │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  Step 3 — QLoRA GRPO Training                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Model: Qwen3.6-35B-A3B (4-bit QLoRA, LoRA r=64)      │   │
│  │ Loss: PPO-clipped (ε=0.2) with KL penalty (β=0.1)    │   │
│  │ 3 epochs, batch=2, grad_accum=8, lr=5e-5             │   │
│  │ Best adapter saved → merge → 输出 merged model        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  输出: outputs/loraed_Qwopus3.6-35B-A3B-v1/grpo_v1/         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块

### 3.1 Golden Anchor 构造 (`build_tool_call_xml`)

将训练数据中的 tool_calls 转换为 Qwen3 原生 XML 格式：

```python
# 输入
tool_calls = [{
    "function": {
        "name": "websearch",
        "arguments": {"query": "TechCorp Inc."}
    }
}]

# 输出
"""
<tool_call>
<function=websearch>
<parameter=query>
TechCorp Inc.
</parameter>
</function>
</tool_call>
"""
```

格式来源于 Qwen3 的 Jinja2 chat template（`tokenizer_config.json`），与推理时 SGLang 的 `--tool-call-parser qwen3_coder` 保持一致。

### 3.2 Reward Function

```python
def reward_fn(has_tool_call: bool) -> float:
    return 1.0 if has_tool_call else -1.0
```

二元 reward（±1.0），仅依赖是否包含 `<tool_call>` 标签。不评估输出内容质量——目标是纯粹纠正策略偏置。

### 3.3 Group-Relative Advantage

```python
rewards_t = torch.tensor(rewards)  # [N+1] per group
if rewards_t.std() > 0:
    advantages = (rewards_t - rewards_t.mean()) / rewards_t.std()
else:
    advantages = torch.zeros_like(rewards_t)
```

关键特性：当组内所有模型生成的 completions 都是 direct JSON（reward=-1），唯一 golden anchor 是 +1 时：
- μ ≈ (-4 + 1) / 5 = -0.6
- σ > 0 (因为存在 +1 的 outlier)
- golden 的 advantage > 0，其余 < 0
- 模型被推向提高 golden completion（tool_call）的概率

### 3.4 GRPO Loss

```
ratio = exp(log P_θ(completion) - log P_ref(completion))
L = -min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)
```

实现要点：
- `P_θ`: 当前 policy（LoRA adapter enabled）
- `P_ref`: 参考 policy（LoRA adapter disabled = frozen base model）
- Log-prob 计算：batched `F.cross_entropy(ignore_index=-100)` 在 completion tokens 上
- `-100` 标记 prompt tokens 和 completion padding，确保 loss 只在 completion 区域计算
- KL penalty (β=0.1) 通过 ratio 中的 P_ref 隐式实现

### 3.5 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA r/alpha | 64/64 | 中等 rank |
| GRPO clip ε | 0.2 | PPO 标准值 |
| KL β | 0.1 | 限制策略偏离 |
| batch size | 2 | QLoRA + 35B 模型的 GPU 内存限制 |
| grad accum | 8 | effective batch = 16 |
| epochs | 3 | 1000 samples, ~188 steps/epoch |
| lr | 5e-5 | cosine schedule with 10% warmup |
| max_len | 4096 | prompt 1536 + completion 2560 |

---

## 4. 数据流

### 输入

```
data/processed/tool_mandatory_dataset.json (200 samples)
├── messages[]
│   ├── system: "You are an information extraction assistant..."
│   ├── user: "Find information about company: X..."
│   ├── assistant (tool_calls): [websearch, knowledge_base]
│   ├── tool (response): search results
│   └── assistant (JSON): {"company_name": ..., ...}
```

### 训练样本格式

```
{
    "system": str,
    "user": str,
    "completion": str,        # model gen: direct JSON, golden: XML tool_call
    "has_tool_call": bool,
    "reward": float,          # ±1.0
    "advantage": float,       # group-relative normalized
    "is_golden": bool,
}
```

### Tokenization

```
prompt:  <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
comp:    {completion}<|im_end|>
```

prompt tokens → padding value 0；completion tokens → padding value -100。

### 输出

```
outputs/loraed_Qwopus3.6-35B-A3B-v1/
├── adapter_grpo_v1/          # Best LoRA adapter
└── grpo_v1/                  # Merged full model (--merge)
```

---

## 5. SGLang 集成

### 生成请求格式

```json
{
    "model": "qw36-35b-a3b",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "tools": [...],
    "response_format": {"type": "json_schema", ...},
    "temperature": 0.8,
    "max_tokens": 512
}
```

### 并发策略

- `ThreadPoolExecutor(max_workers=16)` — 匹配 SGLang `--max-running-requests 64`
- 800 请求在 ~3 分钟内完成（~4.3 req/s）

### Tool-call 检测

```python
def has_tool_call_in_text(text: str) -> bool:
    return "<tool_call>" in text
```

双重检测：SGLang API 返回的 `tool_calls` 字段（结构化）+ `content` 中的 `<tool_call>` XML 标签（文本级）。

---

## 6. 命令行接口

```bash
python scripts/06_train_grpo.py [OPTIONS]

Options:
  --data PATH        训练数据路径 (default: tool_mandatory_dataset.json)
  --sglang_url URL   SGLang 服务地址 (default: http://localhost:8082)
  --n_gen N          每个 prompt 的模型生成数 (default: 4)
  --epochs N         训练轮数 (default: 3)
  --batch N          Batch size (default: 2)
  --grad_accum N     梯度累积步数 (default: 8)
  --lr FLOAT         学习率 (default: 5e-5)
  --merge            训练后 merge LoRA adapter 为完整模型

示例:
  # 完整训练
  python scripts/06_train_grpo.py --n_gen 4 --epochs 3 --merge

  # 快速测试 (少生成、少 epoch)
  python scripts/06_train_grpo.py --n_gen 1 --epochs 1 --batch 1 --grad_accum 1
```

---

## 7. 预期结果与判据

### 成功标志

训练后 T1/T2/T3 测试：
- **T2 > 0%**: 在 `response_format` 约束下模型开始调用工具（核心目标）
- **kl < 0.5**: 模型未过度偏离 base 行为
- **ratio_mean ≈ 1.0 → 逐步偏离 1.0**: LoRA 正在学习

### 关键监控指标

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `loss` | PPO clipped loss | 负值且绝对值递增 |
| `kl` | approx KL divergence | < 0.5 |
| `ratio_mean` | mean(P_θ/P_ref) | 初始≈1.0，逐渐偏离 |
| `clipped` | PPO clip 触发比例 | 0-20% |
| Model tool-call rate | 模型生成中 tool_call 占比 | 训练初期接近 0%，随训练提升 |

### 失败模式

- **ratio_mean 爆炸 (>10)**: 学习率过高或 KL penalty 不够
- **kl > 1.0**: 策略偏离过大，可能遗忘基础能力
- **loss 不收敛**: reward 信号无 variance（所有 sample 同 reward）
- **CUDA OOM**: 降低 batch_size 或 max_len

---

## 8. 已知限制

1. **Golden anchor 格式偏差**: 训练时 golden 使用 XML tool_call，但推理时 SGLang 通过 `--tool-call-parser qwen3_coder` 解析。两者的 token 级表示可能有微小差异。

2. **Natural language vs API schema**: 训练 prompt 的 schema 约束来自 system prompt 的自然语言描述，而推理时同时有 system prompt + `response_format` API 参数。如果 `response_format` 在模型内部注入了特殊 token，则存在 training-inference mismatch。

3. **Offline generation**: 当前采用预生成模式（SGLang 生成全部 completions 后再训练），非 true online RL。在线模式可以形成正向反馈循环（模型改进 → 生成更多 tool_call → 更多正样本），但需要 LoRA hot-swap 能力。

4. **Reference model via disable_adapter()**: 依赖 PEFT 的 `disable_adapter()` context manager 获取 base policy log-probs。如果 4-bit QLoRA 与该机制不完全兼容（已知 `TokenizersBackend` 相关问题），ratio 可能始终为 1.0。

---

## 9. 环境依赖

- **训练环境**: `lora_qwen_0612` (conda)
  - torch 2.10.0+cu128
  - transformers 5.5.0 (支持 qwen3_5_moe)
  - peft 0.19.1
  - bitsandbytes 0.49.2

- **推理/SGLang 环境**: `sglang_059` (conda)
  - SGLang 0.5.9
  - 启动参数: `--tp-size 2 --tool-call-parser qwen3_coder --reasoning-parser qwen3`
