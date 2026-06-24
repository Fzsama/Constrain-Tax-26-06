#!/usr/bin/env python3
"""Phase 3: QLoRA SFT 训练。

基于 Unsloth + SFTTrainer，对 Qwen3.6-35B-A3B (Qwopus) 进行 LoRA 微调，
纠正 Constraint Tax（response_format 约束下跳过工具调用的行为缺陷）。

用法:
  # 单 GPU
  python scripts/03_train_sft.py

  # 多 GPU (DDP)
  torchrun --nproc_per_node=2 scripts/03_train_sft.py

  # 增量训练
  python scripts/03_train_sft.py --prev-adapter ./outputs/loraed_Qwopus3.6-35B-A3B/adapter_sft_v1

输出:
  - adapter: outputs/loraed_Qwopus3.6-35B-A3B/adapter_sft_v1/
  - 合并模型: outputs/loraed_Qwopus3.6-35B-A3B/sft_v1/ (--merge)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# --- 环境 ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from datasets import Dataset

from config import (
    PROCESSED_DIR, OUTPUTS_DIR,
    STUDENT_MODEL_PATH, STUDENT_MODEL_NAME,
    TRAINING_CONFIG,
)


# ============================================================
# 自定义 Tokenization（绕过 transformers 5.x chat_template 对 tool_calls 的兼容问题）
# ============================================================

def format_message_qwen(msg: dict, tokenizer) -> str:
    """手动将单条消息格式化为 Qwen chat 格式文本。

    处理三种消息类型：
    - system/user/assistant(纯文本): <|im_start|>role\ncontent<|im_end|>
    - assistant(tool_calls): <|im_start|>assistant\n<tool_call>JSON</tool_call><|im_end|>
    - tool(result): <|im_start|>tool\nJSON<|im_end|>
    """
    role = msg.get("role", "user")
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")

    # 获取特殊 token（兼容不同 tokenizer 类型）
    im_start = tokenizer.special_tokens_map.get("eos_token", "<|im_end|>")
    # Qwen 使用 <|im_end|> 既是开始也是结束标记
    im_end = im_start

    # 实际的 Qwen 格式用 \n 分隔 role 和 content
    start_marker = f"<|im_start|>{role}\n"
    end_marker = f"<|im_end|>\n"

    if tool_calls:
        # assistant 的工具调用消息
        tc_json = json.dumps(tool_calls, ensure_ascii=False)
        body = f"<tool_call>\n{tc_json}\n</tool_call>"
    elif role == "tool":
        # 工具返回消息
        tool_name = msg.get("name", "tool")
        body = json.dumps({"name": tool_name, "content": content}, ensure_ascii=False)
    else:
        body = content

    return f"{start_marker}{body}{end_marker}"


def tokenize_sample(sample: dict, tokenizer, max_length: int) -> dict:
    """将含 tool_calls 的 messages 样本 tokenize 并生成 labels mask。

    Loss 只计算 assistant 消息的 token（含 tool_calls 和最终 JSON 输出）。
    system / user / tool 消息的 label 设为 -100（忽略）。
    """
    msgs = sample["messages"]

    # 1. 手动构建完整对话文本
    parts = []
    for msg in msgs:
        parts.append(format_message_qwen(msg, tokenizer))
    full_text = "".join(parts)

    # 2. Tokenize 全文
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    full_ids = full_ids[:max_length]

    # 3. 构建 labels：初始全部 mask
    labels = [-100] * len(full_ids)

    # 4. 找到每个 assistant 消息的 token span，取消 mask
    # 策略：逐条 encode 每个消息，累加 token 偏移量，标记 assistant 部分
    offset = 0
    for msg in msgs:
        msg_text = format_message_qwen(msg, tokenizer)
        msg_ids = tokenizer.encode(msg_text, add_special_tokens=False)
        msg_len = len(msg_ids)

        if offset + msg_len > max_length:
            break

        if msg.get("role") == "assistant":
            for i in range(offset, min(offset + msg_len, max_length)):
                labels[i] = full_ids[i]

        offset += msg_len

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ============================================================
# 数据加载
# ============================================================

def load_training_data(data_path: str = None) -> Dataset:
    """加载合并后的训练数据 JSON 文件。"""
    path = data_path or str(PROCESSED_DIR / "training_data.json")
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)

    # 确保 assistant 的 content 是字符串（不是 None）
    for sample in samples:
        for msg in sample:
            if msg.get("content") is None:
                msg["content"] = ""

    ds = Dataset.from_list([{"messages": s} for s in samples])
    print(f"加载 {len(ds)} 条训练样本")
    return ds


# ============================================================
# 模型加载
# ============================================================

def load_model_and_tokenizer(args):
    """用 Unsloth 加载 QLoRA 模型和 tokenizer。"""
    from unsloth import FastLanguageModel
    from peft import LoraConfig, get_peft_model

    cfg = TRAINING_CONFIG
    lr = args.lr if args.lr else cfg["learning_rate"]

    print(f"加载模型: {STUDENT_MODEL_PATH}")
    print(f"  training_type={cfg['training_type']}, lora_r={cfg['lora_r']}, lr={lr}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=STUDENT_MODEL_PATH,
        max_seq_length=cfg["max_seq_len"],
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": "cuda:0"} if not torch.distributed.is_initialized() else None,
        trust_remote_code=True,
    )

    # 获取内部文本 tokenizer（Qwen3 可能返回 Processor）
    if not hasattr(tokenizer, "encode") and hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 增量训练：加载已有 adapter
    if args.prev_adapter:
        from peft import PeftModel
        print(f"加载已有 adapter: {args.prev_adapter}")
        model = PeftModel.from_pretrained(model, args.prev_adapter)
        model = model.merge_and_unload()

    # 应用 LoRA
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.is_parallelizable = True
    model.model_parallel = True
    model.print_trainable_parameters()

    return model, tokenizer, lr


# ============================================================
# 训练主流程
# ============================================================

def train(args):
    cfg = TRAINING_CONFIG
    output_root = OUTPUTS_DIR / f"loraed_{STUDENT_MODEL_NAME}"
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    dataset = load_training_data(args.data)
    dataset = dataset.train_test_split(test_size=0.1, seed=cfg["seed"])
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    print(f"train: {len(train_ds)}, eval: {len(eval_ds)}")

    # 2. 加载模型
    model, tokenizer, lr = load_model_and_tokenizer(args)

    # 3. Tokenize
    max_len = cfg["max_seq_len"]
    if args.max_seq_len:
        max_len = args.max_seq_len

    print(f"Tokenizing (max_seq_len={max_len})...")
    train_ds = train_ds.map(
        lambda s: tokenize_sample(s, tokenizer, max_len),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    eval_ds = eval_ds.map(
        lambda s: tokenize_sample(s, tokenizer, max_len),
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    )

    # 打印一个样本的 token 分布
    sample_labels = train_ds[0]["labels"]
    effective_tokens = sum(1 for l in sample_labels if l != -100)
    print(f"样本 0: total={len(sample_labels)}, effective={effective_tokens} "
          f"({effective_tokens/max(len(sample_labels),1)*100:.0f}%)")

    # 4. 训练
    from transformers import TrainingArguments
    from trl import SFTTrainer

    adapter_dir = output_root / "adapter_sft_v1"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs or cfg["epochs"],
        per_device_train_batch_size=args.batch or cfg["batch_size"],
        per_device_eval_batch_size=args.batch or cfg["batch_size"],
        gradient_accumulation_steps=args.grad_accum or cfg["grad_accum"],
        learning_rate=lr,
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_steps=20,
        eval_steps=20,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        seed=cfg["seed"],
    )

    # SFTTrainer：因为已预 tokenize，用 dataset_text_field=None 跳过内部格式化
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"\n{'='*60}")
    print(f"开始训练: epochs={training_args.num_train_epochs}, "
          f"batch={training_args.per_device_train_batch_size}, "
          f"grad_accum={training_args.gradient_accumulation_steps}")
    print(f"有效 batch size ≈ {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"train samples: {len(train_ds)}, eval samples: {len(eval_ds)}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model(adapter_dir)
    print(f"\n✅ Adapter 已保存: {adapter_dir}")

    # 5. 可选：合并权重
    if args.merge:
        merge_dir = output_root / "sft_v1"
        print(f"合并权重 → {merge_dir}...")
        merge_and_save(adapter_dir, merge_dir, STUDENT_MODEL_PATH)
        print(f"✅ 完整模型已保存: {merge_dir}")

    return adapter_dir


def merge_and_save(adapter_path: Path, save_path: Path, base_model_path: str):
    """合并 LoRA adapter 到基座模型并保存。"""
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base = FastLanguageModel.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()
    merged.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"合并完成: {save_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="QLoRA SFT 训练 — Constraint Tax 纠正")
    p.add_argument("--data", type=str, default=None, help="训练数据 JSON 路径")
    p.add_argument("--epochs", type=int, default=None, help="训练轮次")
    p.add_argument("--batch", type=int, default=None, help="每设备 batch 大小")
    p.add_argument("--grad_accum", type=int, default=None, help="梯度累积步数")
    p.add_argument("--lr", type=float, default=None, help="学习率")
    p.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    p.add_argument("--max_seq_len", type=int, default=None, help="最大序列长度")
    p.add_argument("--merge", action="store_true", help="训练后合并保存完整模型")
    p.add_argument("--prev_adapter", type=str, default=None, help="增量训练：已有 adapter 路径")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
