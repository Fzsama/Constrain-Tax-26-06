#!/usr/bin/env python3
"""GRPO 训练 (pure PyTorch + SGLang generation, no TRL).

GRPO 流程:
  1. 从 Tool Mandatory 数据提取 prompts + golden tool_calls
  2. SGLang 生成 N 个 completions (tools+schema, 预期 T2≈0%)
  3. Golden anchor: 手动构造 XML 格式 tool call 作为正样本
  4. 组内 reward → advantage，PPO clipped loss

Qwen3 tool call XML 格式:
  <tool_call>
  <function=NAME>
  <parameter=KEY>
  VALUE
  </parameter>
  </function>
  </tool_call>
"""

import argparse, json, os, sys, math
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import requests

from config import STUDENT_MODEL_PATH, STUDENT_MODEL_NAME, OUTPUTS_DIR
from dotenv import load_dotenv; load_dotenv(_PROJECT_ROOT / ".env")

CFG = {
    "lora_r": 64, "lora_alpha": 64,
    "grpo_n_gen": 4,         # model-generated completions per prompt
    "grpo_clip": 0.2,        # PPO clip
    "beta": 0.1,             # KL penalty
    "max_len": 4096,
    "max_prompt": 1536,
    "batch": 2,
    "grad_accum": 8,
    "epochs": 3,
    "lr": 5e-5,
    "warmup": 0.1,
}


# ── Qwen3 XML tool call builder ─────────────────────────────────
def build_tool_call_xml(tool_calls: list) -> str:
    """Build Qwen3 XML format tool call string from training data tool_calls."""
    parts = []
    for tc in tool_calls:
        func = tc.get("function", tc)
        name = func["name"]
        args = func["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        part = f"<tool_call>\n<function={name}>\n"
        for k, v in args.items():
            part += f"<parameter={k}>\n{v}\n</parameter>\n"
        part += f"</function>\n</tool_call>"
        parts.append(part)
    return "\n".join(parts)


# ── Reward ──────────────────────────────────────────────────────
def reward_fn(has_tool_call: bool) -> float:
    return 1.0 if has_tool_call else -1.0


def has_tool_call_in_text(text: str) -> bool:
    """Only match the actual XML tool_call tag, not JSON field names."""
    return "<tool_call>" in text


# ── SGLang generation ──────────────────────────────────────────
def generate_via_sglang(prompts, sglang_url, n_per=4, temperature=0.8, max_tokens=512, timeout=120):
    """Generate N completions per prompt via SGLang."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _gen_one(args):
        """Generate one completion."""
        p, _ = args
        payload = {
            "model": "qw36-35b-a3b",
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ],
            "tools": p["tools"],
            "response_format": p.get("response_format"),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(f"{sglang_url}/v1/chat/completions", json=payload, timeout=timeout)
            data = resp.json()
            choice = data["choices"][0]["message"]
            text = choice.get("content", "") or ""
            tc = choice.get("tool_calls") or []
            htc = bool(tc) or has_tool_call_in_text(text)
            return {"text": text, "tool_calls": tc, "has_tool_call": htc}
        except Exception as e:
            return {"text": "", "tool_calls": [], "has_tool_call": False, "_error": str(e)}

    total = len(prompts) * n_per
    # Build flat task list: (prompt_idx, generation_idx)
    tasks = [(pi, gi) for pi in range(len(prompts)) for gi in range(n_per)]

    results = [[] for _ in prompts]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_gen_one, (prompts[pi], gi)): (pi, gi) for pi, gi in tasks}
        with tqdm(total=total, desc="Generating", unit="req") as pbar:
            for future in as_completed(futures):
                pi, gi = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    r = {"text": "", "tool_calls": [], "has_tool_call": False, "_error": str(e)}
                results[pi].append(r)
                pbar.update(1)

    return results


# ── GRPO loss ──────────────────────────────────────────────────
def compute_seq_logp_fn(logits, labels, comp_mask):
    """Batched mean log-prob of completion tokens via CrossEntropy(ignore_index=-100).

    logits: [B, S, V]  — must have at least 2 positions
    labels: [B, S]     — prompt=-100, comp=token_ids, padding=-100
    comp_mask: [B, S]  — 1 for completion tokens
    Returns: mean_logp [B]
    """
    S = logits.size(1)
    if S < 2:
        return torch.zeros(logits.size(0), device=logits.device)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = comp_mask[:, 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none", ignore_index=-100,
    )
    ce = ce.reshape(shift_labels.shape)
    token_count = shift_mask.sum(dim=-1).clamp(min=1)
    return -(ce * shift_mask).sum(dim=-1) / token_count


def grpo_loss(model, prompt_ids, comp_ids, advantages, beta, clip_eps, device):
    """GRPO loss with batched CE log-prob computation."""
    B = prompt_ids.size(0)

    # Safe input_ids (replace comp padding -100→0)
    safe_comp = comp_ids.clone()
    safe_comp[safe_comp == -100] = 0
    input_ids = torch.cat([prompt_ids, safe_comp], dim=1)  # [B, P+C]

    # Labels: -100 for prompt, actual ids for completion, -100 for comp padding
    labels = torch.cat([
        torch.full_like(prompt_ids, -100),
        comp_ids.clone(),
    ], dim=1)

    # Completion token mask
    comp_mask = torch.cat([
        torch.zeros_like(prompt_ids),
        (comp_ids != -100).long(),
    ], dim=1)

    # Attention mask
    attn_mask = (input_ids != 0).long()

    # Policy logits
    pol_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

    # Reference logits (LoRA disabled = frozen base)
    with torch.no_grad():
        with model.disable_adapter():
            ref_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

    pol_lp = compute_seq_logp_fn(pol_logits, labels, comp_mask)
    ref_lp = compute_seq_logp_fn(ref_logits, labels, comp_mask)

    log_ratio = pol_lp - ref_lp
    ratio = torch.exp(log_ratio)

    # Ensure advantages on same device as ratio (handles device_map multi-GPU)
    advantages = advantages.to(ratio.device)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    with torch.no_grad():
        clipped_frac = (surr2 != surr1).float().mean().item()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

    return loss, {"loss": loss.item(), "ratio_mean": ratio.mean().item(), "kl": approx_kl, "clipped": clipped_frac}


# ── Training ───────────────────────────────────────────────────
def train(args):
    output_root = OUTPUTS_DIR / f"loraed_{STUDENT_MODEL_NAME}"
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    data_path = args.data or str(_PROJECT_ROOT / "data/processed/tool_mandatory_dataset.json")
    with open(data_path) as f:
        raw_data = json.load(f)

    prompts = []
    for item in raw_data:
        msgs = item["messages"]
        system, user, tool_calls, final_json = "", "", [], ""
        for m in msgs:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "user":
                user = m["content"]
            elif m["role"] == "assistant" and m.get("tool_calls"):
                tool_calls = m["tool_calls"]
            elif m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
                final_json = m["content"]
        prompts.append({
            "system": system,
            "user": user,
            "tool_calls": tool_calls,
            "gold_json": final_json,
            "tools": [
                {"type": "function", "function": {"name": "websearch", "description": "Search for company information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
                {"type": "function", "function": {"name": "knowledge_base", "description": "Query for compliance info",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "company_info",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "company_name": {"type": "string"},
                            "company_info": {"type": "string"},
                            "compliance_notes": {"type": "string"},
                        },
                        "required": ["company_name", "company_info", "compliance_notes"],
                        "additionalProperties": False,
                    },
                },
            },
        })

    print(f"Loaded {len(prompts)} prompts")

    # ── Step 1: Generate completions via SGLang ──
    sglang_url = args.sglang_url or "http://localhost:8082"
    n_gen = args.n_gen or CFG["grpo_n_gen"]
    print(f"Generating {n_gen} completions/prompt via SGLang ({sglang_url})...")

    all_generations = generate_via_sglang(prompts, sglang_url, n_per=n_gen, temperature=0.8)

    # ── Step 2: Build training data with golden anchor ──
    training_samples = []
    tool_call_count, total_gen = 0, 0

    for p, gens in zip(prompts, all_generations):
        # Model-generated samples
        rewards = []
        samples = []
        for g in gens:
            r = reward_fn(g["has_tool_call"])
            rewards.append(r)
            samples.append(g)
            total_gen += 1
            if g["has_tool_call"]:
                tool_call_count += 1

        # Golden anchor: XML tool call (reward +1)
        gold_text = build_tool_call_xml(p["tool_calls"])
        samples.append({"text": gold_text, "tool_calls": p["tool_calls"], "has_tool_call": True})
        rewards.append(1.0)

        # Group-relative advantage
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        if rewards_t.std() > 0:
            advantages = (rewards_t - rewards_t.mean()) / rewards_t.std()
        else:
            advantages = torch.zeros_like(rewards_t)

        for j, (g, r, adv) in enumerate(zip(samples, rewards, advantages)):
            training_samples.append({
                "system": p["system"],
                "user": p["user"],
                "completion": g["text"],
                "has_tool_call": g["has_tool_call"],
                "reward": float(r),
                "advantage": float(adv),
                "is_golden": (j == len(samples) - 1),
            })

    tc_rate = 100 * tool_call_count / total_gen if total_gen > 0 else 0
    print(f"Model tool-call rate: {tool_call_count}/{total_gen} = {tc_rate:.1f}%")
    print(f"Samples: {len(training_samples)} ({len(prompts)} golden + {total_gen} generated)")

    # ── Debug: show sample outputs ──
    n_tc = sum(1 for s in training_samples if s["has_tool_call"])
    n_notc = sum(1 for s in training_samples if not s["has_tool_call"])
    print(f"has_tool_call: {n_tc}, no_tool_call: {n_notc}")
    # Print 2 examples of each type
    for label, filter_val in [("WITH tool_call", True), ("WITHOUT tool_call", False)]:
        examples = [s for s in training_samples if s["has_tool_call"] == filter_val][:2]
        for ex in examples:
            print(f"  [{label}] golden={ex['is_golden']} text[:200]: {ex['completion'][:200]}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        batch = {"prompt_input_ids": [], "completion_input_ids": [], "advantages": []}
        for i in range(len(examples["system"])):
            prompt_text = f"<|im_start|>system\n{examples['system'][i]}<|im_end|>\n<|im_start|>user\n{examples['user'][i]}<|im_end|>\n<|im_start|>assistant\n"
            comp_text = f"{examples['completion'][i]}<|im_end|>"

            p_tok = tokenizer(prompt_text, truncation=True, max_length=CFG["max_prompt"])
            c_tok = tokenizer(comp_text, truncation=True, max_length=CFG["max_len"] - CFG["max_prompt"])

            batch["prompt_input_ids"].append(p_tok["input_ids"])
            batch["completion_input_ids"].append(c_tok["input_ids"])
            batch["advantages"].append(examples["advantage"][i])
        return batch

    ds = Dataset.from_list(training_samples)
    ds = ds.map(tokenize_fn, batched=True)

    def collate_fn(batch):
        p = [torch.tensor(b["prompt_input_ids"]) for b in batch]
        c = [torch.tensor(b["completion_input_ids"]) for b in batch]
        a = torch.tensor([b["advantages"] for b in batch])
        return {
            "prompt_ids": torch.nn.utils.rnn.pad_sequence(p, batch_first=True, padding_value=0),
            "comp_ids": torch.nn.utils.rnn.pad_sequence(c, batch_first=True, padding_value=-100),
            "advantages": a,
        }

    bs = args.batch or CFG["batch"]
    ga = args.grad_accum or CFG["grad_accum"]
    train_loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)

    # ── Model ──
    print("Loading model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    device = model.device

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or CFG["lr"])
    total_steps = len(train_loader) * (args.epochs or CFG["epochs"]) // ga
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # ── Train ──
    adapter_dir = output_root / "adapter_grpo_v1"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs or CFG["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"GRPO Epoch {epoch+1}")
        for i, batch in enumerate(pbar):
            loss, stats = grpo_loss(
                model, batch["prompt_ids"], batch["comp_ids"],
                batch["advantages"].to(device),
                CFG["beta"], CFG["grpo_clip"], device,
            )
            loss = loss / ga; loss.backward(); total_loss += loss.item()
            if (i + 1) % ga == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            pbar.set_postfix({k: f"{stats[k]:.4f}" for k in ["loss","kl","ratio_mean","clipped"]})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(adapter_dir)
            print(f"  ✓ Saved to {adapter_dir}")

    # ── Merge ──
    if args.merge:
        merge_dir = output_root / "grpo_v1"
        print(f"Merging → {merge_dir}...")
        merged = model.merge_and_unload()
        merged.save_pretrained(merge_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(merge_dir)
        print(f"✅ {merge_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--sglang_url", type=str, default="http://localhost:8082")
    p.add_argument("--n_gen", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--merge", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
