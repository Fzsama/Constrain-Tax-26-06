#!/usr/bin/env python3
"""DPO 训练 (pure PyTorch, no TRL dependencies)"""

import argparse, json, os, sys, math
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

from config import STUDENT_MODEL_PATH, STUDENT_MODEL_NAME, OUTPUTS_DIR
from dotenv import load_dotenv; load_dotenv(_PROJECT_ROOT / ".env")

CFG = {"lora_r":64,"lora_alpha":64,"beta":0.1,"max_len":4096,"max_prompt":1024,
       "batch":2,"grad_accum":8,"epochs":3,"lr":5e-5,"warmup":0.1}

def collate_dpo(batch):
    prompts = [torch.tensor(b["prompt_input_ids"]) for b in batch]
    chosen = [torch.tensor(b["chosen_input_ids"]) for b in batch]
    rejected = [torch.tensor(b["rejected_input_ids"]) for b in batch]
    return {"prompt": torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=0),
            "chosen": torch.nn.utils.rnn.pad_sequence(chosen, batch_first=True, padding_value=-100),
            "rejected": torch.nn.utils.rnn.pad_sequence(rejected, batch_first=True, padding_value=-100),
            "p_attn": (torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=0) != 0).long(),
            "c_attn": (torch.nn.utils.rnn.pad_sequence(chosen, batch_first=True, padding_value=-100) != -100).long(),
            "r_attn": (torch.nn.utils.rnn.pad_sequence(rejected, batch_first=True, padding_value=-100) != -100).long()}

def dpo_loss(model, ref_model, batch, beta, device):
    """Compute DPO loss for one batch."""
    p_ids = batch["prompt"].to(device)
    c_ids = batch["chosen"].to(device)
    r_ids = batch["rejected"].to(device)
    p_mask = batch["p_attn"].to(device)
    c_mask = batch["c_attn"].to(device)
    r_mask = batch["r_attn"].to(device)

    # Concatenate prompt + completion for forward pass
    pc_ids = torch.cat([p_ids, c_ids], dim=1)
    pr_ids = torch.cat([p_ids, r_ids], dim=1)
    pc_mask = torch.cat([p_mask, c_mask], dim=1)
    pr_mask = torch.cat([p_mask, r_mask], dim=1)

    # Reference: use model with adapters disabled (base weights are frozen via LoRA)
    with torch.no_grad():
        if ref_model:
            ref_pc = ref_model(pc_ids.to(ref_model.device), attention_mask=pc_mask.to(ref_model.device)).logits.to(device)
            ref_pr = ref_model(pr_ids.to(ref_model.device), attention_mask=pr_mask.to(ref_model.device)).logits.to(device)
        else:
            with model.disable_adapter():
                ref_pc = model(pc_ids, attention_mask=pc_mask).logits
                ref_pr = model(pr_ids, attention_mask=pr_mask).logits

    pol_pc = model(pc_ids, attention_mask=pc_mask).logits
    pol_pr = model(pr_ids, attention_mask=pr_mask).logits

    # Log probabilities of completion tokens only
    def logp(logits, ids, mask):
        shift_logits = logits[:, :-1, :]
        shift_ids = ids[:, 1:]
        shift_mask = mask[:, 1:]
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss_per_token = ce(shift_logits.reshape(-1, shift_logits.size(-1)), shift_ids.reshape(-1))
        loss_per_token = loss_per_token.reshape(shift_ids.shape)
        return -(loss_per_token * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)

    ref_chosen_lp = logp(ref_pc, pc_ids, pc_mask)
    ref_rejected_lp = logp(ref_pr, pr_ids, pr_mask)
    pol_chosen_lp = logp(pol_pc, pc_ids, pc_mask)
    pol_rejected_lp = logp(pol_pr, pr_ids, pr_mask)

    # DPO loss
    chosen_diff = pol_chosen_lp - ref_chosen_lp
    rejected_diff = pol_rejected_lp - ref_rejected_lp
    loss = -torch.nn.functional.logsigmoid(beta * (chosen_diff - rejected_diff)).mean()
    chosen_reward = (beta * (pol_chosen_lp - ref_chosen_lp)).detach().mean()
    rejected_reward = (beta * (pol_rejected_lp - ref_rejected_lp)).detach().mean()
    return loss, chosen_reward, rejected_reward

def train(args):
    output_root = OUTPUTS_DIR / f"loraed_{STUDENT_MODEL_NAME}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Data
    path = args.data or str(_PROJECT_ROOT / "data/processed/dpo_training_data.json")
    with open(path) as f: samples = json.load(f)
    ds = Dataset.from_list(samples).train_test_split(test_size=0.1, seed=42)
    print(f"DPO: train={len(ds['train'])}, eval={len(ds['test'])}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.tokenizer = tokenizer

    def tokenize(examples):
        batch = {}
        for i in range(len(examples["prompt"])):
            p = tokenizer(examples["prompt"][i], truncation=True, max_length=CFG["max_prompt"])
            c = tokenizer(examples["chosen"][i], truncation=True, max_length=CFG["max_len"]-CFG["max_prompt"])
            r = tokenizer(examples["rejected"][i], truncation=True, max_length=CFG["max_len"]-CFG["max_prompt"])
            for k in ["prompt","chosen","rejected"]:
                ids = p if k=="prompt" else (c if k=="chosen" else r)
                batch.setdefault(f"{k}_input_ids",[]).append(ids["input_ids"])
                batch.setdefault(f"{k}_attention_mask",[]).append(ids["attention_mask"])
        return batch

    ds["train"] = ds["train"].map(tokenize, batched=True)
    ds["test"] = ds["test"].map(tokenize, batched=True)
    train_loader = DataLoader(ds["train"], batch_size=CFG["batch"], shuffle=True, collate_fn=collate_dpo)

    # Model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=0, bias="none", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # No separate ref model — use LoRA base weights as frozen reference

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or CFG["lr"])
    total_steps = len(train_loader) * (args.epochs or CFG["epochs"]) // (args.grad_accum or CFG["grad_accum"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Train
    device = model.device
    adapter_dir = output_root / "adapter_dpo_v1"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs or CFG["epochs"]):
        model.train()
        total_loss = 0; step = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(pbar):
            loss, cr, rr = dpo_loss(model, None, batch, CFG["beta"], device)
            loss = loss / (args.grad_accum or CFG["grad_accum"])
            loss.backward()
            total_loss += loss.item()

            if (i+1) % (args.grad_accum or CFG["grad_accum"]) == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "cr": f"{cr:.2f}", "rr": f"{rr:.2f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(adapter_dir)
            print(f"  Saved best to {adapter_dir}")

    # Merge
    if args.merge:
        merge_dir = output_root / "dpo_v1"
        print(f"Merging → {merge_dir}...")
        merged = model.merge_and_unload()
        merged.save_pretrained(merge_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(merge_dir)
        print(f"✅ {merge_dir}")

def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--data",type=str,default=None)
    p.add_argument("--epochs",type=int,default=None); p.add_argument("--batch",type=int,default=None)
    p.add_argument("--grad_accum",type=int,default=None); p.add_argument("--lr",type=float,default=None)
    p.add_argument("--merge",action="store_true"); return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
