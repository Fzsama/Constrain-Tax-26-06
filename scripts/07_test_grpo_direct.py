#!/usr/bin/env python3
"""Quick T1/T2 test for GRPO adapter — direct model loading, no server needed."""
import sys, json
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config import STUDENT_MODEL_PATH
from dotenv import load_dotenv; load_dotenv(_PROJECT_ROOT / ".env")

ADAPTER_PATH = str(_PROJECT_ROOT / "outputs/loraed_Qwopus3.6-35B-A3B-v1/adapter_grpo_v1")
N_TESTS = 20  # test prompts per condition

TOOLS = [
    {"type": "function", "function": {"name": "websearch", "description": "Search company info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "knowledge_base", "description": "Query compliance info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "company_info", "strict": True,
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
}

def run_test(model, tokenizer, tools, response_format, label):
    """Run N_TESTS and count tool_call outputs."""
    tool_calls = 0
    for i in range(N_TESTS):
        msgs = [
            {"role": "system", "content": "You are an information extraction assistant. To answer any question, you MUST first use websearch and knowledge_base to retrieve information. Then output ONLY a JSON object containing the retrieved facts."},
            {"role": "user", "content": f"Find information about company: TestCorp-{i} and compliance requirements for EU market."},
        ]
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            msgs, tools=tools, add_generation_prompt=True,
            tokenize=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=256, temperature=0.8,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the generated part
        gen_tokens = outputs[0][inputs.shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        has_tc = "<tool_call>" in text
        if has_tc:
            tool_calls += 1
        if i < 2:
            print(f"  [{label}] sample {i}: tool_call={has_tc} text[:120]={text[:120]}")

    rate = 100 * tool_calls / N_TESTS
    print(f"  [{label}] tool_call rate: {tool_calls}/{N_TESTS} = {rate:.0f}%")
    return rate


def main():
    print("Loading base model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH, trust_remote_code=True)

    print("Loading GRPO adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print("\n=== BASE MODEL (no adapter) ===")
    model.disable_adapter()
    t1_base = run_test(model, tokenizer, TOOLS, None, "T1-base")
    model.enable_adapter()

    print("\n=== GRPO MODEL ===")
    t1_grpo = run_test(model, tokenizer, TOOLS, None, "T1-grpo")

    # Note: T2 with response_format can't be tested this way because
    # transformers' generate() doesn't support response_format API param.
    # We can only test T1 (tools=ON, schema=OFF) for now.
    print(f"\n=== SUMMARY ===")
    print(f"T1 base: {t1_base:.0f}%  |  T1 grpo: {t1_grpo:.0f}%")

    print("\n⚠ T2 (tools+schema) requires SGLang/vLLM server — not testable directly.")


if __name__ == "__main__":
    main()
