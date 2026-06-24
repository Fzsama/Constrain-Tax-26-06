#!/usr/bin/env python3
"""T1/T2/T3 test using vLLM Python API with GRPO LoRA adapter."""
import os, sys, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Must set LD_LIBRARY_PATH before importing vllm
os.environ["LD_LIBRARY_PATH"] = "/root/miniforge3/envs/vllm_0605/lib/python3.12/site-packages/nvidia/cu13/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

STUDENT_MODEL_PATH = "/root/.cache/modelscope/hub/models/Jackrong/Qwopus3.6-35B-A3B-v1"
ADAPTER_PATH = "/root/0420-fz/lora-qwen-0612/outputs/loraed_Qwopus3.6-35B-A3B-v1/adapter_grpo_v1"
N_TESTS = 20

SYSTEM = "You are an information extraction assistant. To answer any question, you MUST first use websearch and knowledge_base to retrieve information. Then output ONLY a JSON object containing the retrieved facts."

TOOLS = [
    {"type": "function", "function": {"name": "websearch", "description": "Search for company info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "knowledge_base", "description": "Query for compliance info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "company_info": {"type": "string"},
        "compliance_notes": {"type": "string"},
    },
    "required": ["company_name", "company_info", "compliance_notes"],
    "additionalProperties": False,
}


def run_test(llm, lora_req, tools, guided_json, label, n=N_TESTS):
    """Run N test queries and count tool calls."""
    prompts = []
    for i in range(n):
        user = f"Find information about company: TestCorp-{i} and compliance requirements for EU market."
        # Format as chat
        text = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(text)

    sampling = SamplingParams(temperature=0.8, max_tokens=256)

    tool_calls = 0
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        has_tc = "<tool_call>" in text
        if has_tc:
            tool_calls += 1
        if i < 2:
            print(f"  [{label}] sample {i}: tool_call={has_tc} text[:100]={text[:100]}")

    rate = 100 * tool_calls / n
    print(f"  [{label}] tool_call rate: {tool_calls}/{n} = {rate:.0f}%")
    return rate


def main():
    print("Loading vLLM with LoRA...")
    llm = LLM(
        model=STUDENT_MODEL_PATH,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=2,
        max_model_len=8192,
    )

    # Load GRPO adapter
    lora_req = LoRARequest("grpo", 1, ADAPTER_PATH)

    print("\n=== T1: tools=ON, schema=OFF ===")
    t1 = run_test(llm, lora_req, TOOLS, None, "T1-grpo")

    print("\n=== T2: tools=ON, schema=ON (guided_json) ===")
    # Use guided_json as vLLM's equivalent of response_format
    sampling_json = SamplingParams(temperature=0.8, max_tokens=256,
                                    guided_decoding=type('', (), {'json': JSON_SCHEMA, 'backend': 'outlines'})())
    # T2: tools + JSON schema
    prompts_t2 = []
    for i in range(N_TESTS):
        user = f"Find information about company: TestCorp-{i} and compliance requirements for EU market."
        prompts_t2.append(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")

    tool_calls_t2 = 0
    outputs_t2 = llm.generate(prompts_t2, sampling_json, lora_request=lora_req)
    for i, out in enumerate(outputs_t2):
        text = out.outputs[0].text
        has_tc = "<tool_call>" in text
        if has_tc:
            tool_calls_t2 += 1
        if i < 2:
            print(f"  [T2-grpo] sample {i}: tool_call={has_tc} text[:100]={text[:100]}")
    rate_t2 = 100 * tool_calls_t2 / N_TESTS
    print(f"  [T2-grpo] tool_call rate: {tool_calls_t2}/{N_TESTS} = {rate_t2:.0f}%")

    print(f"\n=== SUMMARY ===")
    print(f"T1 (tools ON,  schema OFF): {t1:.0f}%")
    print(f"T2 (tools ON,  schema ON):  {rate_t2:.0f}%")


if __name__ == "__main__":
    main()
