#!/usr/bin/env python3
"""A2 模型 T1/T2/T3 测试 — 验证 JSON 内嵌 _tool_calls 是否突破 Constraint Tax。

A2 方案: 模型被训练输出 {"_tool_calls": [...], "company_name": "...", ...}
第一个 token 是 "{" → FSM 允许 → tool_call 信息编码在 JSON 内部
框架层解析 _tool_calls 字段即可恢复工具调用。

测试条件:
  A2-T1: tools=ON,  schema=OFF (基线: 传统工具调用)
  A2-T2: tools=ON,  schema=ON  (★关键: 含 _tool_calls 的 schema, 检测 JSON 内的 _tool_calls)
  A2-T3: tools=OFF, schema=ON  (对照: 仅 JSON 合规, 不含 _tool_calls)

用法:
  python scripts/10_test_a2_model.py
  python scripts/10_test_a2_model.py --api http://localhost:8082/v1/chat/completions --model qwen-a2
  CT_API_URL=http://localhost:8082/v1/chat/completions CT_MODEL_NAME=qwen-a2 python scripts/10_test_a2_model.py
"""

import argparse, json, os, sys, time
import requests

# 默认值 — 可通过命令行参数或环境变量覆盖
DEFAULT_API = os.environ.get("CT_API_URL", "http://localhost:8082/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("CT_MODEL_NAME", "qwen-a2")
DEFAULT_ROUNDS = 20

# ── 标准工具 ──
TOOLS = [
    {"type": "function", "function": {
        "name": "websearch", "description": "Search for company information",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "knowledge_base", "description": "Query for compliance info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
]

# ── A2 Schema (含 _tool_calls 字段) ──
A2_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "a2_response",
        "strict": True,
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
                "company_name": {"type": "string"},
                "company_info": {"type": "string"},
                "compliance_notes": {"type": "string"},
            },
            "required": ["company_name", "company_info", "compliance_notes"],
            "additionalProperties": False,
        },
    },
}

# ── System Prompt (匹配 A2 训练格式) ──
A2_SYSTEM = (
    "You are an information extraction assistant. "
    "To answer any question, you MUST first use websearch and knowledge_base to retrieve information. "
    "Then output a SINGLE JSON object containing a _tool_calls field with your tool calls "
    "encoded as a JSON array of objects (each with name and arguments), followed by the retrieved facts."
)

# ── 旧格式 System Prompt (用于 T1 基线对比) ──
OLD_SYSTEM = (
    "You are an information extraction assistant. "
    "To answer any question, you MUST first use websearch and knowledge_base to retrieve information. "
    "Then output ONLY a JSON object containing the retrieved facts."
)


def test_t1(api_url, model, rounds):
    """A2-T1: tools=ON, schema=OFF — 标准工具调用基线"""
    print(f"\n{'='*60}")
    print("A2-T1: tools=ON, schema=OFF (传统 tool_call 基线)")
    print(f"{'='*60}")

    tc_count, direct_json = 0, 0
    for i in range(rounds):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": A2_SYSTEM},
                {"role": "user", "content": f"Find info about company TestCorp-{i} and EU compliance."},
            ],
            "tools": TOOLS,
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        resp = requests.post(api_url, json=payload, timeout=120).json()
        msg = resp["choices"][0]["message"]
        tc = msg.get("tool_calls")
        content = msg.get("content") or ""

        has_tc = bool(tc and len(tc) > 0)
        is_json = content.strip().startswith("{")

        if has_tc:
            tc_count += 1
        elif is_json:
            direct_json += 1

        if i < 2:
            print(f"  [{i}] tool_calls={len(tc) if tc else 0}, content[:80]={content[:80]}")

    rate = tc_count * 100 / rounds
    print(f"  => 传统 tool_call 率: {tc_count}/{rounds} = {rate:.0f}% (direct JSON: {direct_json})")
    return rate


def test_t2(api_url, model, rounds):
    """A2-T2: tools=ON, schema=ON — ★关键测试: JSON 内嵌 _tool_calls"""
    print(f"\n{'='*60}")
    print("A2-T2: tools=ON, schema=ON (★ A2 关键: _tool_calls in JSON)")
    print(f"{'='*60}")

    tc_traditional = 0     # 传统 API tool_calls
    tc_in_valid_json = 0   # _tool_calls 出现在合法解析的 JSON 中 (可靠指标)
    tc_raw_emission = 0    # _tool_calls 出现在 raw content 中 (含无法解析的, 宽松指标)
    valid_json = 0         # 合法 JSON (json.loads 成功)
    malformed_json = 0     # JSON 格式不合法但 raw content 含 _tool_calls

    for i in range(rounds):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": A2_SYSTEM},
                {"role": "user", "content": f"Find info about company TestCorp-{i} and EU compliance."},
            ],
            "tools": TOOLS,
            "response_format": A2_SCHEMA,
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        resp = requests.post(api_url, json=payload, timeout=120).json()
        msg = resp["choices"][0]["message"]
        tc = msg.get("tool_calls")
        content = msg.get("content") or ""

        # 检测 1: 传统 API tool_calls
        has_tc = bool(tc and len(tc) > 0)

        # 检测 2: JSON 内含 _tool_calls (两级指标)
        has_tc_in_valid_json = False   # 严格: 解析成功 + _tool_calls 非空
        has_tc_raw = False             # 宽松: raw content 含 _tool_calls 模式
        tc_count = 0
        parse_ok = False
        if content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
                valid_json += 1
                parse_ok = True
                if "_tool_calls" in parsed:
                    tc_list = parsed["_tool_calls"]
                    if isinstance(tc_list, list) and len(tc_list) > 0:
                        has_tc_in_valid_json = True
                        tc_count = len(tc_list)
            except json.JSONDecodeError:
                malformed_json += 1
                pass

        # 宽松检测: raw content 含 _tool_calls 和 "name" (即使 JSON 解析失败)
        if '"name"' in content and '_tool_calls' in content:
            has_tc_raw = True

        if has_tc:
            tc_traditional += 1
        if has_tc_in_valid_json:
            tc_in_valid_json += 1
        if has_tc_raw:
            tc_raw_emission += 1

        if i < 3:
            print(f"  [{i}] trad_tc={has_tc}, json_parsed={parse_ok}, _tool_calls items: {tc_count}")
            print(f"        content[:150]={content[:150]}")

    rate_traditional = tc_traditional * 100 / rounds
    rate_strict = tc_in_valid_json * 100 / rounds   # 严格: 合法 JSON + _tool_calls
    rate_raw = tc_raw_emission * 100 / rounds        # 宽松: raw content 含 _tool_calls
    print(f"  => 传统 tool_call 率: {tc_traditional}/{rounds} = {rate_traditional:.0f}%")
    print(f"  => ★ A2 _tool_calls 严格率 (合法JSON+_tool_calls): {tc_in_valid_json}/{rounds} = {rate_strict:.0f}%")
    print(f"  => ★ A2 _tool_calls 宽松率 (raw content 模式匹配): {tc_raw_emission}/{rounds} = {rate_raw:.0f}%")
    print(f"  => 合法 JSON: {valid_json}/{rounds}  |  非法 JSON (含 _tool_calls 模式): {malformed_json}")
    return rate_strict  # 返回严格指标作为主指标


def test_t3(api_url, model, rounds):
    """A2-T3: tools=OFF, schema=ON — JSON 合规对照"""
    print(f"\n{'='*60}")
    print("A2-T3: tools=OFF, schema=ON (JSON 合规对照)")
    print(f"{'='*60}")

    valid_json = 0
    for i in range(rounds):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": A2_SYSTEM},
                {"role": "user", "content": f"Find info about company TestCorp-{i} and EU compliance."},
            ],
            "response_format": A2_SCHEMA,
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        resp = requests.post(api_url, json=payload, timeout=120).json()
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""

        try:
            json.loads(content)
            valid_json += 1
        except:
            pass

        if i < 2:
            print(f"  [{i}] content[:100]={content[:100]}")

    rate = valid_json * 100 / rounds
    print(f"  => JSON 合规率: {valid_json}/{rounds} = {rate:.0f}%")
    return rate


def main():
    parser = argparse.ArgumentParser(
        description="A2 模型 T1/T2/T3 测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="环境变量: CT_API_URL (默认 http://localhost:8082/v1/chat/completions), CT_MODEL_NAME (默认 qwen-a2)"
    )
    parser.add_argument("--api", default=DEFAULT_API,
                        help="API endpoint URL (默认: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Served model name (默认: %(default)s)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help="每条件测试轮数 (默认: %(default)s)")
    parser.add_argument("--temp", type=float, default=0.3,
                        help="Temperature (默认: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="max_tokens (默认: %(default)s)")
    args = parser.parse_args()

    print(f"A2 模型测试")
    print(f"  API:          {args.api}")
    print(f"  Model:        {args.model}")
    print(f"  每条件轮数:    {args.rounds}")
    print(f"  Temperature:  {args.temp}")
    print(f"  Max tokens:   {args.max_tokens}")

    t1 = test_t1(args.api, args.model, args.rounds)
    t2_strict = test_t2(args.api, args.model, args.rounds)
    t3 = test_t3(args.api, args.model, args.rounds)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  A2-T1 (传统 tool_call 基线): {t1:.0f}%")
    print(f"  A2-T2 (★ JSON 内嵌 _tool_calls, 严格=合法JSON): {t2_strict:.0f}%")
    print(f"  A2-T3 (JSON 合规对照): {t3:.0f}%")

    print(f"\n  📊 指标说明:")
    print(f"     T2 严格率 = _tool_calls 出现在 json.loads() 成功解析的 JSON 中")
    print(f"     T2 宽松率 = raw content 字符串匹配 _tool_calls 模式 (含 JSON 解析失败)")
    print(f"     论文应同时报告两个指标: emission rate 和 valid JSON rate")

    print(f"\n  🎯 判定:")
    if t2_strict > 0:
        print(f"     ✅ A2 方案有效! T2 严格率 = {t2_strict:.0f}%")
        print(f"        xgrammar FSM 允许 {{ → _tool_calls 通过 JSON 内嵌成功绕过")
    else:
        print(f"     ⚠️ A2 方案 T2 严格率为 {t2_strict:.0f}%")
        print(f"        但 raw content _tool_calls 检测率可能更高 (见上方宽松率)")

    print(f"\n  对比历史:")
    print(f"     旧格式 SFT (6000条): T2 = 0%")
    print(f"     GRPO:                T2 = 0%")
    print(f"     A2 (8000条):         T2 strict = {t2_strict:.0f}%")


if __name__ == "__main__":
    main()
