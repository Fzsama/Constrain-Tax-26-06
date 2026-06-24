#!/usr/bin/env python3
"""A2 方案种子数据生成：将 tool_call XML 编码进 JSON _tool_calls 字段。

A2 策略: 不直接输出裸 <tool_call>（被 FSM block），而是：
  1. 先输出 JSON 的 { 和 _tool_calls 字段
  2. 在 _tool_calls 的 string value 中嵌入完整的 tool_call XML
  3. 继续输出 JSON 的其他字段
  4. 框架层解析 _tool_calls string，提取 tool_call 并执行

格式:
  {"_tool_calls": "<tool_call>\n<function=websearch>\n<parameter=query>\nX\n</parameter>\n</function>\n<function=knowledge_base>\n...</tool_call>", "company_name": "X", "company_info": "X", "compliance_notes": "X"}
"""

import json, random, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TARGET = 30  # 30 条种子

COMPANIES = [
    ("TechCorp Inc.", "San Jose, CA", "Founded 2012, 150 employees, SaaS platform provider"),
    ("GlobalTrade Ltd.", "London, UK", "Founded 2008, 500 employees, international logistics"),
    ("MediSupply Co.", "Berlin, DE", "Founded 2015, 80 employees, medical device distributor"),
    ("GreenEnergy Solutions", "Austin, TX", "Founded 2019, 45 employees, solar panel installer"),
    ("PacificBridge Corp.", "Singapore", "Founded 2010, 300 employees, B2B trading company"),
    ("NordicSteel AB", "Stockholm, SE", "Founded 1995, 1200 employees, steel manufacturer"),
    ("Sunrise Electronics", "Shenzhen, CN", "Founded 2005, 2000 employees, PCB manufacturer"),
    ("AlpineFoods GmbH", "Zurich, CH", "Founded 2018, 60 employees, organic food exporter"),
    ("BlueOcean Logistics", "Rotterdam, NL", "Founded 2003, 350 employees, freight forwarder"),
    ("SmartParts Inc.", "Detroit, MI", "Founded 2017, 90 employees, automotive parts supplier"),
]

COMPLIANCES = [
    ("EU market", "CE marking required, REACH compliance, RoHS 2.0 mandatory"),
    ("US market", "UL certification required, FCC Part 15, FDA registration if applicable"),
    ("Middle East", "SASO/SABER certification, Halal certification if food-related"),
    ("Southeast Asia", "Import license required, local agent mandatory in some countries"),
    ("Australia", "RCM compliance, AS/NZS standards, biosecurity inspection"),
    ("South America", "Mercosur standards, INMETRO certification for Brazil, Spanish labeling"),
    ("Africa", "SONCAP for Nigeria, PVoC for Kenya, import declaration required"),
    ("Generic", "ISO 9001 recommended, product liability insurance advised"),
]

SCHEMA_3FIELD = {
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
}

TOOLS = [
    {"type": "function", "function": {
        "name": "websearch", "description": "Search for company information",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "knowledge_base", "description": "Query for compliance and regulatory info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
]


def build_tool_call_xml(tool_calls: list) -> str:
    """将 tool_calls 转为 Qwen3 XML 格式字符串。"""
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


def build_a2_output(tool_calls: list, final_json: dict) -> str:
    """构造 A2-JSON 格式输出：_tool_calls 作为 JSON array of objects。

    注意：_tool_calls 不在 schema required 中（由框架层消费后移除）。
    _tool_calls 使用 JSON 数组格式（非 XML string），因为 xgrammar FSM
    会在 string 内也拦截特殊 token <tool_call> (id=248058)。
    """
    tc_json = []
    for tc in tool_calls:
        func = tc.get("function", tc)
        args = func["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"query": str(args)}
        tc_json.append({"name": func["name"], "arguments": args})

    a2_output = {"_tool_calls": tc_json}
    a2_output.update(final_json)
    return json.dumps(a2_output, ensure_ascii=False)


def build_system_prompt() -> str:
    return (
        "You are an information extraction assistant. "
        "To answer any question, you MUST first use websearch and knowledge_base to retrieve information. "
        "Then output a SINGLE JSON object containing a _tool_calls field with your tool calls "
        "encoded as a JSON array of objects (each with name and arguments), followed by the retrieved facts.\n\n"
        "CRITICAL: The first character of your response MUST be '{'. "
        "Your tool calls go inside the _tool_calls field as a JSON array. "
        "Then output the actual data fields.\n\n"
        "Format: {\"_tool_calls\": [{\"name\": \"websearch\", \"arguments\": {\"query\": \"...\"}}], "
        "\"company_name\": \"...\", ...}"
    )


def build_scenario(is_positive: bool = True) -> dict:
    company, location, desc = random.choice(COMPANIES)
    market, compliance_text = random.choice(COMPLIANCES)

    user_msg = f"Find information about company: {company} and compliance requirements for {market}."

    # Tool calls
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {
            "name": "websearch", "arguments": json.dumps({"query": company})}},
        {"id": "call_2", "type": "function", "function": {
            "name": "knowledge_base", "arguments": json.dumps({"query": f"{market} compliance requirements"})}},
    ]

    if is_positive:
        w_result = f"{company} — {desc}\nLocation: {location}"
        kb_result = f"{market} requirements: {compliance_text}"
        final_json = {
            "company_name": company,
            "company_info": f"{desc}. Located in {location}.",
            "compliance_notes": compliance_text,
        }
    else:
        if random.random() < 0.5:
            w_result = f"No results found for '{company}'."
            kb_result = f"{market} requirements: {compliance_text}"
            final_json = {
                "company_name": "unknown",
                "company_info": f"websearch returned no results for '{company}'.",
                "compliance_notes": compliance_text,
            }
        else:
            w_result = f"{company} — {desc}\nLocation: {location}"
            kb_result = f"No compliance data found for {market}."
            final_json = {
                "company_name": company,
                "company_info": f"{desc}. Located in {location}.",
                "compliance_notes": f"unknown — knowledge_base returned no data for {market}.",
            }

    msgs = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "name": "websearch", "content": w_result},
        {"role": "tool", "tool_call_id": "call_2", "name": "knowledge_base", "content": kb_result},
        {"role": "assistant", "content": build_a2_output(tool_calls, final_json)},
    ]

    return {"messages": msgs}


def main():
    positive_n = int(TARGET * 0.8)
    negative_n = TARGET - positive_n

    samples = []
    for _ in range(positive_n):
        samples.append(build_scenario(is_positive=True))
    for _ in range(negative_n):
        samples.append(build_scenario(is_positive=False))

    random.shuffle(samples)

    # Validate
    ok = 0
    for s in samples:
        msgs = s["messages"]
        has_tc = any(m.get("tool_calls") for m in msgs)
        final = next((m for m in msgs if m["role"] == "assistant" and m.get("content", "").startswith("{")), None)
        if has_tc and final:
            parsed = json.loads(final["content"])
            if "_tool_calls" in parsed:
                ok += 1
            else:
                print(f"  ⚠ Missing _tool_calls in output")

    print(f"Generated: {len(samples)} samples, {ok} valid ({100*ok//len(samples)}%)")
    print(f"  Positive (tool has data): {positive_n}")
    print(f"  Negative (partial empty): {negative_n}")

    out = _PROJECT_ROOT / "data" / "processed" / "a2_seed_data.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")

    # Show a sample
    print(f"\n=== Sample output ===")
    final_msg = samples[0]["messages"][-1]
    print(f"  role: {final_msg['role']}")
    print(f"  content (first 300 chars): {final_msg['content'][:300]}")


if __name__ == "__main__":
    main()
