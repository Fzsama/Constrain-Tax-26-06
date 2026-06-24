#!/usr/bin/env python3
"""A2 v2 种子数据：50:50 tools_required vs tools_optional。

tools_required: _tool_calls 数组包含 websearch + knowledge_base → 业务字段基于工具数据
tools_optional: _tool_calls 为空数组 [] → 直接基于已有信息输出业务字段
"""

import json, random, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

TARGET = 40  # 40 条种子 (20 required + 20 optional)

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

SYSTEM_PROMPT = (
    "You are an information extraction assistant. "
    "Always output a SINGLE JSON object. The first field must be _tool_calls (JSON array). "
    "If external search is needed, fill _tool_calls with tool call objects. "
    "If all needed information is already provided, use empty _tool_calls: []. "
    "Then output the retrieved or derived facts as business fields."
)

TOOLS = [
    {"type": "function", "function": {"name": "websearch", "description": "Search for company info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "knowledge_base", "description": "Query for compliance info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def build_required_sample() -> dict:
    """tools_required: 工具调用 → 基于工具数据的 JSON"""
    company, location, desc = random.choice(COMPANIES)
    market, compliance_text = random.choice(COMPLIANCES)

    user_msg = f"Find information about company: {company} and compliance requirements for {market}."

    tool_calls = [
        {"id": "call_1", "type": "function", "function": {
            "name": "websearch", "arguments": json.dumps({"query": company})}},
        {"id": "call_2", "type": "function", "function": {
            "name": "knowledge_base", "arguments": json.dumps({"query": f"{market} compliance requirements"})}},
    ]

    w_result = f"{company} — {desc}\nLocation: {location}"
    kb_result = f"{market} requirements: {compliance_text}"

    final_json = {
        "_tool_calls": [
            {"name": "websearch", "arguments": {"query": company}},
            {"name": "knowledge_base", "arguments": {"query": f"{market} compliance requirements"}},
        ],
        "company_name": company,
        "company_info": f"{desc}. Located in {location}.",
        "compliance_notes": compliance_text,
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "call_1", "name": "websearch", "content": w_result},
            {"role": "tool", "tool_call_id": "call_2", "name": "knowledge_base", "content": kb_result},
            {"role": "assistant", "content": json.dumps(final_json, ensure_ascii=False)},
        ]
    }


def build_optional_sample() -> dict:
    """tools_optional: 无需外部工具 → 直接基于已知信息输出 JSON"""
    company, location, desc = random.choice(COMPANIES)
    market, compliance_text = random.choice(COMPLIANCES)

    # Scenario where user already provides complete info
    user_msg = (
        f"Based on your existing knowledge, output company info for: {company} "
        f"(located in {location}, {desc}) and compliance notes for {market} ({compliance_text})."
    )

    final_json = {
        "_tool_calls": [],
        "company_name": company,
        "company_info": f"{desc}. Located in {location}. (Information provided in inquiry — no external search needed.)",
        "compliance_notes": f"{compliance_text} (Compliance information provided in inquiry — no knowledge_base query needed.)",
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(final_json, ensure_ascii=False)},
        ]
    }


def main():
    required_n = TARGET // 2
    optional_n = TARGET // 2

    samples = []
    for _ in range(required_n):
        samples.append(build_required_sample())
    for _ in range(optional_n):
        samples.append(build_optional_sample())
    random.shuffle(samples)

    # Validate
    ok_req = 0
    ok_opt = 0
    for s in samples:
        msgs = s["messages"]
        final = next((m for m in msgs if m["role"] == "assistant" and m.get("content", "").startswith("{")), None)
        if final:
            parsed = json.loads(final["content"])
            if "_tool_calls" in parsed:
                tc = parsed["_tool_calls"]
                if len(tc) > 0:
                    ok_req += 1
                else:
                    ok_opt += 1

    print(f"Generated: {len(samples)} samples")
    print(f"  tools_required: {ok_req}")
    print(f"  tools_optional: {ok_opt}")
    print(f"  Total valid: {ok_req + ok_opt}/{len(samples)}")

    out = _PROJECT_ROOT / "data" / "processed" / "a2_v2_seed_data.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")

    # Show samples
    print("\n=== tools_required sample ===")
    req = next(s for s in samples if len(json.loads(s["messages"][-1]["content"])["_tool_calls"]) > 0)
    print(f"  _tool_calls: {json.loads(req['messages'][-1]['content'])['_tool_calls']}")

    print("\n=== tools_optional sample ===")
    opt = next(s for s in samples if len(json.loads(s["messages"][-1]["content"])["_tool_calls"]) == 0)
    print(f"  _tool_calls: []")
    print(f"  company_info: {json.loads(opt['messages'][-1]['content'])['company_info'][:80]}")


if __name__ == "__main__":
    main()
