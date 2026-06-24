#!/usr/bin/env python3
"""Phase 4: 极简 Tool Mandatory 数据集生成。

只保留 Tool → Field 因果依赖：
- 正向: tool 查到 → output 直接复制
- 负向: tool 空结果 → output 标注 "unknown"
- 不要卡片、不要外贸、不要业务分析
"""

import json, random, sys
from pathlib import Path
from openai import AsyncOpenAI
import asyncio

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")
from config import TEACHER_MODEL, TEACHER_API_BASE, TEACHER_API_KEY

TARGET = 200  # 100-200 条即可
POSITIVE_RATIO = 0.8
CONCURRENCY = 5

# 极简 Schema（3 个字段，全部来自 tool）
SCHEMA = {
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
    ("Australia", "RCM compliance, AS/NZS standards, biosecurity inspection for wood products"),
    ("South America", "Mercosur standards, INMETRO certification for Brazil, Spanish labeling"),
    ("Africa", "SONCAP for Nigeria, PVoC for Kenya, import declaration for all shipments"),
    ("Generic", "ISO 9001 recommended, product liability insurance advised"),
]


def build_system_prompt():
    return "You are an information extraction assistant. To answer any question, you MUST first use websearch and knowledge_base to retrieve information. Then output ONLY a JSON object containing the retrieved facts. If a tool returns no results, output \"unknown\" for that field."


def build_scenario(is_positive: bool) -> dict:
    """生成一个极简 Tool Mandatory 场景。

    is_positive=True: tool 返回有效数据 → output 包含数据
    is_positive=False: tool 返回空 → output 标注 unknown
    """
    company, location, desc = random.choice(COMPANIES)
    compliance_market, compliance_text = random.choice(COMPLIANCES)

    user_msg = f"Find information about company: {company} and compliance requirements for {compliance_market}."

    # System
    msgs = [{"role": "system", "content": build_system_prompt()}]
    msgs.append({"role": "user", "content": user_msg})

    # Tool calls
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "websearch", "arguments": json.dumps({"query": company})}},
        {"id": "call_2", "type": "function", "function": {
            "name": "knowledge_base", "arguments": json.dumps({"query": f"{compliance_market} compliance requirements"})}},
    ]})

    if is_positive:
        w_result = f"{company} — {desc}\nLocation: {location}"
        kb_result = f"{compliance_market} requirements: {compliance_text}"
        final_json = {
            "company_name": company,
            "company_info": f"{desc}. Located in {location}.",
            "compliance_notes": compliance_text,
        }
    else:
        # Negative: tool returns empty
        if random.random() < 0.5:
            w_result = f"No results found for '{company}'."
            kb_result = f"{compliance_market} requirements: {compliance_text}"
            final_json = {
                "company_name": "unknown",
                "company_info": f"websearch returned no results for '{company}'.",
                "compliance_notes": compliance_text,
            }
        else:
            w_result = f"{company} — {desc}\nLocation: {location}"
            kb_result = f"No compliance data found for {compliance_market}."
            final_json = {
                "company_name": company,
                "company_info": f"{desc}. Located in {location}.",
                "compliance_notes": f"unknown — knowledge_base returned no data for {compliance_market}.",
            }

    msgs.append({"role": "tool", "tool_call_id": "call_1", "name": "websearch", "content": w_result})
    msgs.append({"role": "tool", "tool_call_id": "call_2", "name": "knowledge_base", "content": kb_result})
    msgs.append({"role": "assistant", "content": json.dumps(final_json, ensure_ascii=False)})

    return {"messages": msgs}


async def generate_with_llm(client, scenario_template, n_variations):
    """Use GPT-5.4-mini to generate variations of a template scenario."""
    results = []
    for _ in range(n_variations):
        company, location, desc = random.choice(COMPANIES)
        compliance_market, compliance_text = random.choice(COMPLIANCES)
        is_pos = random.random() < POSITIVE_RATIO

        prompt = f"""Generate a training data sample in this exact format. Output ONLY the JSON object.

Format:
{{"messages": [
  {{"role":"system","content":"You are an information extraction assistant. To answer any question, you MUST first use websearch and knowledge_base to retrieve information. Then output ONLY a JSON object containing the retrieved facts. If a tool returns no results, output \\"unknown\\" for that field."}},
  {{"role":"user","content":"Find information about company: {company} and compliance requirements for {compliance_market}."}},
  {{"role":"assistant","content":"","tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"websearch","arguments":"{{\\"query\\":\\"{company}\\"}}"}}}},{{"id":"call_2","type":"function","function":{{"name":"knowledge_base","arguments":"{{\\"query\\":\\"{compliance_market} compliance requirements\\"}}"}}}}]}},
  {{"role":"tool","tool_call_id":"call_1","name":"websearch","content":"{company} — {desc}. Location: {location}."}},
  {{"role":"tool","tool_call_id":"call_2","name":"knowledge_base","content":"{compliance_market} requirements: {compliance_text}."}},
  {{"role":"assistant","content":"{{\\"company_name\\":\\"{company}\\",\\"company_info\\":\\"{desc}. Located in {location}.\\",\\"compliance_notes\\":\\"{compliance_text}\\"}}"}}
]}}

Vary the company name, location, description, compliance market, and compliance text. {"For this sample, make ONE tool return empty/no results and mark the corresponding field as 'unknown'." if not is_pos else "Make both tools return useful data."}
"""
        try:
            resp = await client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9, max_completion_tokens=4096,
            )
            raw = resp.choices[0].message.content
            # Extract JSON
            raw = raw.strip()
            if raw.startswith("```"): raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            obj = json.loads(raw)
            results.append(obj)
        except Exception as e:
            print(f"  ❌ {e}")
    return results


def main():
    positive_n = int(TARGET * POSITIVE_RATIO)
    negative_n = TARGET - positive_n

    samples = []
    # Direct template generation (faster, more reliable)
    for i in range(positive_n):
        samples.append(build_scenario(is_positive=True))
    for i in range(negative_n):
        samples.append(build_scenario(is_positive=False))

    random.shuffle(samples)

    # Validate
    ok = 0
    for s in samples:
        msgs = s["messages"]
        has_tc = any(m.get("tool_calls") for m in msgs)
        has_json = any(m.get("role") == "assistant" and not m.get("tool_calls") and m.get("content", "").startswith("{") for m in msgs)
        if has_tc and has_json:
            ok += 1

    print(f"Generated: {len(samples)} total, {ok} valid")
    print(f"  Positive (tool has data): {positive_n}")
    print(f"  Negative (partial empty): {negative_n}")

    out = _PROJECT_ROOT / "data" / "processed" / "tool_mandatory_dataset.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
