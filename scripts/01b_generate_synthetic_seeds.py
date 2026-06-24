#!/usr/bin/env python3
"""Phase 1b: 合成种子训练数据生成。

替代原 Phase 1 的 EvoAgent 链路——不再依赖真实工具执行（真实 API 经常失败），
改用 GPT-5.4-mini 一次生成完整的 agent 对话轨迹（含 tool_calls + tool_results + final JSON）。

核心改进（针对 GPT 诊断的 6 个问题）：
1. Tool Grounding：prompt 强制 final JSON 必须显式引用 tool 结果
2. Tool Quality：tool 结果由 LLM 生成，模拟真实数据，无 403/空结果
3. Schema Simplification：不生成 suggested_questions，只保留 4 张核心卡片
4. Evidence Linking：expert_insight 和 ai_judgement 必须有证据来源标注
5. Empty Tool Handling：如果模拟"未查到"，final JSON 标注"信息不足"
6. Tool-First Sequence：代码强制 tool_call 在 JSON 之前

用法:
  python scripts/01b_generate_synthetic_seeds.py --count 30    # 生成 30 条
  python scripts/01b_generate_synthetic_seeds.py --count 5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import AsyncOpenAI

from config import (
    SEEDS_DIR, SEED_TRACES_DIR,
    TEACHER_MODEL, TEACHER_API_BASE, TEACHER_API_KEY,
    EXTERNAL_TOOLS, SCHEMA_DESCRIPTION,
    INDUSTRIES, REGIONS, BUYER_TYPES, COMPLEXITY_LEVELS,
    build_training_system_prompt,
)
from data.seeds.inquiry_templates import SEED_SCENARIOS

# ============================================================
# 场景变体（复用 Phase 2 的组合替换逻辑）
# ============================================================

COMPANY_NAMES = {
    "西欧": ["TechVision SAS", "Industria GmbH", "EuroComponents BV", "NordicTech AB", "Alpine Trade AG", "Mediterranea SRL"],
    "北美": ["Pinnacle Imports", "CoreSupply Inc", "Meridian Trading Co", "Vertex Global LLC", "Atlas Distribution", "Summit Sourcing"],
    "中东": ["Al-Majd Trading Est", "Gulf Commercial Group", "Emirates Supply Chain", "Barakah Import LLC", "Orient Business"],
    "东南亚": ["Pacific Bridge Corp", "Mekong Trade Partners", "IndoSource Intl", "ASEAN Supply Ltd", "Maharaja Imports"],
    "南美": ["Mercosur Trading SA", "Andina Comercial Ltda", "BrasilGlobal Import", "Patagonia Trade Co", "Caribe Supply SA"],
    "非洲": ["Safari Trade Ltd", "AfriCore Solutions", "Nile Valley Commerce", "Ubuntu Imports Pty", "Gold Coast Ent."],
    "大洋洲": ["Southern Cross Trading", "Kiwi Pacific", "Outback Supply", "Great Southern Goods", "Blue Ocean Trade"],
}

PRODUCTS = {
    "LED 照明": ["LED Panel 600×600 40W", "LED High Bay 150W", "LED Street Light 100W", "LED Track Light 30W COB", "LED Neon Flex RGB"],
    "消费电子": ["Wireless Charger 15W", "Bluetooth Speaker IPX7", "USB-C Dock 14-in-1", "ANC Headphones", "Smart Watch"],
    "机械设备": ["Hydraulic Press 200T", "Laser Cutter 1000W", "Plastic Crusher 15HP", "Filling Machine 4-head", "Industrial Mixer 500L"],
    "纺织服装": ["Recycled Polyester Jacket", "Organic Cotton T-shirt", "Bamboo Fiber Socks", "Softshell Vest", "Linen Dress Shirt"],
    "化工材料": ["Epoxy Resin Clear", "Silicone Sealant RTV", "Activated Carbon", "PVC Stabilizer Ca/Zn", "PU Foam System"],
    "家居用品": ["Silicone Utensil Set", "Bamboo Cutting Board", "SS Food Container", "Laundry Basket", "Ceramic Fry Pan"],
    "汽车配件": ["LED Headlight H7 80W", "Car Floor Mats", "Oxygen Sensor OE", "Turbocharger CHRA", "Electric Steering Rack"],
    "医疗器械": ["Digital Thermometer", "Pulse Oximeter", "Surgical Gown Level 3", "Foldable Wheelchair", "Glucose Monitor Kit"],
    "太阳能": ["Solar Panel 600W Bifacial", "MPPT Controller 100A", "Li Battery 48V 200Ah", "Solar Pump 3HP", "Power Station 2000Wh"],
    "五金工具": ["Cordless Drill 21V", "Laser Level 360°", "Digital Multimeter", "Heat Gun 2000W LCD", "Electric Chain Saw"],
}

PERSONS = ["James Wilson", "Maria Garcia", "Ahmed Al-Rashid", "Sophie Laurent", "Chen Wei", "Michael Brown", "Priya Patel", "Lucas Silva"]


def generate_scenarios(count: int, required_true: int = None, required_false: int = None) -> List[Dict]:
    """基于种子 + 组合替换生成多样询盘场景。

    required_true: target count for requires_tools=true (default: 75% of count)
    required_false: target count for requires_tools=false (default: 25% of count)
    """
    if required_true is None:
        required_true = int(count * 0.75)
    if required_false is None:
        required_false = count - required_true

    scenarios = []
    n_true = 0; n_false = 0

    def _make_scenario(requires):
        seed = random.choice(SEED_SCENARIOS)
        region_name, countries = random.choice(REGIONS)
        country = random.choice(countries)
        company = random.choice(COMPANY_NAMES.get(region_name, COMPANY_NAMES["西欧"]))
        product = random.choice(PRODUCTS.get(seed["industry"], ["Product"]))
        msg = f"""请分析这封询盘:\n公司: {company}, {country}\n产品: {product}\n数量: {random.randint(500,10000)} units\n要求: 报价和交期"""
        if requires == "false":
            msg += f"\n已知信息: {company}是{country}知名{seed['buyer_type']}，主营{seed['industry']}产品，询盘中已包含完整的产品规格和认证要求。"
        msg += f"\n\n{random.choice(PERSONS)}"
        return {"industry": seed["industry"], "region": region_name,
            "country": country, "buyer_type": random.choice(BUYER_TYPES),
            "complexity": random.choice(COMPLEXITY_LEVELS),
            "user_message": msg, "source": f"var_{seed['id']}", "requires_tools": requires}

    while n_true < required_true or n_false < required_false:
        if n_true < required_true and (n_false >= required_false or random.random() < 0.75):
            scenarios.append(_make_scenario("true")); n_true += 1
        elif n_false < required_false:
            scenarios.append(_make_scenario("false")); n_false += 1

    return scenarios[:count]


# ============================================================
# 生成 Prompt（解决 GPT 诊断的 6 个问题）
# ============================================================

def build_generation_prompt(scenario: Dict) -> str:
    """构建对话轨迹生成 prompt，含 few-shot 示例。"""
    sys_prompt = build_training_system_prompt()

    # Few-shot 示例（精简版，展示正确的格式和 grounding）
    example = """{"messages": [
  {"role":"system","content":"你是贸探·跟单专家。分析客户询盘时，必须先搜索收集外部信息，再基于真实数据输出结构化分析报告。\\n\\n## 可用工具\\n1. websearch — 搜索网络实时信息\\n2. knowledge_base — 查询外贸知识库\\n\\n## 输出格式\\n最终回复必须是JSON：{\\"blocks\\":[{\\"type\\":\\"card\\",\\"card\\":{...}},...]}\\n禁止```json围栏，禁止在JSON外输出文本。\\n\\n## 核心原则\\n- 先搜索、再分析\\n- 禁止编造\\n- 信息不足标注\\"未查到\\""},
  {"role":"user","content":"请分析这封询盘:\\n公司: LichtDesign GmbH, 德国\\n产品: LED Strip Lights, 24V, CRI>90\\n数量: 5000 meters"},
  {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"websearch","arguments":"{\\"query\\":\\"LichtDesign GmbH Germany LED lighting company\\"}"}},{"id":"call_2","type":"function","function":{"name":"knowledge_base","arguments":"{\\"query\\":\\"LED Strip Lights CE RoHS 认证德国进口要求\\"}"}}]},
  {"role":"tool","tool_call_id":"call_1","name":"websearch","content":"LichtDesign GmbH — German LED Lighting Brand\\nFounded: 2012, Munich\\nBusiness: High-end architectural LED lighting solutions, OEM/ODM manufacturing partnership with Asian suppliers\\nScale: 50-200 employees, €15M annual revenue\\nKey markets: DACH region (Germany, Austria, Switzerland)"},
  {"role":"tool","tool_call_id":"call_2","name":"knowledge_base","content":"EU LED Lighting Import Requirements:\\n1. CE-EMC (EN 55015) + CE-LVD (EN 60598) mandatory\\n2. RoHS 2.0 (2011/65/EU) compliance required\\n3. REACH regulation — SVHC substance declaration\\n4. WEEE directive — producer registration in Germany (Stiftung EAR)\\n5. Energy Label (EU 2019/2015) for lighting products\\nDDP Hamburg: importer must handle German customs clearance (EORI number required)"},
  {"role":"assistant","content":"{\\"blocks\\":[{\\"type\\":\\"card\\",\\"card\\":{\\"type\\":\\"expert_insight\\",\\"message\\":\\"AI 专家指点\\",\\"list\\":[{\\"avatarType\\":\\"business\\",\\"title\\":\\"业务专家\\",\\"description\\":\\"\\",\\"comment\\":\\"根据websearch结果，LichtDesign是2012年成立的慕尼黑高端照明品牌，年营收约1500万欧元。DDP汉堡条款意味着他们对供应链稳定性和认证合规要求极高。建议首轮报价突出CE/RoHS/REACH资质和德国市场的交付经验。\\"}]}},{\\"type\\":\\"card\\",\\"card\\":{\\"type\\":\\"detail_card\\",\\"message\\":\\"买家画像\\",\\"list\\":[{\\"title\\":\\"公司\\",\\"description\\":\\"LichtDesign GmbH\\"},{\\"title\\":\\"类型\\",\\"description\\":\\"高端照明品牌商\\"},{\\"title\\":\\"联系人\\",\\"description\\":\\"未提取\\"},{\\"title\\":\\"地区\\",\\"description\\":\\"德国慕尼黑\\"},{\\"title\\":\\"联系阶段\\",\\"description\\":\\"首次询价\\"}]}},{\\"type\\":\\"card\\",\\"card\\":{\\"type\\":\\"detail_card\\",\\"message\\":\\"需求摘要\\",\\"list\\":[{\\"title\\":\\"产品\\",\\"description\\":\\"LED Strip Lights 24V CRI>90\\"},{\\"title\\":\\"采购量\\",\\"description\\":\\"5000米\\"},{\\"title\\":\\"交期\\",\\"description\\":\\"未提取\\"},{\\"title\\":\\"核心关注\\",\\"description\\":\\"高显色指数(CRI>90)和认证合规\\"},{\\"title\\":\\"匹配程度\\",\\"description\\":\\"中等\\"}]}},{\\"type\\":\\"card\\",\\"card\\":{\\"type\\":\\"ai_judgement\\",\\"title\\":\\"AI 关键判断\\",\\"description\\":\\"- **认证合规排雷：** 根据knowledge_base结果，出口德国LED灯带需CE-EMC+CE-LVD+RoHS+REACH四重认证，DDP条款下进口商需EORI编号。\\\\n- **买家动机研判：** 根据websearch结果，LichtDesign年采购规模可观，CRI>90的高端需求表明他们对光品质有专业要求，非价格导向型询盘。\\",\\"list\\":[{\\"label\\":\\"买家意向\\",\\"value\\":\\"高\\",\\"level\\":\\"success\\"},{\\"label\\":\\"询盘真实性\\",\\"value\\":\\"高可靠\\",\\"level\\":\\"success\\"},{\\"label\\":\\"交易风险\\",\\"value\\":\\"中风险-需确认DDP清关责任划分和EORI注册\\",\\"level\\":\\"warning\\"}]}}]}"}
]}"""

    return f"""Generate a complete AI assistant conversation trace for the inquiry below, following the EXACT format shown in the example.

## Example (study this format carefully)
{example}

## New Scenario
Industry: {scenario['industry']}
Country: {scenario.get('country','')}
Buyer type: {scenario['buyer_type']}

Inquiry:
{scenario['user_message']}

## Rules (same as example)
1. Call websearch + knowledge_base BEFORE outputting final JSON
2. Tool results MUST contain realistic data (company info, certifications, policies)
3. Final JSON: expert_insight → detail_card(buyer) → detail_card(demand) → ai_judgement (4 cards only)
4. Every claim MUST cite tool evidence ("根据websearch结果..." or similar)
5. NO suggested_questions. NO "Simulated Results". NO markdown fences.

Output ONLY a JSON object with a "messages" array, identical in structure to the example."""


# ============================================================
# JSON 提取 + 校验
# ============================================================

def extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    brace = text.find("{")
    if brace >= 0:
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[brace:i+1])
    raise ValueError("无法提取 JSON")


def validate_generated(msgs: List[dict]) -> tuple[bool, List[str]]:
    """校验生成的对话轨迹。"""
    issues = []

    # 1. 必须有 tool_call
    tool_calls = []
    first_tool = -1
    first_json = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tool_calls.extend(m["tool_calls"])
            if first_tool == -1: first_tool = i
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            c = m.get("content", "")
            if c.strip().startswith("{"):
                if first_json == -1: first_json = i

    if not tool_calls:
        issues.append("NO_TOOL_CALLS")

    # 2. tool 在 JSON 之前
    if first_json > 0 and first_tool >= first_json:
        issues.append("JSON_BEFORE_TOOL")

    # 3. 最终 JSON 不能有 suggested_questions
    if first_json > 0:
        try:
            obj = json.loads(msgs[first_json]["content"])
            js = json.dumps(obj, ensure_ascii=False).lower()
            if "suggested_questions" in js:
                issues.append("HAS_SUGGESTED_QUESTIONS")
            if "blocks" not in obj:
                issues.append("NO_BLOCKS")
        except:
            issues.append("INVALID_FINAL_JSON")

    # 4. 检查 tool result 非空
    for m in msgs:
        if m.get("role") == "tool":
            c = m.get("content", "")
            if len(c) < 50:
                issues.append(f"EMPTY_TOOL_RESULT:{m.get('name','?')}")

    # 5. 检查 grounding 关键字
    if first_json > 0:
        try:
            final = msgs[first_json]["content"].lower()
            has_evidence = any(kw in final for kw in [
                "根据", "搜索结果显示", "websearch", "查询到", "来源",
                "according to", "based on", "search result", "found that"
            ])
            if not has_evidence:
                issues.append("NO_EVIDENCE_MARKER")
        except:
            pass

    return len(issues) == 0, issues


# ============================================================
# 生成
# ============================================================

async def generate_one(
    client: AsyncOpenAI, scenario: Dict, idx: int, total: int, max_retries: int = 2
) -> Optional[Dict]:
    """生成一条训练数据，失败自动重试。"""
    sid = f"syn_{idx:04d}"
    t0 = time.monotonic()
    label = f"[{idx+1}/{total}] {sid} | {scenario['industry']:6s} | {scenario['region']:4s}"

    for attempt in range(max_retries):
        try:
            prompt = build_generation_prompt(scenario)
            resp = await client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 if attempt == 0 else 0.9,
                max_completion_tokens=8192,
            )
            raw = resp.choices[0].message.content
            obj = extract_json(raw)

            # 处理 LLM 把 messages 放在不同 key 下的情况
            msgs = None
            if "messages" in obj and isinstance(obj["messages"], list):
                msgs = obj["messages"]
            else:
                for key in obj:
                    v = obj[key]
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and "role" in v[0]:
                        msgs = v; break
                    elif isinstance(v, dict) and "messages" in v:
                        msgs = v["messages"]; break
                if msgs is None and isinstance(obj, list) and len(obj) > 0:
                    if isinstance(obj[0], dict) and "role" in obj[0]:
                        msgs = obj

            if msgs is None:
                if attempt < max_retries - 1:
                    print(f"  🔄 {label} | no messages (keys: {list(obj.keys())[:3]})")
                    await asyncio.sleep(1); continue
                print(f"  ❌ {label} | 无法定位messages"); return None

            # 注入系统 prompt
            training_prompt = build_training_system_prompt()
            if msgs and msgs[0]["role"] == "system":
                msgs[0]["content"] = training_prompt
            else:
                msgs.insert(0, {"role": "system", "content": training_prompt})

            ok, issues = validate_generated(msgs)
            elapsed = time.monotonic() - t0
            tc = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))

            if ok:
                print(f"  ✅ {label} | tc:{tc} | {elapsed:.0f}s")
            else:
                if attempt < max_retries - 1:
                    print(f"  🔄 {label} | {'; '.join(issues[:2])} | retry")
                    await asyncio.sleep(1); continue
                print(f"  ⚠️ {label} | {'; '.join(issues)} | {elapsed:.0f}s")

            out = {
                "meta": {"syn_id": sid, "industry": scenario["industry"],
                    "region": scenario["region"], "country": scenario.get("country", ""),
                    "buyer_type": scenario["buyer_type"], "model": TEACHER_MODEL,
                    "passed": ok, "issues": issues,
                    "requires_tools": scenario.get("requires_tools", "true")},
                "messages": msgs,
            }
            out_path = SEED_TRACES_DIR / f"{sid}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            return {"messages": msgs} if ok else None

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  🔄 {label} | {type(e).__name__}: {str(e)[:60]}")
                await asyncio.sleep(1); continue
            print(f"  ❌ {label} | {type(e).__name__}: {str(e)[:80]}")

    return None


async def main():
    args = parse_args()
    scenarios = generate_scenarios(args.count, args.required_true, args.required_false)
    rt = sum(1 for s in scenarios if s.get('requires_tools') == 'true')
    rf = sum(1 for s in scenarios if s.get('requires_tools') == 'false')
    print(f"  required=true: {rt}, required=false: {rf}")
    print(f"场景数: {len(scenarios)}")

    if args.dry_run:
        for i, s in enumerate(scenarios[:5]):
            print(f"  {i+1}. {s['industry']} | {s['region']} | {s['country']} | {s['source']}")
        return

    client = AsyncOpenAI(base_url=TEACHER_API_BASE, api_key=TEACHER_API_KEY, timeout=120.0)
    sem = asyncio.Semaphore(3)

    async def bounded(s, i):
        async with sem:
            return await generate_one(client, s, i, len(scenarios))

    t0 = time.monotonic()
    results = await asyncio.gather(*[bounded(s, i) for i, s in enumerate(scenarios)])
    samples = [r for r in results if r is not None]

    elapsed = time.monotonic() - t0
    print(f"\n{'─'*60}")
    print(f"通过: {len(samples)}/{len(scenarios)} ({len(samples)/max(len(scenarios),1)*100:.0f}%)")
    print(f"耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1b: 合成种子训练数据")
    p.add_argument("--count", type=int, default=30, help="总生成数量")
    p.add_argument("--required-true", type=int, default=0, help="required=true 数量（默认 75% of count）")
    p.add_argument("--required-false", type=int, default=0, help="required=false 数量（默认 25% of count）")
    p.add_argument("--prefix", type=str, default="syn", help="文件前缀（默认 syn）")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
