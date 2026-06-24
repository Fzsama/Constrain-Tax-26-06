#!/usr/bin/env python3
"""Phase 2: 规模训练数据生成。

策略变更：不做 few-shot 文本生成（GPT 容易混淆输出层级），
改为复用 Phase 1 的 EvoAgent 链路：生成更多询盘场景变体 →
通过 EvoAgent + GPT-5.4-mini 真实执行工具调用 → 捕获完整轨迹。

工作流程:
1. 基于 30 条种子场景生成变体（组合替换：行业/地区/产品/公司名）
2. 对每个变体场景，通过 EvoAgent SubAgentRunner 运行 GPT-5.4-mini
3. 真实执行工具调用（websearch/knowledge_base/fetchurl）
4. 质量校验 + 保存轨迹
5. train/eval split → data/processed/

用法:
  cd /root/0420-fz/lora-qwen-0612
  python scripts/02_generate_training_data.py --count 50          # 生成 50 条
  python scripts/02_generate_training_data.py --count 200         # 生成 200 条
  python scripts/02_generate_training_data.py --dry-run           # 只看场景不执行
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# --- 环境设置 ---
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
_EVOAGENT_ROOT = Path(os.getenv("EVOAGENT_ROOT", "/root/0420-fz/ea-aim-fz-qwen-0602"))
for _p in (str(_PROJECT_ROOT), str(_EVOAGENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_EVOAGENT_ROOT / ".env")

from config import (
    SEED_TRACES_DIR, GENERATED_DIR, PROCESSED_DIR, REPORTS_DIR,
    TEACHER_MODEL, TEACHER_API_BASE, TEACHER_API_KEY,
    AGENT_TYPE, AGENT_USER_ID, AGENT_TEAM_ID,
    INDUSTRIES, REGIONS, BUYER_TYPES, COMPLEXITY_LEVELS,
    TARGET_TRAIN_SIZE, EVAL_SPLIT,
    build_training_system_prompt,
)
from scripts.lib.trace_utils import convert_to_training_format, validate_tool_json_sequence
from scripts.lib.quality_check import validate_training_sample, validate_batch, generate_report

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 场景变体生成（组合替换，不调 LLM）
# ============================================================

COMPANY_NAMES = {
    "西欧": ["TechVision SAS", "Industria GmbH", "EuroComponents BV", "NordicTech AB", "Alpine Trade AG"],
    "北美": ["Pinnacle Imports", "CoreSupply Inc", "Meridian Trading Co", "Vertex Global LLC", "Atlas Distribution"],
    "中东": ["Al-Majd Trading Est", "Gulf Commercial Group", "Emirates Supply Chain", "Barakah Import LLC", "Orient Business Solutions"],
    "东南亚": ["Pacific Bridge Corp", "Mekong Trade Partners", "IndoSource International", "ASEAN Supply Ltd", "Maharaja Imports"],
    "南美": ["Mercosur Trading SA", "Andina Comercial Ltda", "BrasilGlobal Import", "Patagonia Trade Co", "Caribe Supply SA"],
    "非洲": ["Safari Trade Ltd", "AfriCore Solutions", "Nile Valley Commerce", "Ubuntu Imports Pty", "Gold Coast Enterprises"],
    "大洋洲": ["Southern Cross Trading", "Kiwi Pacific Importers", "Outback Supply Chain", "Great Southern Goods", "Blue Ocean Trade"],
}

PRODUCTS_BY_INDUSTRY = {
    "LED 照明": ["LED Panel Light 600×600 40W", "LED High Bay 150W UFO", "Solar LED Street Light 100W", "LED Track Light 30W COB", "LED Neon Flex 24V RGB"],
    "消费电子": ["Wireless Charger 15W Qi2", "Bluetooth Speaker IPX7 30W", "USB-C Docking Station 14-in-1", "Smart Watch Fitness Tracker", "Noise Cancelling Headphones ANC"],
    "机械设备": ["Hydraulic Press 200T", "Laser Cutting Machine 1000W", "Plastic Crusher 15HP", "Automatic Filling Machine 4-head", "Industrial Mixer 500L"],
    "纺织服装": ["Recycled Polyester Jacket", "Organic Cotton T-shirt 180gsm", "Bamboo Fiber Socks", "Waterproof Softshell Vest", "Linen Blend Dress Shirt"],
    "化工材料": ["Epoxy Resin Clear Casting", "Silicone Sealant RTV", "Activated Carbon Powder", "PVC Stabilizer Ca/Zn", "Polyurethane Foam System"],
    "家居用品": ["Silicone Kitchen Utensil Set 12pc", "Bamboo Cutting Board Set", "Stainless Steel Food Container", "Collapsible Laundry Basket", "Non-stick Ceramic Fry Pan"],
    "汽车配件": ["LED Headlight Bulb H7 80W", "Car Floor Mats Custom Fit", "Oxygen Sensor OE Replacement", "Turbocharger Cartridge CHRA", "Electric Power Steering Rack"],
    "医疗器械": ["Digital Thermometer Infrared", "Pulse Oximeter Fingertip", "Disposable Surgical Gown Level 3", "Wheelchair Foldable Lightweight", "Blood Glucose Monitor Kit"],
    "太阳能": ["Solar Panel 600W Bifacial", "MPPT Solar Charge Controller 100A", "Lithium Battery 48V 200Ah", "Solar Water Pump 3HP", "Portable Power Station 2000Wh"],
    "五金工具": ["Cordless Drill 21V Brushless", "Laser Level 360° Green Beam", "Digital Multimeter True RMS", "Heat Gun 2000W LCD", "Electric Chain Saw 16 inch"],
}

PERSON_NAMES = ["James Wilson", "Maria Garcia", "Ahmed Al-Rashid", "Sophie Laurent", "Chen Wei", "Michael Brown", "Priya Patel", "Lucas Silva"]


def load_seed_traces() -> List[Dict]:
    traces = []
    for f in sorted(SEED_TRACES_DIR.glob("*.json")):
        if f.is_file():
            with open(f, encoding="utf-8") as fp:
                traces.append(json.load(fp))
    return traces


def generate_scenario_variations(seed_traces: List[Dict], count: int) -> List[Dict]:
    """基于种子场景生成变体。策略：同行业换地区 / 同地区换行业 / 换产品/公司名。"""
    variations = []
    for seed_trace in seed_traces:
        meta = seed_trace.get("meta", {})
        seed_industry = meta.get("industry", "")
        seed_region = meta.get("region", "")

        # 变体 1-3: 同行业不同地区
        other_regions = [r for r in REGIONS if r[0] != seed_region]
        for region_name, countries in random.sample(other_regions, min(3, len(other_regions))):
            country = random.choice(countries)
            company = random.choice(COMPANY_NAMES.get(region_name, COMPANY_NAMES["西欧"]))
            product = random.choice(PRODUCTS_BY_INDUSTRY.get(seed_industry, ["Product X"]))
            variations.append({
                "industry": seed_industry, "region": region_name, "country": country,
                "buyer_type": random.choice(BUYER_TYPES),
                "complexity": random.choice(COMPLEXITY_LEVELS),
                "user_message": f"""请分析这封询盘:\n公司: {company}, {country}\n产品: {product}\n数量: {random.randint(500,10000)} units\n要求: FOB 价格和最佳交期\n\nBest regards,\n{random.choice(PERSON_NAMES)}""",
                "source_seed": meta.get("seed_id", ""),
            })

        # 变体 4: 同地区不同行业
        other_industries = [i for i in INDUSTRIES if i != seed_industry]
        new_industry = random.choice(other_industries)
        region_countries = next((c_list for r, c_list in REGIONS if r == seed_region), ["US"])
        country = random.choice(region_countries)
        company = random.choice(COMPANY_NAMES.get(seed_region, COMPANY_NAMES["西欧"]))
        product = random.choice(PRODUCTS_BY_INDUSTRY.get(new_industry, ["Product"]))
        variations.append({
            "industry": new_industry, "region": seed_region, "country": country,
            "buyer_type": random.choice(BUYER_TYPES),
            "complexity": random.choice(COMPLEXITY_LEVELS),
            "user_message": f"""请分析这封询盘:\n公司: {company}, {country}\n产品: {product}\n数量: {random.randint(500,10000)} units\n\n请提供报价和交期。\n{random.choice(PERSON_NAMES)}""",
            "source_seed": meta.get("seed_id", ""),
        })

    if count and len(variations) > count:
        variations = random.sample(variations, count)
    return variations


# ============================================================
# 通过 EvoAgent 采集轨迹（复用 Phase 1 可靠链路）
# ============================================================

async def collect_trace_via_evoagent(scenario: Dict, index: int, total: int, timeout: int = 300) -> Optional[Dict]:
    """通过 EvoAgent SubAgentRunner + GPT-5.4-mini 采集一条训练轨迹。"""
    gen_id = f"gen_{index:04d}"
    t_start = time.monotonic()
    label = f"[{index+1}/{total}] {gen_id} | {scenario.get('industry',''):6s} | {scenario.get('region',''):4s}"

    _orig_cwd = os.getcwd()
    os.chdir(str(_EVOAGENT_ROOT))
    try:
        from core.sub_agent import SubAgentRunner
        from utils.workspace import Workspace

        session_id = f"lora_gen_{gen_id}_{int(time.time())}"
        runner = SubAgentRunner(
            agent_type=AGENT_TYPE, user_id=AGENT_USER_ID,
            session_id=session_id, team_id=AGENT_TEAM_ID, role="owner",
        )
        runner.config.model = TEACHER_MODEL
        runner.config.api_base = TEACHER_API_BASE
        runner.config.api_key = TEACHER_API_KEY

        async with asyncio.timeout(timeout):
            async for _ in runner.run_stream(scenario["user_message"]):
                pass

        workspace = Workspace(AGENT_USER_ID)
        history_data = workspace.history_load(runner.session_id)
        if not history_data:
            print(f"  {label} ⚠️ 无历史")
            return None

        conversation = history_data.get("conversation", [])
        if not conversation:
            return None

        sample = convert_to_training_format(
            conversation,
            system_prompt_override=build_training_system_prompt(),
            truncate_tools=True, tool_max_chars=1500,
        )

        seq = validate_tool_json_sequence(sample["messages"])
        ok, issues = validate_training_sample(sample)

        elapsed = time.monotonic() - t_start
        status = "✅" if ok else "⚠️"
        print(f"  {status} {label} | tc:{seq['tool_call_count']} | {elapsed:.0f}s")

        output = {
            "meta": {
                "gen_id": gen_id, "industry": scenario.get("industry", ""),
                "region": scenario.get("region", ""), "country": scenario.get("country", ""),
                "buyer_type": scenario.get("buyer_type", ""),
                "source_seed": scenario.get("source_seed", ""),
                "model": TEACHER_MODEL, "session_id": runner.session_id,
                "tool_call_count": seq["tool_call_count"], "tool_names": seq["tool_names"],
                "passed_validation": ok, "issues": issues,
            },
            "messages": sample["messages"],
        }

        out_path = GENERATED_DIR / f"{gen_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return sample if ok else None

    except asyncio.TimeoutError:
        print(f"  ❌ {label} | timeout")
        return None
    except Exception as e:
        print(f"  ❌ {label} | {type(e).__name__}: {str(e)[:80]}")
        return None
    finally:
        os.chdir(_orig_cwd)


# ============================================================
# 批量采集 + Split
# ============================================================

async def collect_all(scenarios: List[Dict], concurrency: int = 2, timeout: int = 300) -> List[Dict]:
    semaphore = asyncio.Semaphore(concurrency)
    total = len(scenarios)

    async def bounded(scenario, index):
        async with semaphore:
            return await collect_trace_via_evoagent(scenario, index, total, timeout)

    tasks = [bounded(s, i) for i, s in enumerate(scenarios)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def split_and_save(samples: List[Dict]):
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - EVAL_SPLIT))
    for name, subset in [("train", samples[:split_idx]), ("eval", samples[split_idx:])]:
        path = PROCESSED_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in subset:
                f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(subset)} 条 → {path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: 规模训练数据生成（EvoAgent 链路）")
    p.add_argument("--count", type=int, default=None, help="生成目标数量（默认 = 种子数 × 7 ≈ 210）")
    p.add_argument("--dry-run", action="store_true", help="只生成场景预览")
    p.add_argument("--concurrency", type=int, default=2, help="并发数（默认 2）")
    p.add_argument("--timeout", type=int, default=300, help="单条超时秒数")
    return p.parse_args()


async def main():
    args = parse_args()
    seed_traces = load_seed_traces()
    target = args.count or (len(seed_traces) * 7)
    scenarios = generate_scenario_variations(seed_traces, count=target)

    print(f"Phase 2: 规模生成（EvoAgent 链路）")
    print(f"种子: {len(seed_traces)} → 变体: {len(scenarios)}")
    print(f"并发: {args.concurrency} | 超时: {args.timeout}s\n")

    if args.dry_run:
        for i, s in enumerate(scenarios[:10]):
            print(f"  {i+1}. {s['industry']} | {s['region']} | {s['country']} | {s['buyer_type']}")
        return

    t0 = time.monotonic()
    samples = await collect_all(scenarios, concurrency=args.concurrency, timeout=args.timeout)
    elapsed = time.monotonic() - t0

    all_samples = list(samples)
    for t in seed_traces:
        all_samples.append({"messages": t["messages"]})

    print(f"\n{'─'*60}")
    print(f"通过: {len(samples)}/{len(scenarios)} ({len(samples)/max(len(scenarios),1)*100:.0f}%)")
    print(f"总样本: {len(all_samples)} (生成{len(samples)} + 种子{len(seed_traces)})")
    print(f"耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    split_and_save(all_samples)

    batch_result = validate_batch(all_samples)
    print(f"\n{generate_report(batch_result)}")


if __name__ == "__main__":
    asyncio.run(main())
