"""Constraint Tax 验证 — 微调后 Qwen3.6-35B-A3B LoRA 模型

T1: tools=ON,  rfmt=OFF  → 基线工具率（期望≥80%）
T2: tools=ON,  rfmt=ON   → ★关键测试（微调前0%，期望≥80%）
T3: tools=OFF, rfmt=ON   → JSON合规对照（期望≥80%）
"""
import asyncio, json, time
from openai import AsyncOpenAI

TOOLS = [
    {"type": "function", "function": {
        "name": "websearch",
        "description": "Search the web for real-time information about companies and markets",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "knowledge_base",
        "description": "Query foreign trade knowledge base for industry standards",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
]

RFMT = {
    "type": "json_schema",
    "json_schema": {
        "name": "inquiry_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "buyer_background": {"type": "string"},
                "product_analysis": {"type": "string"},
                "recommendations": {"type": "string"},
                "key_findings": {"items": {"type": "string"}, "type": "array"},
            },
            "required": ["buyer_background", "product_analysis", "recommendations", "key_findings"],
            "additionalProperties": False,
        },
    },
}

SYS = "你是外贸询盘分析助手。收到客户询盘后:\n1.使用 websearch 搜索买家公司背景\n2.使用 knowledge_base 查询产品行业标准\n3.基于调研结果给出分析"
USR = "请分析这封询盘:\n公司: BrightLight Inc. 美国照明产品进口商\n产品: LED strip lights, IP65 waterproof, 5050 SMD, RGB+W, 5m/reel\n数量: 2000 reels\n要求: FOB报价, UL listed"

API_BASE = "http://127.0.0.1:8082/v1"
API_KEY = "EMPTY"
MODEL = "qw36-35b-a3b-ct-lora"
ROUNDS = 5


async def test(label, tools_on, rfmt_on, rounds=ROUNDS):
    results = {"tools": [], "len": [], "time": [], "tc_names": [], "valid_json": []}
    errors = []

    for i in range(rounds):
        client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY, timeout=120.0)
        t0 = time.monotonic()
        kwargs = dict(
            model=MODEL,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": USR}],
            stream=True, max_completion_tokens=4096, temperature=0.5,
        )
        if tools_on:
            kwargs["tools"] = TOOLS
        if rfmt_on:
            kwargs["response_format"] = RFMT

        try:
            resp = await client.chat.completions.create(**kwargs)
            content = ""
            tcs = []
            async for chunk in resp:
                if not chunk.choices: continue
                d = chunk.choices[0].delta
                if d.tool_calls:
                    for tc in d.tool_calls:
                        found = False
                        for t in tcs:
                            if t["idx"] == tc.index:
                                t["name"] = t["name"] or (tc.function.name or "")
                                t["args"] += tc.function.arguments or ""
                                found = True; break
                        if not found:
                            tcs.append({"idx": tc.index, "name": tc.function.name or "", "args": tc.function.arguments or ""})
                if d.content:
                    content += d.content

            elapsed = time.monotonic() - t0
            results["tools"].append(len(tcs))
            results["len"].append(len(content))
            results["time"].append(elapsed)
            tc_str = "+".join(t["name"] for t in tcs) if tcs else "none"
            results["tc_names"].append(tc_str)
            try:
                if content.strip(): json.loads(content); results["valid_json"].append(True)
                else: results["valid_json"].append(False)
            except json.JSONDecodeError:
                results["valid_json"].append(False)
        except Exception as e:
            results["tools"].append(0); results["len"].append(0)
            results["time"].append(time.monotonic()-t0)
            results["tc_names"].append("ERROR"); results["valid_json"].append(False)
            errors.append(str(e)[:200])

    avg_t = sum(results["tools"])/rounds
    avg_l = sum(results["len"])/rounds
    avg_s = sum(results["time"])/rounds
    tool_rate = sum(1 for t in results["tools"] if t>0)/rounds*100
    json_rate = sum(1 for v in results["valid_json"] if v)/rounds*100

    print(f"\n  [{label}] tools={'ON' if tools_on else 'OFF'} rfmt={'ON' if rfmt_on else 'OFF'}")
    print(f"  {rounds}轮: tools={results['tools']} tc={results['tc_names']}")
    print(f"  content len={results['len']} time={[f'{x:.1f}s' for x in results['time']]}")
    print(f"  => 工具率={tool_rate:.0f}% JSON率={json_rate:.0f}% 平均{avg_t:.1f}次 {avg_l:.0f}字 {avg_s:.1f}s")
    if errors: print(f"  ⚠️ {errors[0][:200]}")

    return {"tools": avg_t, "len": avg_l, "time": avg_s, "tool_rate": tool_rate, "json_rate": json_rate}


async def main():
    print(f"Constraint Tax 验证 — LoRA 微调模型: {MODEL}")
    print(f"微调前基线: T1=100%, T2=0%, T3=100%")
    print("="*70)

    t1 = await test("T1: 基线(tools+rfmt)",  True, False)
    t2 = await test("T2: ★关键★(tools+rfmt)", True, True)
    t3 = await test("T3: 对照(仅rfmt)",       False, True)

    print(f"\n{'='*70}")
    print(f"📊 对比结果")
    print(f"{'测试':<25} {'微调前':<12} {'微调后':<12} {'变化':<10}")
    print("-"*62)
    for label, before, r in [
        ("T1 工具率(基线)",     "100%", t1),
        ("T2 工具率(关键)",     "0%",   t2),
        ("T3 JSON率(对照)",     "100%", t3),
    ]:
        after = f"{r['tool_rate']:.0f}%" if "JSON" not in label else f"{r['json_rate']:.0f}%"
        direction = "✅ 保持" if (before == "100%" and after == "100%") or (before == "0%" and after != "0%") else "⚠️"
        print(f"{label:<25} {before:<12} {after:<12} {direction}")

    print(f"\n🔍 判定:")
    if t2["tool_rate"] >= 80:
        print(f"   🎉 Constraint Tax 已修复！T2 工具率={t2['tool_rate']:.0f}%")
    elif t2["tool_rate"] >= 40:
        print(f"   🟡 部分改善。T2 工具率={t2['tool_rate']:.0f}%（微调前 0%）")
    else:
        print(f"   🔴 未显著改善。T2 工具率={t2['tool_rate']:.0f}%（微调前 0%）")


if __name__ == "__main__":
    asyncio.run(main())
