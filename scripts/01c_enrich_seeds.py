#!/usr/bin/env python3
"""后处理：为已有合成种子数据注入 tool_dependency + claim_id → evidence 链。

在已有的 syn_*.json 上运行，不重新生成。
用法: python scripts/01c_enrich_seeds.py
"""

import json, re, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
SEED_TRACES_DIR = _PROJECT_ROOT / "data" / "seeds" / "seed_traces"


def enrich_sample(data: dict) -> dict:
    """为单条训练数据注入 tool_dependency。"""
    msgs = data.get("messages", [])
    tool_results = {}
    final_idx = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            tool_results[m.get("name", "?")] = m.get("content", "")
        if m.get("role") == "assistant" and not m.get("tool_calls") and m.get("content", "").startswith("{"):
            final_idx = i

    if final_idx == -1:
        return data

    try:
        obj = json.loads(msgs[final_idx]["content"])
    except Exception:
        return data

    # Remove old tool_dependency if present
    obj.pop("tool_dependency", None)

    # Check scenario metadata for requires_tools flag
    requires = data.get("meta", {}).get("requires_tools", "true")
    if requires == "false" or not tool_results:
        obj["tool_dependency"] = {
            "required": False, "tools_used": [], "claims": [],
            "reason": "Inquiry already contains all necessary information for analysis.",
        }
    else:
        # Build claims from tool results
        claims = []
        cid = 0
        for tool_name, content in tool_results.items():
            facts = re.findall(r'(?:^|\n)(?:[\•\-\d]+\.?\s*)?([^\•\-\n]{20,120})(?:\n|$)', content)
            if not facts:
                facts = [content.strip()[:120]]
            for fact in facts[:2]:
                cid += 1
                claims.append({
                    "id": f"claim_{cid:03d}",
                    "claim": fact.strip()[:120],
                    "source": tool_name,
                    "if_missing": "cannot_determine",
                })

        # Add evidence refs to card items
        for b in obj.get("blocks", []):
            if b.get("type") != "card":
                continue
            for item in b.get("card", {}).get("list", []):
                desc = (item.get("description") or item.get("comment") or "")
                if not desc:
                    continue
                item.pop("evidence", None)  # Clean old
                matching = [c["id"] for c in claims
                    if len(set(re.findall(r'[a-zA-Z一-鿿]{3,}', c["claim"].lower())) &
                           set(re.findall(r'[a-zA-Z一-鿿]{3,}', desc.lower()))) >= 2]
                if matching:
                    item["evidence"] = matching

        obj["tool_dependency"] = {
            "required": True,
            "tools_used": list(tool_results.keys()),
            "claims": claims,
            "reason": f"买家背景和合规信息无法从询盘原文推断，必须通过{', '.join(tool_results.keys())}获取。缺少工具将导致分析不可靠。",
        }

    msgs[final_idx]["content"] = json.dumps(obj, ensure_ascii=False)
    data["messages"] = msgs
    return data


def main():
    syn_files = sorted(list(SEED_TRACES_DIR.glob("syn_*.json")) + list(SEED_TRACES_DIR.glob("syn2_*.json")))
    if not syn_files:
        print("No syn_*.json files found")
        return

    enriched = 0
    for f in syn_files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        data = enrich_sample(data)
        with open(f, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

        td = data["messages"][-1] if data["messages"] else {}
        try:
            obj = json.loads(td.get("content", "{}"))
            td_info = obj.get("tool_dependency", {})
            claims_n = len(td_info.get("claims", []))
            ev_items = sum(1 for b in obj.get("blocks", [])
                if b.get("type") == "card"
                for item in b.get("card", {}).get("list", [])
                if "evidence" in item)
            print(f"  ✅ {f.name}: required={td_info.get('required')}, claims={claims_n}, evidence_items={ev_items}")
        except:
            print(f"  ⚠️ {f.name}: parse error")
        enriched += 1

    print(f"\nEnriched {enriched} samples")


if __name__ == "__main__":
    main()
