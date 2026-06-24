"""训练数据质量校验。

对每条训练轨迹进行自动化校验，确保数据质量。
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple


def validate_training_sample(sample: Dict) -> Tuple[bool, List[str]]:
    """校验单条训练数据。

    Args:
        sample: {"messages": [...]}  格式的训练样本

    Returns:
        (passed, issues): 是否通过 + 问题列表
    """
    msgs = sample.get("messages", [])
    if not msgs:
        return False, ["消息列表为空"]

    issues = []

    # ============================================================
    # 1. 结构校验
    # ============================================================
    roles = [m.get("role", "") for m in msgs]

    if "system" not in roles:
        issues.append("MISSING_SYSTEM: 缺少 system 消息")
    if roles.count("user") < 1:
        issues.append("MISSING_USER: 缺少 user 消息")
    if roles.count("assistant") < 1:
        issues.append("MISSING_ASSISTANT: 缺少 assistant 消息")

    # ============================================================
    # 2. 工具调用校验
    # ============================================================
    tool_call_count = 0
    final_json_idx = -1
    last_tool_call_idx = -1
    tool_names_called = []

    for i, msg in enumerate(msgs):
        role = msg.get("role", "")
        if role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "unknown")
                    tool_names_called.append(name)
                    tool_call_count += 1

                    # 校验 tool_call 参数
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                        # 检查参数非空
                        arg_values = [v for v in args.values() if v]
                        if not arg_values:
                            issues.append(f"EMPTY_ARGS: {name} 工具调用参数为空 (msg {i})")
                    except json.JSONDecodeError:
                        issues.append(f"INVALID_ARGS_JSON: {name} 工具调用参数不是合法 JSON (msg {i})")

                last_tool_call_idx = i

            elif msg.get("content") and _is_json_object(msg["content"]):
                if final_json_idx == -1:
                    final_json_idx = i

    if tool_call_count == 0:
        issues.append("NO_TOOL_CALLS: 整个轨迹没有任何工具调用")
    else:
        # 检查是否有重复的工具调用
        seen_queries = []
        for i, msg in enumerate(msgs):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "")
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                        query = args.get("query") or args.get("url", "")
                        key = (name, query)
                        if key in seen_queries:
                            issues.append(f"DUP_TOOL_CALL: {name}({query[:50]}) 重复调用")
                        seen_queries.append(key)
                    except json.JSONDecodeError:
                        pass

    # ============================================================
    # 3. 执行顺序校验
    # ============================================================
    if last_tool_call_idx > 0 and final_json_idx > 0 and last_tool_call_idx >= final_json_idx:
        issues.append(
            f"ORDER_VIOLATION: 工具调用(msg {last_tool_call_idx}) 不在 "
            f"JSON 输出(msg {final_json_idx}) 之前"
        )

    # ============================================================
    # 4. JSON 合法性校验
    # ============================================================
    if final_json_idx > 0:
        json_content = msgs[final_json_idx]["content"]
        try:
            obj = json.loads(json_content)
            if "blocks" not in obj:
                issues.append("NO_BLOCKS_FIELD: 最终 JSON 缺少 blocks 字段")
            elif not isinstance(obj["blocks"], list):
                issues.append("BLOCKS_NOT_LIST: blocks 字段不是数组")
            elif len(obj["blocks"]) == 0:
                issues.append("EMPTY_BLOCKS: blocks 数组为空（可能是空值填充）")
        except json.JSONDecodeError as e:
            # 尝试修复常见问题
            fixed = _try_fix_json(json_content)
            if fixed:
                try:
                    obj = json.loads(fixed)
                    if "blocks" in obj and isinstance(obj["blocks"], list) and len(obj["blocks"]) > 0:
                        issues.append("JSON_FIXED: JSON 存在格式问题但已自动修复")
                    else:
                        issues.append(f"INVALID_JSON: 最终输出不是合法 JSON: {str(e)[:80]}")
                except json.JSONDecodeError:
                    issues.append(f"INVALID_JSON: 最终输出不是合法 JSON: {str(e)[:80]}")
            else:
                issues.append(f"INVALID_JSON: 最终输出不是合法 JSON: {str(e)[:80]}")
    else:
        issues.append("NO_JSON_OUTPUT: 未找到 JSON 格式的最终输出")

    # ============================================================
    # 5. Token 长度估算
    # ============================================================
    total_chars = sum(len(m.get("content", "") or "") for m in msgs)
    if total_chars > 32000:
        issues.append(f"TOO_LONG: 总字符数 {total_chars} 可能超过 max_seq_len（8192 tokens ≈ ~32000 chars）")

    # ============================================================
    # 6. 检查是否有伪造/占位符内容
    # ============================================================
    for i, msg in enumerate(msgs):
        content = msg.get("content", "") or ""
        if "Simulated" in content and "Search" in content:
            issues.append(f"PLACEHOLDER: msg {i} 包含占位符 'Simulated Search'")
        # Only flag tool name in content if it's NOT valid JSON (evidence引用是合法的)
        if not _is_json_object(content) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            if "websearch" in content.lower() or "knowledge_base" in content.lower():
                issues.append(f"TOOL_NAME_IN_CONTENT: msg {i} 在正文中出现了工具名称（可能把工具名写成字段值）")

    return len(issues) == 0, issues


def _is_json_object(text: str) -> bool:
    """检查文本是否可能是 JSON 对象。"""
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def _try_fix_json(text: str) -> str:
    """尝试修复常见的 JSON 格式问题。

    - 移除首尾空白
    - 移除 BOM
    - 尝试提取第一个完整 JSON 对象
    """
    text = text.strip().lstrip("﻿")

    # 如果以 { 开头，找到匹配的 }
    if text.startswith("{"):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[:i + 1]

    return ""


def validate_batch(samples: List[Dict]) -> Dict:
    """批量校验训练数据。

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "pass_rate": float,
            "common_issues": {issue_type: count},
            "results": [{sample_index, passed, issues}],
        }
    """
    from collections import Counter

    results = []
    issue_counter = Counter()
    passed_count = 0

    for i, sample in enumerate(samples):
        passed, issues = validate_training_sample(sample)
        results.append({
            "sample_index": i,
            "passed": passed,
            "issues": issues,
        })
        if passed:
            passed_count += 1
        for issue in issues:
            # 提取问题类型（冒号前的部分）
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counter[issue_type] += 1

    total = len(samples)
    return {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": passed_count / total if total > 0 else 0.0,
        "common_issues": dict(issue_counter.most_common()),
        "results": results,
    }


def generate_report(batch_results: Dict) -> str:
    """生成人类可读的校验报告。"""
    lines = [
        "=" * 60,
        "训练数据质量校验报告",
        "=" * 60,
        f"总样本数:   {batch_results['total']}",
        f"通过:       {batch_results['passed']}",
        f"失败:       {batch_results['failed']}",
        f"通过率:     {batch_results['pass_rate']:.1%}",
        "",
        "常见问题:",
    ]

    if batch_results["common_issues"]:
        for issue_type, count in batch_results["common_issues"].items():
            lines.append(f"  {issue_type}: {count} 次")
    else:
        lines.append("  (无)")

    lines.append("")
    lines.append("-" * 60)

    # 列出失败的样本
    failed = [r for r in batch_results["results"] if not r["passed"]]
    if failed:
        lines.append(f"\n失败样本详情 ({len(failed)} 条):")
        for r in failed[:10]:  # 最多显示 10 条
            lines.append(f"\n  样本 #{r['sample_index']}:")
            for issue in r["issues"]:
                lines.append(f"    - {issue}")
        if len(failed) > 10:
            lines.append(f"\n  ... 还有 {len(failed) - 10} 条失败样本")

    return "\n".join(lines)
