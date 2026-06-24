"""轨迹提取与格式转换工具。

功能：
- 从 EvoAgent SubAgentRunner 执行结果提取 conversation history
- 将 EvoAgent history 转换为 SFT 训练数据格式
- 过滤内部工具调用，保留外部工具有用轨迹
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Dict, List, Optional

# ============================================================
# 外部工具白名单（这些工具调用和结果保留在训练数据中）
# ============================================================
EXTERNAL_TOOL_NAMES = {"websearch", "knowledge_base", "fetchurl"}

# ============================================================
# 内部工具黑名单关键词（这些工具调用和结果从训练数据中移除）
# ============================================================
INTERNAL_TOOL_KEYWORDS = [
    "skill_manager",
    "update_user_profile",
    "update_memory",
    "todo_manager",
    "ask_user_question",
    "record_team_experience",
    "sub_agent",
]


def is_external_tool(name: str) -> bool:
    """判断是否是外部工具（需要在训练数据中保留）。"""
    return name in EXTERNAL_TOOL_NAMES


def is_internal_tool(name: str) -> bool:
    """判断是否是内部工具（需要从训练数据中移除）。"""
    for kw in INTERNAL_TOOL_KEYWORDS:
        if kw in name.lower():
            return True
    return False


def filter_tool_messages(messages: List[dict]) -> List[dict]:
    """过滤内部工具调用和结果，只保留外部工具（websearch/knowledge_base/fetchurl）。

    处理逻辑：
    1. 移除 role=tool 且 name 是内部工具的消息
    2. 移除 assistant 消息中仅包含内部工具 tool_calls 的消息
    3. 对于同时包含内外工具的 assistant 消息，只保留外部 tool_calls
    """
    # Pass 1: 收集要移除的 tool_call_id 列表
    internal_tool_ids = set()
    for msg in messages:
        if msg.get("role") == "tool" and is_internal_tool(msg.get("name", "")):
            internal_tool_ids.add(msg.get("tool_call_id", ""))

    # Pass 2: 过滤和清理
    filtered = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "tool":
            # 移除内部工具的 tool_result
            if msg.get("tool_call_id") in internal_tool_ids:
                continue
            filtered.append(msg)

        elif role == "assistant" and msg.get("tool_calls"):
            # 过滤 tool_calls
            external_calls = [
                tc for tc in msg["tool_calls"]
                if tc.get("function", {}).get("name") in EXTERNAL_TOOL_NAMES
            ]
            # 也检查 tool_call id 不在内部列表中
            external_calls = [
                tc for tc in external_calls
                if tc.get("id") not in internal_tool_ids
            ]

            if external_calls:
                msg_copy = deepcopy(msg)
                msg_copy["tool_calls"] = external_calls
                filtered.append(msg_copy)
            # 如果过滤后没有外部 tool_calls，检查是否有 content
            elif msg.get("content"):
                msg_copy = deepcopy(msg)
                msg_copy.pop("tool_calls", None)
                filtered.append(msg_copy)
            # 纯粹的内部工具调用，跳过

        elif role == "assistant" and not msg.get("tool_calls"):
            # 普通 assistant 消息（含最终 JSON 输出），保留
            filtered.append(msg)

        else:
            # system / user 消息，保留
            filtered.append(msg)

    return filtered


def extract_system_prompt(messages: List[dict]) -> Optional[str]:
    """从消息列表中提取第一条 system 消息。"""
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return None


def deduplicate_system_messages(messages: List[dict]) -> List[dict]:
    """只保留第一条 system 消息，移除后续重复注入。"""
    result = []
    found_system = False
    for msg in messages:
        if msg.get("role") == "system":
            if not found_system:
                result.append(msg)
                found_system = True
            # 跳过后续 system 消息
        else:
            result.append(msg)
    return result


def validate_tool_json_sequence(messages: List[dict]) -> Dict:
    """验证训练数据的工具调用和 JSON 输出顺序。

    返回: {
        "valid": bool,
        "tool_call_count": int,
        "has_final_json": bool,
        "tools_before_json": bool,
        "tool_names": List[str],
        "issues": List[str],
    }
    """
    issues = []
    tool_call_count = 0
    tool_names = []
    final_json_idx = -1
    last_tool_call_idx = -1

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_call_count += 1
                    name = tc.get("function", {}).get("name", "unknown")
                    tool_names.append(name)
                last_tool_call_idx = i
            elif msg.get("content") and _looks_like_json(msg["content"]):
                if final_json_idx == -1:  # 只记录第一个 JSON 输出
                    final_json_idx = i

    has_final_json = final_json_idx > 0
    tools_before_json = last_tool_call_idx < final_json_idx if (last_tool_call_idx > 0 and has_final_json) else True

    if tool_call_count == 0:
        issues.append("无工具调用")
    if not has_final_json:
        issues.append("无最终 JSON 输出")
    if last_tool_call_idx > 0 and has_final_json and last_tool_call_idx >= final_json_idx:
        issues.append(f"工具调用(索引{last_tool_call_idx})在 JSON 输出(索引{final_json_idx})之后")

    return {
        "valid": tool_call_count > 0 and has_final_json and tools_before_json,
        "tool_call_count": tool_call_count,
        "has_final_json": has_final_json,
        "tools_before_json": tools_before_json,
        "tool_names": tool_names,
        "issues": issues,
    }


def _looks_like_json(content: str) -> bool:
    """检查字符串是否看起来像 JSON 对象。"""
    stripped = content.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def truncate_tool_results(messages: List[dict], max_chars: int = 1500) -> List[dict]:
    """截断过长的 tool_result 内容。

    保留前 ~max_chars 字符，添加截断标记。
    确保训练数据不超过 max_seq_len。
    """
    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > max_chars:
                truncated = content[:max_chars] + f"\n\n... (已截断，原{len(content)}字符)"
                result.append({**msg, "content": truncated})
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


def convert_to_training_format(
    messages: List[dict],
    system_prompt_override: Optional[str] = None,
    truncate_tools: bool = True,
    tool_max_chars: int = 1500,
) -> Dict:
    """将 EvoAgent conversation history 转换为 SFT 训练格式。

    Args:
        messages: EvoAgent 原始 history 列表
        system_prompt_override: 如果提供，用此替换原始 system prompt

    Returns:
        {"messages": [...]}  标准训练数据格式
    """
    # 1. 过滤内部工具
    filtered = filter_tool_messages(messages)

    # 2. 截断过长 tool_result
    if truncate_tools:
        filtered = truncate_tool_results(filtered, max_chars=tool_max_chars)

    # 3. 去重 system 消息
    filtered = deduplicate_system_messages(filtered)

    # 4. 如果指定了 system prompt 覆盖，替换第一条
    if system_prompt_override and filtered and filtered[0]["role"] == "system":
        filtered[0] = {**filtered[0], "content": system_prompt_override}

    # 5. 确保以 system 消息开头
    if not filtered or filtered[0]["role"] != "system":
        filtered.insert(0, {
            "role": "system",
            "content": system_prompt_override or "",
        })

    # 6. 清理 tool_call id 引用一致性
    # 确保每个 tool 消息的 tool_call_id 在之前的 assistant tool_calls 中出现过
    valid_tc_ids = set()
    for msg in filtered:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                valid_tc_ids.add(tc.get("id", ""))

    cleaned = []
    for msg in filtered:
        if msg.get("role") == "tool":
            if msg.get("tool_call_id") in valid_tc_ids:
                cleaned.append(msg)
            # 跳过孤儿 tool_result（没有对应的 tool_call）
        else:
            cleaned.append(msg)

    return {"messages": cleaned}


def extract_tool_call_examples(messages: List[dict]) -> List[Dict]:
    """从轨迹中提取工具调用示例，用于分析。

    返回每个工具调用的：
    {tool_name, query/url, result_summary}
    """
    examples = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if not is_external_tool(name):
                    continue

                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                # 找到对应的 tool_result
                tc_id = tc.get("id", "")
                result_content = ""
                for j in range(i + 1, len(messages)):
                    if messages[j].get("tool_call_id") == tc_id:
                        result_content = messages[j].get("content", "")[:200]
                        break

                examples.append({
                    "tool_name": name,
                    "arguments": args,
                    "result_preview": result_content,
                })

    return examples
