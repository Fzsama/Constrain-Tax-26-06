#!/usr/bin/env python3
"""Phase 1: 种子训练数据采集。

用 EvoAgent inquiry-reply-agent + GPT-5.4-mini 在 30 个多样询盘场景上
运行，收集完整的「先调工具 → 后输出 JSON」对话轨迹。

工作流程:
1. 加载种子询盘场景（data/seeds/inquiry_templates.py）
2. 对每个场景:
   a. 创建 SubAgentRunner (inquiry-reply-agent + GPT-5.4-mini)
   b. 运行 run_stream()，收集 SSE 事件
   c. 从 workspace 读取保存的 conversation history
   d. 过滤内部工具调用，转为训练数据格式
   e. 质量校验
   f. 保存轨迹到 data/seeds/seed_traces/{seed_id}.json
3. 输出统计报告

用法:
  cd /root/0420-fz/lora-qwen-0612
  python scripts/01_collect_seed_traces.py              # 全部 30 个
  python scripts/01_collect_seed_traces.py --start 0 --count 5  # 前 5 个
  python scripts/01_collect_seed_traces.py --ids seed_001,seed_002  # 指定 ID
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================
# 0. 环境设置 — 添加项目根目录 + EvoAgent 到 Python path
# ============================================================
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent  # /root/0420-fz/lora-qwen-0612
_EVOAGENT_ROOT = Path(os.getenv("EVOAGENT_ROOT", "/root/0420-fz/ea-aim-fz-qwen-0602"))

for _p in (str(_PROJECT_ROOT), str(_EVOAGENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 加载 EvoAgent .env 配置（FFS 等依赖）
try:
    from dotenv import load_dotenv
    _env_file = _EVOAGENT_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

# ============================================================
# 导入
# ============================================================
from config import (
    SEEDS_DIR, SEED_TRACES_DIR, REPORTS_DIR,
    TEACHER_MODEL, TEACHER_API_BASE, TEACHER_API_KEY,
    AGENT_TYPE, AGENT_USER_ID, AGENT_TEAM_ID,
    build_training_system_prompt,
)
from data.seeds.inquiry_templates import SEED_SCENARIOS
from scripts.lib.trace_utils import (
    convert_to_training_format,
    validate_tool_json_sequence,
    filter_tool_messages,
)
from scripts.lib.quality_check import validate_training_sample, validate_batch, generate_report

# ============================================================
# 全局状态
# ============================================================
SEED_TRACES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_stats = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "skipped": 0,
    "errors": [],
    "tool_call_counts": [],
    "elapsed_times": [],
}


# ============================================================
# 核心采集逻辑
# ============================================================
async def collect_single_seed(
    scenario: dict,
    timeout: int = 300,
    verbose: bool = True,
) -> Optional[Dict]:
    """对单个种子场景采集训练轨迹。

    Args:
        scenario: 种子场景 dict（含 id, industry, region, user_message 等）
        timeout: 单个场景超时（秒）
        verbose: 是否打印详细日志

    Returns:
        训练数据 {"messages": [...]} 或 None（采集失败）
    """
    seed_id = scenario["id"]
    t_start = time.monotonic()

    if verbose:
        print(f"\n{'─'*60}")
        print(f"🌱 {seed_id} | {scenario['industry']} | {scenario['region']} | {scenario['buyer_type']}")
        print(f"   复杂度: {scenario['complexity']}")

    # --- 切换到 EvoAgent 根目录（sub_agent.py 使用相对路径 Path("configs/agents")） ---
    _orig_cwd = os.getcwd()
    os.chdir(str(_EVOAGENT_ROOT))
    try:
        # --- 导入 EvoAgent 内部模块 ---
        from core.sub_agent import SubAgentRunner
        from utils.workspace import Workspace

        # --- 创建 SubAgentRunner ---
        session_id = f"lora_seed_{seed_id}_{int(time.time())}"
        runner = SubAgentRunner(
            agent_type=AGENT_TYPE,
            user_id=AGENT_USER_ID,
            session_id=session_id,
            team_id=AGENT_TEAM_ID,
            role="owner",
        )

        # 覆盖模型为 GPT-5.4-mini（Teacher）
        runner.config.model = TEACHER_MODEL
        runner.config.api_base = TEACHER_API_BASE
        runner.config.api_key = TEACHER_API_KEY

        if verbose:
            print(f"   Model: {TEACHER_MODEL}")
            print(f"   Session: {session_id}")

        # --- 捕获 system prompt（pure_history 不含 system 消息） ---
        system_prompt = runner._build_system_prompt()

        # --- 运行 Agent ---
        events = []
        async with asyncio.timeout(timeout):
            async for event in runner.run_stream(scenario["user_message"]):
                events.append(event)

        # --- 从 workspace 读取保存的 conversation ---
        workspace = Workspace(AGENT_USER_ID)
        history_data = workspace.history_load(runner.session_id)

        if not history_data or not isinstance(history_data, dict):
            print(f"   ⚠️  无法从 workspace 读取 session {runner.session_id}")
            _stats["failed"] += 1
            _stats["errors"].append(f"{seed_id}: workspace read failed")
            return None

        conversation = history_data.get("conversation", [])
        if not conversation:
            print(f"   ⚠️  conversation 为空")
            _stats["failed"] += 1
            _stats["errors"].append(f"{seed_id}: empty conversation")
            return None

        if verbose:
            print(f"   原始消息数: {len(conversation)}")

        # --- 转为训练格式（使用精简训练 system prompt，而非完整生产 prompt） ---
        training_sample = convert_to_training_format(conversation, system_prompt_override=build_training_system_prompt())
        msgs = training_sample["messages"]

        # --- 质量校验 ---
        seq_result = validate_tool_json_sequence(msgs)
        passed, issues = validate_training_sample(training_sample)

        # --- 记录统计 ---
        elapsed = time.monotonic() - t_start
        _stats["total"] += 1
        _stats["elapsed_times"].append(elapsed)
        _stats["tool_call_counts"].append(seq_result["tool_call_count"])

        if passed:
            _stats["success"] += 1
            if verbose:
                print(f"   ✅ 通过 | 工具: {seq_result['tool_call_count']}次 ({', '.join(seq_result['tool_names'])}) | "
                      f"耗时: {elapsed:.0f}s | 训练消息数: {len(msgs)}")
        else:
            _stats["failed"] += 1
            _stats["errors"].append(f"{seed_id}: {', '.join(issues)}")
            if verbose:
                print(f"   ⚠️  校验未通过:")
                for issue in issues:
                    print(f"      - {issue}")

        # --- 保存完整数据（含元数据） ---
        output = {
            "meta": {
                "seed_id": seed_id,
                "industry": scenario["industry"],
                "region": scenario["region"],
                "country": scenario.get("country", ""),
                "buyer_type": scenario["buyer_type"],
                "complexity": scenario["complexity"],
                "model": TEACHER_MODEL,
                "session_id": runner.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed, 1),
                "tool_call_count": seq_result["tool_call_count"],
                "tool_names": seq_result["tool_names"],
                "passed_validation": passed,
                "issues": issues,
            },
            "messages": msgs,
        }

        out_path = SEED_TRACES_DIR / f"{seed_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"   📁 已保存: {out_path}")

        return training_sample

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t_start
        print(f"   ❌ 超时 ({elapsed:.0f}s > {timeout}s)")
        _stats["total"] += 1
        _stats["failed"] += 1
        _stats["errors"].append(f"{seed_id}: timeout ({timeout}s)")
        return None

    except Exception as e:
        elapsed = time.monotonic() - t_start
        print(f"   ❌ 异常: {type(e).__name__}: {e}")
        _stats["total"] += 1
        _stats["failed"] += 1
        _stats["errors"].append(f"{seed_id}: {type(e).__name__}: {str(e)[:200]}")
        return None

    finally:
        os.chdir(_orig_cwd)


# ============================================================
# 批量采集
# ============================================================
async def collect_all(
    scenarios: List[dict],
    timeout: int = 300,
    delay: float = 2.0,
    verbose: bool = True,
) -> List[Dict]:
    """批量采集种子轨迹。

    Args:
        scenarios: 种子场景列表
        timeout: 单个场景超时
        delay: 场景间延迟（秒，避免 API 限流）
        verbose: 详细日志

    Returns:
        成功采集的训练样本列表
    """
    samples = []

    for i, scenario in enumerate(scenarios):
        if i > 0 and delay > 0:
            if verbose:
                print(f"\n   ⏳ 等待 {delay}s...")
            await asyncio.sleep(delay)

        sample = await collect_single_seed(scenario, timeout=timeout, verbose=verbose)
        if sample:
            samples.append(sample)

    return samples


# ============================================================
# 生成报告
# ============================================================
def generate_collection_report(samples: List[Dict]) -> str:
    """生成采集报告。"""
    lines = [
        "=" * 60,
        "种子数据采集报告",
        "=" * 60,
        f"采集时间: {datetime.now(timezone.utc).isoformat()}",
        f"Teacher Model: {TEACHER_MODEL}",
        "",
        f"总场景数:   {_stats['total']}",
        f"成功:       {_stats['success']}",
        f"失败:       {_stats['failed']}",
        f"成功率:     {_stats['success']/_stats['total']*100:.1f}%" if _stats['total'] > 0 else "N/A",
        "",
    ]

    if _stats["tool_call_counts"]:
        avg_tools = sum(_stats["tool_call_counts"]) / len(_stats["tool_call_counts"])
        lines.append(f"平均工具调用次数: {avg_tools:.1f}")
        lines.append(f"工具调用范围: {min(_stats['tool_call_counts'])}-{max(_stats['tool_call_counts'])}")

    if _stats["elapsed_times"]:
        avg_time = sum(_stats["elapsed_times"]) / len(_stats["elapsed_times"])
        lines.append(f"平均耗时: {avg_time:.0f}s")
        lines.append(f"总耗时: {sum(_stats['elapsed_times']):.0f}s")

    lines.append("")

    # 按行业统计
    lines.append("行业分布:")
    from collections import Counter
    industries = Counter(s["meta"]["industry"] for s in [
        json.loads(open(SEED_TRACES_DIR / f, encoding="utf-8").read())
        for f in os.listdir(SEED_TRACES_DIR) if f.endswith(".json")
    ] if os.path.exists(SEED_TRACES_DIR))
    for ind, count in industries.most_common():
        lines.append(f"  {ind}: {count}")

    lines.append("")

    if _stats["errors"]:
        lines.append("错误详情:")
        for err in _stats["errors"]:
            lines.append(f"  - {err}")

    # 质量校验汇总
    if samples:
        lines.append("")
        lines.append("-" * 60)
        batch_result = validate_batch(samples)
        lines.append(generate_report(batch_result))

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="种子训练数据采集 — GPT-5.4-mini Teacher 轨迹生成"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="起始场景索引（默认 0）"
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="采集场景数量（默认全部）"
    )
    parser.add_argument(
        "--ids", type=str, default=None,
        help="指定场景 ID，逗号分隔（如 seed_001,seed_005）"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="单个场景超时秒数（默认 300）"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="场景间延迟秒数（默认 2.0）"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="跳过已存在的轨迹文件"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # 选择场景
    if args.ids:
        id_set = set(args.ids.split(","))
        scenarios = [s for s in SEED_SCENARIOS if s["id"] in id_set]
        print(f"按 ID 筛选: {len(scenarios)} 个场景")
    else:
        start = args.start
        end = start + args.count if args.count else len(SEED_SCENARIOS)
        scenarios = SEED_SCENARIOS[start:end]
        print(f"场景范围: [{start}:{end}] = {len(scenarios)} 个")

    # 跳过已存在的
    if args.skip_existing:
        before = len(scenarios)
        scenarios = [
            s for s in scenarios
            if not (SEED_TRACES_DIR / f"{s['id']}.json").exists()
        ]
        skipped = before - len(scenarios)
        if skipped:
            print(f"跳过已存在: {skipped} 个")
        _stats["skipped"] = skipped

    if not scenarios:
        print("无待采集场景")
        return

    print(f"Teacher Model: {TEACHER_MODEL}")
    print(f"Agent Type: {AGENT_TYPE}")
    print(f"超时: {args.timeout}s | 延迟: {args.delay}s")
    print()

    # 采集
    t_total = time.monotonic()
    samples = await collect_all(scenarios, timeout=args.timeout, delay=args.delay)

    # 报告
    report = generate_collection_report(samples)
    print(f"\n{report}")

    # 保存报告
    report_path = REPORTS_DIR / f"seed_collection_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📊 报告已保存: {report_path}")

    elapsed_total = time.monotonic() - t_total
    print(f"\n⏱️  总耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")


if __name__ == "__main__":
    asyncio.run(main())
