"""全局配置 — LoRA Constraint Tax 项目。

所有路径、超参数、工具定义集中管理。
敏感信息（API Key 等）存放在 .env 文件中，不在此暴露。
"""

from __future__ import annotations

import os
from pathlib import Path

# 加载 .env 文件中的敏感配置
from dotenv import load_dotenv as _load_dotenv
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    _load_dotenv(_env_path)

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(os.getenv("LORA_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
DATA_DIR = PROJECT_ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"
SEED_TRACES_DIR = SEEDS_DIR / "seed_traces"
GENERATED_DIR = DATA_DIR / "generated"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# EvoAgent 项目路径
EVOAGENT_ROOT = Path(os.getenv("EVOAGENT_ROOT", "/root/0420-fz/ea-aim-fz-qwen-0602"))

# ============================================================
# 模型与 API
# ============================================================
TEACHER_MODEL = os.getenv("AEP_MODEL", "gpt-5.4-mini")
TEACHER_API_BASE = os.getenv("AEP_API_BASE", "")
TEACHER_API_KEY = os.getenv("AEP_API_KEY", "")

STUDENT_MODEL_PATH = os.getenv("STUDENT_MODEL_PATH", "")
STUDENT_MODEL_NAME = os.getenv("STUDENT_MODEL_NAME", "")

# ============================================================
# Agent 配置
# ============================================================
AGENT_TYPE = "inquiry-reply-agent"
AGENT_USER_ID = "lora_data_gen"
AGENT_TEAM_ID = "lora_data_gen"

# ============================================================
# 外部工具定义（与生产环境一致，精简版）
# ============================================================
EXTERNAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "搜索网络实时信息，用于调查买家公司背景、市场环境、行业动态",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_base",
            "description": "查询外贸知识库，获取行业标准、贸易政策、认证要求等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "知识库查询关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetchurl",
            "description": "抓取网页内容，用于获取买家官网信息（公司规模、主营业务等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要抓取的网页 URL"},
                    "analyze": {
                        "type": "boolean",
                        "description": "是否触发竞品深度分析（默认 false）",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        },
    },
]

# ============================================================
# AgentResponse Schema 精简描述（训练数据 system prompt 使用）
# ============================================================
SCHEMA_DESCRIPTION = """## 输出格式（最高优先级，覆盖所有其他格式指令）

你的最终回复必须是合法的 JSON 对象：
{"blocks": [<block>, <block>, ...]}

每个 block 是以下两种之一：
- text block：{"type":"text","content":"<markdown 文本>"}
- card block：{"type":"card","card":{"type":"<卡片类型>", ...卡片字段}}

可用的卡片类型：expert_insight / detail_card / ai_judgement / highlight_block / collapsible / inquiry_collapsible / suggested_questions / radio_select / checkbox_select / upload_request / product_images / report_preview

硬约束：
- 禁止在输出里使用 ```json / ``` markdown 围栏
- 禁止在 blocks 数组外面再包一层文本叙述
- 必须按顺序输出卡片"""


def build_training_system_prompt() -> str:
    """构建训练数据用的精简 system prompt。

    只包含 Constraint Tax 纠正所需的核心信息：
    1. Agent 角色
    2. 可用工具定义
    3. 输出格式约束（Schema）

    长度控制在 ~2000 chars，避免训练数据超过 max_seq_len。
    """
    tools_desc = """## 可用工具

1. **websearch** — 搜索网络实时信息
   用法: websearch(query="搜索关键词")
   用途: 调查买家公司背景、市场环境、行业动态

2. **knowledge_base** — 查询外贸知识库
   用法: knowledge_base(query="查询关键词")
   用途: 获取行业标准、贸易政策、认证要求、关税信息

3. **fetchurl** — 抓取网页内容
   用法: fetchurl(url="网页URL", analyze=False)
   用途: 获取买家官网信息（公司规模、主营业务、联系方式）"""

    role_desc = """你是贸探·跟单专家，一名15年经验的外贸成交型跟单老手。
你的任务是分析客户询盘，先搜索和收集买家公司背景、市场环境、行业政策等外部信息，
然后基于真实数据输出结构化的询盘分析报告。"""

    return f"""{role_desc}

{tools_desc}

{SCHEMA_DESCRIPTION}

## 核心原则
- **先搜索、再分析**：必须先使用 websearch / knowledge_base / fetchurl 获取外部信息，再基于真实数据输出 JSON
- **禁止编造**：不能在没有搜索的情况下直接编造买家信息或分析结论
- **信息不足标注**：搜索无结果时标注"未查到"或"未提取"，不要伪造数据"""


# ============================================================
# 训练超参数
# ============================================================
TRAINING_CONFIG = {
    "training_type": "qlora",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 2e-4,
    "batch_size": 4,
    "grad_accum": 8,
    "epochs": 3,
    "max_seq_len": 8192,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.05,
    "optimizer": "adamw_8bit",
    "seed": 42,
}

# ============================================================
# 数据生成参数
# ============================================================
SEED_COUNT = 30
SCALE_FACTOR = 7  # 每个种子生成的变体数（30 × 7 ≈ 210 新场景）
TARGET_TRAIN_SIZE = 500  # 目标训练集大小
EVAL_SPLIT = 0.1

# 场景变体维度
INDUSTRIES = [
    "LED 照明", "消费电子", "机械设备", "纺织服装", "化工材料",
    "家居用品", "汽车配件", "医疗器械", "太阳能", "五金工具",
]
REGIONS = [
    ("北美", ["美国", "加拿大"]),
    ("西欧", ["德国", "法国", "英国", "意大利", "荷兰"]),
    ("中东", ["阿联酋", "沙特阿拉伯", "卡塔尔"]),
    ("东南亚", ["印度", "越南", "泰国", "印度尼西亚"]),
    ("南美", ["巴西", "墨西哥", "阿根廷"]),
    ("非洲", ["尼日利亚", "南非", "埃及"]),
    ("大洋洲", ["澳大利亚", "新西兰"]),
]
BUYER_TYPES = ["品牌商", "分销商", "零售商/终端用户"]
COMPLEXITY_LEVELS = ["简单RFQ", "标准询盘", "复杂询盘"]
