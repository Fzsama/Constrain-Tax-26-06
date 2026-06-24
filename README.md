# Constraint Tax — Tool Calling Suppression Under Structured Output Constraints

> A systematic empirical investigation of how `response_format` (JSON Schema) suppresses tool calling in open-weight LLMs,
> with root cause traced to the xgrammar decoding layer and a model-level workaround (A2) demonstrating 95% emission rate.

[中文版](README_CN.md)

## Key Findings

**Constraint Tax**: When `tools` + `response_format` (JSON Schema) are both active, all 8 tested open-weight models consistently skip tool calls (T2=0%).

- **Root Cause**: xgrammar compiles JSON Schema into an FSM; `<tool_call>` token has bit=0 across all FSM states → logit=-inf
- **Not a Training Problem**: SFT/GRPO modifies weight-level logit preferences but cannot overcome decoding-level hard mask
- **Cross-Framework Confirmed**: SGLang 0.5.9 and vLLM 0.22.0 share the same xgrammar library; behavior is identical
- **Sole Exception**: GPT-5.4-mini (closed-source; implementation unknown)
- **Limitation**: 9-model matrix uses fixed single task × 5 rounds/condition; Qwen3.6-35B-A3B has additional 200+ cross-task validations

See [doc/11-final-report-constraint-tax.md](doc/11-final-report-constraint-tax.md) for the full investigation.

## Experiment Matrix (7 Rounds)

| # | Method | Data | T1 | T2 Emission | T2 Valid JSON | Conclusion |
|:---:|------|:---:|:---:|:---:|:---:|------|
| 1 | SFT Business Data | 870 | 2 calls | 0% | 0% | Learns hallucination |
| 2 | SFT Tool Mandatory | 200 | 109 calls | 0% | 0% | Tool dependency learned |
| 3-4 | SFT Schema Injection | 200 | — | 0% | 0% | Mismatch not root cause |
| 5 | **GRPO** (RL) | 200 | 100% | 0% | 0% | RL cannot flip the mask |
| 6 | SFT Scaled Data | 6000 | 60% | 0% | 0% | Data volume not bottleneck |
| 7 | **A2 JSON-Embedded** | **8000** | **95%** | **95%** | **20%** | ✅ **First breakthrough (emission rate)** |

> **Metric Definitions**: T2 Emission Rate = fraction of raw content containing `_tool_calls` pattern (model's *intent* to invoke tools). T2 Valid JSON Rate = fraction parseable by `json.loads()` with non-empty `_tool_calls` (framework-executable). A2 achieves 95% emission rate, proving the FSM barrier is overcome; valid JSON rate at 20% requires format compliance improvements ([details](doc/16-a2-breakthrough-results.md) §2.2).

See [doc/16-a2-breakthrough-results.md](doc/16-a2-breakthrough-results.md) for the A2 breakthrough report.

## The A2 Solution

Instead of fighting the FSM, the A2 scheme works *with* it by embedding tool calls inside JSON:

```
Legacy path (blocked by FSM):
  model → first token "<" → FSM bit=0 → -inf → FAIL

A2 path (FSM allows):
  model → first token "{" → FSM bit=1 → OK
       → key "_tool_calls" → FSM allows (valid JSON key)
       → value [{name, arguments}] → FSM allows (valid JSON array/object)
       → business fields → OK
       → "}" → complete
```

First token `{` passes FSM; `_tool_calls` is a legal JSON key → framework extracts tool calls post-hoc.

## Project Structure

```
lora-qwen-0612/
├── config.py                          # Global config (paths, API, tools, hyperparams)
├── README.md / README_CN.md
├── CLAUDE.md
├── .env.example                       # Environment variable template
├── .gitignore
│
├── doc/                               # Documentation (17 docs)
│   ├── 00..07_*.md                    # Early design docs & GPT analyses
│   ├── 08-stage2-analysis-next-steps.md       # SFT 4-round analysis
│   ├── 09-grpo-technical-doc.md               # GRPO technical doc
│   ├── 10-grpo-results.md                     # GRPO experimental report
│   ├── 11-final-report-constraint-tax.md      # ★ Final comprehensive report
│   ├── 12-paper-experiment-supplement-v2.md   # Paper experiment supplement
│   ├── 13-a2-progress.md                      # A2 solution progress
│   ├── 14-appendix-test-design-and-tool-schema.md  # Paper appendix
│   ├── 15-handover-lifangzheng.md             # Handover doc
│   └── 16-a2-breakthrough-results.md          # ★ A2 breakthrough report
│
├── scripts/                           # Scripts (12+)
│   ├── 01_collect_seed_traces.py             # Phase 1: seed trace collection
│   ├── 01b_generate_synthetic_seeds.py       # Synthetic seed generation
│   ├── 01c_enrich_seeds.py                   # Seed enrichment
│   ├── 02_generate_training_data.py          # Phase 2: scaled generation
│   ├── 03_train_sft.py                       # Phase 3: SFT training
│   ├── 04_generate_tool_mandatory_data.py     # Tool Mandatory data gen
│   ├── 05_train_dpo.py                        # DPO training (incomplete)
│   ├── 06_train_grpo.py                       # ★ GRPO training (pure PyTorch + SGLang)
│   ├── 07_test_grpo_direct.py                 # GRPO direct model test
│   ├── 07_test_grpo_vllm.py                   # GRPO vLLM test
│   ├── 08_token_mask_analysis.py              # ★ Token-level FSM bitmask analysis
│   ├── 09_generate_a2_seed_data.py            # A2 seed data generation
│   ├── 10_test_a2_model.py                   # ★ A2 model T1/T2/T3 test
│   ├── 11_generate_a2_v2_seed_data.py         # A2 v2 seed (50:50 ratio)
│   └── lib/                                   # Script library
│       ├── trace_utils.py                     #   Trace extraction/format conversion
│       └── quality_check.py                   #   6-dim auto quality validation
│
├── lib/
│   └── two_pass.py                     # Two-Pass inference framework
│
├── tests/
│   └── test_constraint_tax_lora.py     # T1/T2/T3 test script
│
├── evidence/
│   └── cross_model_testing/            # Cross-model/cross-framework evidence
│
├── artifacts/                          # Result summaries (committed)
│   ├── experiment_results_summary.json
│   ├── model_fingerprint.json
│   └── grpo_training_summary.json
│
├── data/                               # Data directory
│   ├── seeds/inquiry_templates.py     #   30 seed scenario templates
│   ├── generated/                     #   Phase 2 scaled data (gitignored)
│   └── processed/                     #   Training/eval datasets (gitignored)
│
└── outputs/                           # Training outputs (gitignored)
```

## Environment

- **Training**: conda `lora_qwen_0612` (torch 2.10, transformers 5.5, PEFT)
- **Inference**: SGLang 0.5.9 (`sglang_059`), vLLM 0.22.0 (`vllm_0605`)
- **GPU**: 2× NVIDIA A800-SXM4-80GB (or equivalent 80GB+ VRAM)
- **Python**: 3.12+ (see [requirements.txt](requirements.txt))

### Launching Inference Servers

```bash
# SGLang (recommended — primary inference framework)
conda activate sglang_059
python -m sglang.launch_server \
    --model-path /path/to/model \
    --served-model-name qwen-a2 \
    --port 8082 --host 0.0.0.0 \
    --tp-size 2 \
    --mem-fraction-static 0.85 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code

# vLLM (cross-framework comparison)
conda activate vllm_0605
export LD_LIBRARY_PATH="/path/to/env/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

## Key Scripts

```bash
# Token-level FSM analysis
python scripts/08_token_mask_analysis.py

# A2 model testing (CLI args override API endpoint and model name)
python scripts/10_test_a2_model.py --api http://localhost:8082/v1/chat/completions --model qwen-a2
# Or via environment variables:
CT_API_URL=http://localhost:8082/v1/chat/completions CT_MODEL_NAME=qwen-a2 python scripts/10_test_a2_model.py

# GRPO training
python scripts/06_train_grpo.py --n_gen 4 --epochs 3 --batch 2 --grad_accum 8 --merge

# A2 seed generation
python scripts/09_generate_a2_seed_data.py

# Two-Pass inference
python -c "from lib.two_pass import TwoPassInference; ..."
```

## HF Datasets

| Dataset | Size | Format | Used In |
|------|:---:|------|------|
| [FZSAMA/qwen-constraint-tax-training-data](https://huggingface.co/datasets/FZSAMA/qwen-constraint-tax-training-data) | 6000 | Legacy tool_call → JSON | SFT Round 6 |
| [FZSAMA/qwen-a2-constraint-tax-data](https://huggingface.co/datasets/FZSAMA/qwen-a2-constraint-tax-data) | 8000 | A2 JSON-embedded `_tool_calls` | A2 Round 7 |

> Note: Raw synthetic data also exists locally at `data/processed/synthetic_6000/` and `data/processed/a2_synthetic_8000/` (gitignored).

## Artifacts

| Artifact | Location | Description |
|------|------|------|
| Experiment Summary | [`artifacts/experiment_results_summary.json`](artifacts/experiment_results_summary.json) | 7-round results + cross-model data |
| Model Fingerprint | [`artifacts/model_fingerprint.json`](artifacts/model_fingerprint.json) | A2 model config + adapter info |
| GRPO Training Summary | [`artifacts/grpo_training_summary.json`](artifacts/grpo_training_summary.json) | 3-epoch training metrics |
| GRPO Full Log | `logs/GRPO.log` (gitignored, 446KB) | Raw GRPO training log |
| A2 Model Weights | `outputs/loraed_Qwopus3.6-35B-A3B-a2-0623/` (gitignored) | 26 safetensors + config |
| A2 Seed Data | `data/processed/a2_seed_data.json` (gitignored) | 30 v1 seeds |
| A2 v2 Seed Data | `data/processed/a2_v2_seed_data.json` (gitignored) | 40 v2 seeds (50:50) |
| Cross-Model Evidence | [`evidence/cross_model_testing/`](evidence/cross_model_testing/) | 9-model matrix + raw logs |
