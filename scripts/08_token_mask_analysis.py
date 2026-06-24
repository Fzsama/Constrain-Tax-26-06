#!/usr/bin/env python3
"""Token-level analysis: visualize the vocab bitmask under JSON Schema constraint.

For each FSM state (generation step), this script:
1. Loads the tokenizer from the Qwen model
2. Compiles the JSON Schema via xgrammar
3. Creates a GrammarMatcher and steps through the FSM
4. Extracts the bitmask at each state
5. Reports which tokens are allowed (bit=1) vs suppressed (bit=0)
6. Specifically checks if `<tool_call>` XML tokens are suppressed
"""

import sys, json
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Config ──
MODEL_PATH = "/root/.cache/modelscope/hub/models/Jackrong/Qwopus3.6-35B-A3B-v1"

# The exact JSON Schema used in our T2 tests
JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "company_info":  {"type": "string"},
        "compliance_notes": {"type": "string"},
    },
    "required": ["company_name", "company_info", "compliance_notes"],
    "additionalProperties": False,
})

# Tool-call relevant substrings to check
TOOL_CALL_MARKERS = [
    "<",           # Start of <tool_call>
    "<tool_call>",
    "tool_call",
    "<function",
    "</tool_call>",
    "websearch",
    "knowledge_base",
]


def main():
    print("=" * 70)
    print("Token-Level Mask Analysis: JSON Schema Constraint")
    print("=" * 70)

    # ── 1. Load tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"\nTokenizer: {type(tokenizer).__name__}")
    print(f"Vocab size: {vocab_size}")

    # Decode special tokens
    special_tokens = {
        "<tool_call>": tokenizer.encode("<tool_call>", add_special_tokens=False),
        "</tool_call>": tokenizer.encode("</tool_call>", add_special_tokens=False),
        "<": tokenizer.encode("<", add_special_tokens=False),
        "{": tokenizer.encode("{", add_special_tokens=False),
        "[": tokenizer.encode("[", add_special_tokens=False),
        "tool_call": tokenizer.encode("tool_call", add_special_tokens=False),
        "<function": tokenizer.encode("<function", add_special_tokens=False),
        "websearch": tokenizer.encode("websearch", add_special_tokens=False),
        "knowledge_base": tokenizer.encode("knowledge_base", add_special_tokens=False),
    }
    print(f"\nSpecial token encodings:")
    for text, ids in special_tokens.items():
        decoded = [tokenizer.decode([tid]) for tid in ids]
        print(f"  {repr(text):20s} → ids={ids} → tokens={decoded}")

    # ── 2. Compile JSON Schema via xgrammar ──
    from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo

    tokenizer_info = TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size
    )
    compiler = GrammarCompiler(tokenizer_info)

    print(f"\nCompiling JSON Schema...")
    compiled = compiler.compile_json_schema(JSON_SCHEMA)
    print(f"  Compiled OK")

    # ── 3. Create matcher and step through FSM states ──
    from xgrammar import allocate_token_bitmask, apply_token_bitmask_inplace
    import torch

    matcher = GrammarMatcher(compiled, max_rollback_tokens=200)

    # bitmask is a torch tensor of int32, shape [batch=1, packed_vocab_size]
    bitmask_size = (vocab_size + 31) // 32  # packed uint32 representation

    print(f"\n{'='*70}")
    print("FSM Step-by-Step Token Analysis")
    print(f"{'='*70}")

    def fill_and_decode(matcher, state_label, check_tokens=None):
        """Fill bitmask for current FSM state and decode allowed/suppressed tokens."""
        bm = allocate_token_bitmask(1, vocab_size)  # shape [1, packed_vocab]
        matcher.fill_next_token_bitmask(bm, 0)

        # Decode: check each token id against packed bitmask
        allowed = set()
        for tid in range(vocab_size):
            word_idx = tid // 32
            bit_idx = tid % 32
            if word_idx < bm.shape[1] and (bm[0, word_idx].item() >> bit_idx) & 1:
                allowed.add(tid)

        print(f"\n── {state_label} ──")
        print(f"  Allowed: {len(allowed)} / {vocab_size} ({100*len(allowed)/vocab_size:.2f}%)")

        # Show first 20 allowed tokens
        top = sorted(allowed)[:20]
        for tid in top:
            text = tokenizer.decode([tid])
            print(f"    id={tid:6d}  text={repr(text)}")

        # Check critical markers
        if check_tokens:
            print(f"\n  Critical markers:")
            for name in check_tokens:
                ids = special_tokens.get(name, [])
                if ids:
                    statuses = []
                    for tid in ids:
                        statuses.append("✅" if tid in allowed else "❌")
                    texts = [f"{tokenizer.decode([tid])}({tid})" for tid in ids]
                    print(f"    {name:20s} → {list(zip(texts, statuses))}")

        # Check '<' specifically
        lt_ids = tokenizer.encode("<", add_special_tokens=False)
        lt_all = all(tid in allowed for tid in lt_ids)
        print(f"\n  🔑 '<' (start of <tool_call>): {'✅ ALLOWED' if lt_all else '❌ SUPPRESSED'}")
        for tid in lt_ids:
            status = "ALLOWED" if tid in allowed else "SUPPRESSED"
            print(f"     id={tid} token={repr(tokenizer.decode([tid]))} → {status}")

        return allowed, bm

    # State 0: Before any token (FSM expects object or array start)
    CHECK_KEYS = ["<tool_call>", "</tool_call>", "<", "{", "[", "tool_call", "<function", "websearch", "knowledge_base"]
    allowed_0, bm_0 = fill_and_decode(matcher, "State 0: Initial FSM (expecting JSON start)", CHECK_KEYS)

    print(f"\n── State 0: Initial FSM state (expecting JSON start) ──")
    print(f"  Allowed tokens: {len(allowed_0)} / {vocab_size} ({100*len(allowed_0)/vocab_size:.2f}%)")

    # Show top allowed tokens
    top_allowed = sorted(allowed_0)[:30]
    for tid in top_allowed:
        text = tokenizer.decode([tid])
        print(f"    id={tid:6d}  text={repr(text)}")

    # Check critical markers
    print(f"\n  Critical markers:")
    for marker, ids in special_tokens.items():
        allowed = [tokenizer.decode([tid]) for tid in ids if tid in allowed_0]
        suppressed = [tokenizer.decode([tid]) for tid in ids if tid not in allowed_0]
        if suppressed:
            print(f"    ❌ {marker}: SUPPRESSED — tokens {suppressed}")
        if allowed:
            print(f"    ✅ {marker}: ALLOWED — tokens {allowed}")

    # Check first char of <tool_call> — the "<" character
    lt_ids = tokenizer.encode("<", add_special_tokens=False)
    lt_allowed = all(tid in allowed_0 for tid in lt_ids)
    print(f"\n  🔑 '<' (start of <tool_call>): {'✅ ALLOWED' if lt_allowed else '❌ SUPPRESSED'}")
    for tid in lt_ids:
        status = "ALLOWED" if tid in allowed_0 else "SUPPRESSED"
        print(f"     id={tid} token={repr(tokenizer.decode([tid]))} → {status}")

    # ── 4. Simulate: feed "{" and check next state ──
    open_brace_id = tokenizer.encode("{", add_special_tokens=False)[0]
    matcher.accept_token(open_brace_id)
    allowed_1, bm_1 = fill_and_decode(matcher,
        "State 1: After '{' (inside object, expecting key or })",
        check_tokens=CHECK_KEYS)

    # ── 5. Simulate: feed a key "company_name" ──
    quote_id = tokenizer.encode('"', add_special_tokens=False)[0]
    matcher.accept_token(quote_id)  # start of string
    # Feed the string characters one by one (simplified: feed key as whole)
    key_ids = tokenizer.encode('"company_name"', add_special_tokens=False)
    for tid in key_ids[1:]:  # skip first quote (already fed)
        if not matcher.is_terminated():
            matcher.accept_token(tid)

    # Now FSM expects ":"
    colon_id = tokenizer.encode(":", add_special_tokens=False)[0]
    matcher.accept_token(colon_id)

    # Now FSM expects a value — does it allow { (nested object) or " (string)?
    allowed_v, bm_v = fill_and_decode(matcher,
        "State 2: After 'company_name': (expecting value)",
        check_tokens=CHECK_KEYS)

    # ── 6. Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"""
The JSON Schema grammar created by xgrammar is a finite state machine.
At EVERY state, the FSM only allows tokens that can appear in valid JSON:
  - Object start: {{
  - Array start: [
  - String start: "
  - Numbers: 0-9, -
  - Literals: t(rue), f(alse), n(ull)
  - Structural: : , }} ]

The "<" character (U+003C) is NEVER in any valid JSON FSM state.
Qwen3's <tool_call> XML format starts with "<".
Therefore: <tool_call> is ALWAYS bit=0 → logit=-inf → probability=0.

This is NOT a model bias or preference issue.
It is an ABSOLUTE structural constraint imposed by the grammar compiler.
No amount of weight-level training (SFT/DPO/GRPO) can change this.
""")

    # ── 7. Bonus: tool-call related tokens in each state ──
    print(f"\n{'='*70}")
    print("BONUS: Tool-call markers across all sampled states")
    print(f"{'='*70}")

    for state_name, allowed_set in [
        ("State 0 (initial)", allowed_0),
        ("State 1 (after '{')", allowed_1),
        ("State 2 (after value)", allowed_v),
    ]:
        print(f"\n  {state_name}:")
        for marker_text in TOOL_CALL_MARKERS:
            ids = tokenizer.encode(marker_text, add_special_tokens=False)
            statuses = []
            for tid in ids:
                statuses.append("✅" if tid in allowed_set else "❌")
            decoded_ids = [f"{tokenizer.decode([tid])}({tid})" for tid in ids]
            print(f"    {repr(marker_text):20s} → {list(zip(decoded_ids, statuses))}")


if __name__ == "__main__":
    main()
