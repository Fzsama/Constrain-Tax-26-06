#!/usr/bin/env python3
"""Framework-level Two-Pass inference to bypass Constraint Tax.

Problem: When `response_format` (JSON Schema) is active, the model suppresses
tool calls (T2=0% across all 8 tested open-weight models on a fixed task).
This is a decoding-level policy bias that cannot be fixed by SFT, DPO, or
GRPO — it requires framework intervention.

Solution: Split the request into two passes:
  Pass 1: tools=ON,  response_format=OFF  → model freely calls tools
  Pass 2: tools=OFF, response_format=ON   → model formats tool results as JSON

Usage:
    from lib.two_pass import TwoPassInference
    tpi = TwoPassInference(base_url="http://localhost:8082", model="qw36-35b-a3b")
    result = tpi.chat_completion(messages, tools, response_format)
"""

import json, time
from typing import Optional, List, Dict, Any
import requests


class TwoPassInference:
    """Chat completion wrapper that splits tools+schema requests into two passes."""

    def __init__(self, base_url: str = "http://localhost:8082", model: str = "default",
                 max_tokens: int = 1024, temperature: float = 0.7, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def _call_sglang(self, messages, tools=None, response_format=None,
                     max_tokens=None, temperature=None) -> Dict[str, Any]:
        """Single call to SGLang chat completions API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def chat_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                        response_format: Optional[Dict] = None,
                        max_tokens: Optional[int] = None,
                        temperature: Optional[float] = None,
                        tool_executor: Optional[callable] = None,
                        ) -> Dict[str, Any]:
        """Main entry point. Automatically uses two-pass if tools AND schema are both present.

        Args:
            messages: Chat messages (system + user)
            tools: Tool definitions for function calling
            response_format: JSON Schema for structured output
            tool_executor: Optional async function(tool_calls) -> tool_results.
                           If None, returns pass-1 result for client-side tool execution.
        """
        if tools and response_format:
            return self._two_pass(messages, tools, response_format,
                                  max_tokens, temperature, tool_executor)
        else:
            return self._call_sglang(messages, tools, response_format,
                                     max_tokens, temperature)

    def _two_pass(self, messages, tools, response_format,
                  max_tokens, temperature, tool_executor) -> Dict[str, Any]:
        """Execute two-pass inference.

        Pass 1: tools=ON, response_format=OFF -> model selects and calls tools
        Pass 2: tools=OFF, response_format=ON -> model formats results as JSON
        """
        # ── Pass 1: Tool Selection (no schema constraint) ──
        pass1 = self._call_sglang(
            messages=messages,
            tools=tools,
            response_format=None,  # ← KEY: no schema to avoid Constraint Tax
            max_tokens=max_tokens or 512,
            temperature=temperature,
        )
        choice1 = pass1["choices"][0]
        msg1 = choice1["message"]
        tc = msg1.get("tool_calls")

        if not tc:
            # Model didn't call tools — fall back to single pass with schema
            # This handles the case where the model genuinely doesn't need tools
            pass1_fallback = self._call_sglang(
                messages=messages,
                tools=None,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            pass1_fallback["_two_pass"] = {"passes": 1, "tool_calls": 0, "fallback": True}
            return pass1_fallback

        # ── Execute tools ──
        tool_results = self._execute_tools(tc, tool_executor)

        # ── Pass 2: Structured JSON output (no tools to avoid conflict) ──
        # Build extended conversation: original + tool_call + tool_results
        extended = list(messages)
        extended.append({"role": "assistant", "content": msg1.get("content") or "",
                         "tool_calls": tc})
        for tr in tool_results:
            extended.append(tr)

        # Add schema instruction to guide formatting
        extended.append({
            "role": "user",
            "content": "Using the tool results above, output a JSON object that matches the required schema."
        })

        pass2 = self._call_sglang(
            messages=extended,
            tools=None,                # ← KEY: no tools in second pass
            response_format=response_format,
            max_tokens=max_tokens or self.max_tokens,
            temperature=0.3,           # Lower temp for formatting
        )
        pass2["_two_pass"] = {"passes": 2, "tool_calls": len(tc), "fallback": False}
        return pass2

    def _execute_tools(self, tool_calls: List[Dict],
                       tool_executor: Optional[callable] = None) -> List[Dict]:
        """Execute tool calls and return tool response messages.

        If tool_executor is provided, delegates to it.
        Otherwise returns placeholder responses for testing.
        """
        if tool_executor:
            return tool_executor(tool_calls)

        responses = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            # Placeholder responses for testing
            if name == "websearch":
                query = args.get("query", "")
                result = f"Search results for '{query}': [Company information would be retrieved from web search]"
            elif name == "knowledge_base":
                query = args.get("query", "")
                result = f"Knowledge base results for '{query}': [Compliance data would be retrieved from knowledge base]"
            else:
                result = f"Tool '{name}' executed with args: {json.dumps(args)}"

            responses.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": result,
            })
        return responses


# ── Convenience function ──
def create_two_pass_client(base_url="http://localhost:8082", model="default"):
    """Create a TwoPassInference client for quick use."""
    return TwoPassInference(base_url=base_url, model=model)
