"""System prompt and response parsing for the LLM ReAct agent."""

from __future__ import annotations

import json
import re
from typing import Any


SYSTEM_PROMPT = """\
You are an AIOps agent responsible for monitoring and maintaining a \
microservice application. Your job is to detect anomalies, identify root \
causes, and take remediation actions.

You have access to ML-powered tools that provide anomaly detection, \
explainability, and root cause analysis. Use these tools to investigate \
incidents step by step.

GUIDELINES:
- Always check metrics before taking action
- Use explain_anomaly to understand WHY something was flagged
- Use localize_root_cause to find the origin of cascading issues
- For security events (brute force, DDoS), act quickly
- If confidence is below 0.7, use alert_human instead of auto-remediating
- If a remediation action fails, try an alternative action or escalate with alert_human
- Do not retry the same failed action type more than once
- After taking a remediation action, you are done — respond with ACTION: done

Available services: {service_list}

Service topology:
{topology_description}

Available tools:
{tool_descriptions}

Respond in EXACTLY this format on every turn:

THOUGHT: [your reasoning about what to do next]
ACTION: [tool_name]
ARGS: {{"param": "value"}}

When you have completed the investigation and remediation, respond:

THOUGHT: [final summary of what happened and what you did]
ACTION: done
ARGS: {{}}
"""


def format_system_prompt(
    service_list: list[str],
    topology_edges: list[tuple[str, str]],
    tool_descriptions: str,
) -> str:
    """Fill in the system prompt template."""
    topo_lines = [f"  {u} -> {v}" for u, v in topology_edges]
    return SYSTEM_PROMPT.format(
        service_list=", ".join(service_list),
        topology_description="\n".join(topo_lines),
        tool_descriptions=tool_descriptions,
    )


def parse_react_response(text: str) -> tuple[str, str | None, dict[str, Any]]:
    """Parse LLM output into (thought, tool_name, args).

    Returns (thought, None, {}) if the response cannot be parsed or
    if ACTION is 'done'.
    """
    thought = ""
    tool_name: str | None = None
    args: dict[str, Any] = {}

    thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\nACTION:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"ACTION:\s*(\S+)", text)
    if action_match:
        raw = action_match.group(1).strip()
        if raw.lower() == "done":
            return thought, None, {}
        tool_name = raw

    args_match = re.search(r"ARGS:\s*(\{.*\})", text, re.DOTALL)
    if args_match:
        try:
            args = json.loads(args_match.group(1))
        except json.JSONDecodeError:
            pass

    return thought, tool_name, args
