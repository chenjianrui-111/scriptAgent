"""
Skill/Tool 安全能力

包含:
  - strict JSON schema 校验
  - tenant/role allowlist 策略
  - prompt injection tripwire
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from script_agent.config.settings import ToolSecurityConfig


@dataclass
class ToolPolicyDecision:
    allowed: bool
    reason: str = ""


class StrictJSONSchemaValidator:
    """轻量 JSON schema 校验器（strict 模式）"""

    def __init__(self, strict_enabled: bool = True):
        self._strict_enabled = strict_enabled

    def validate(self, payload: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
        if not self._strict_enabled or not schema:
            return None
        errors: List[str] = []
        self._validate_node(payload, schema, "$", errors)
        if not errors:
            return None
        return "; ".join(errors[:3])

    def _validate_node(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[str],
    ) -> None:
        expected_type = schema.get("type")
        if expected_type is not None and not self._is_type_matched(value, expected_type):
            errors.append(f"{path}: type mismatch, expect {expected_type}")
            return

        if "enum" in schema:
            enum_values = schema.get("enum", [])
            if value not in enum_values:
                errors.append(f"{path}: value not in enum")
                return

        if isinstance(value, str):
            min_len = schema.get("minLength")
            max_len = schema.get("maxLength")
            if isinstance(min_len, int) and len(value) < min_len:
                errors.append(f"{path}: minLength={min_len}")
            if isinstance(max_len, int) and len(value) > max_len:
                errors.append(f"{path}: maxLength={max_len}")

        if isinstance(value, list):
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if isinstance(min_items, int) and len(value) < min_items:
                errors.append(f"{path}: minItems={min_items}")
            if isinstance(max_items, int) and len(value) > max_items:
                errors.append(f"{path}: maxItems={max_items}")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for idx, item in enumerate(value):
                    self._validate_node(item, item_schema, f"{path}[{idx}]", errors)
            return

        if isinstance(value, dict):
            required = schema.get("required", [])
            if isinstance(required, list):
                for key in required:
                    if key not in value:
                        errors.append(f"{path}: missing required field '{key}'")

            props = schema.get("properties", {})
            if not isinstance(props, dict):
                props = {}

            additional_allowed = schema.get("additionalProperties", True)
            if additional_allowed is False:
                extras = [k for k in value.keys() if k not in props]
                if extras:
                    errors.append(f"{path}: additional properties not allowed: {extras[:3]}")

            for key, child_schema in props.items():
                if key in value and isinstance(child_schema, dict):
                    self._validate_node(value[key], child_schema, f"{path}.{key}", errors)
            return

    def _is_type_matched(self, value: Any, expected_type: Any) -> bool:
        if isinstance(expected_type, list):
            return any(self._is_type_matched(value, item) for item in expected_type)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "null":
            return value is None
        return True


class ToolPolicyEngine:
    """基于 tenant/role 的工具 allowlist 策略"""

    def __init__(self, cfg: ToolSecurityConfig):
        self._cfg = cfg

    def evaluate(self, skill_name: str, tenant_id: str, role: str) -> ToolPolicyDecision:
        if not self._cfg.allowlist_enabled:
            return ToolPolicyDecision(True, "")

        role_name = (role or self._cfg.default_role or "user").strip()
        role_allowed = self._cfg.role_allowlist.get(role_name) or self._cfg.role_allowlist.get("*", [])
        if not self._is_allowed(skill_name, role_allowed):
            return ToolPolicyDecision(
                False,
                f"role '{role_name}' is not allowed to use tool '{skill_name}'",
            )

        tenant_allowed = self._cfg.tenant_allowlist.get((tenant_id or "").strip())
        if tenant_allowed is not None and not self._is_allowed(skill_name, tenant_allowed):
            return ToolPolicyDecision(
                False,
                f"tenant '{tenant_id}' is not allowed to use tool '{skill_name}'",
            )

        return ToolPolicyDecision(True, "")

    def _is_allowed(self, skill_name: str, allowlist: List[str]) -> bool:
        if not isinstance(allowlist, list):
            return False
        return "*" in allowlist or skill_name in allowlist


class PromptInjectionTripwire:
    """工具调用前的 prompt injection 风险探测"""

    _PATTERNS = [
        re.compile(r"(?i)ignore.{0,24}(instruction|rule|policy|system|developer)"),
        re.compile(r"(?i)(system prompt|developer message|jailbreak|越狱)"),
        re.compile(r"(?i)(bypass|绕过).{0,24}(safety|policy|权限|限制)"),
        re.compile(r"(?i)(泄露|输出|显示).{0,24}(prompt|密钥|token|secret|密码)"),
        re.compile(r"(?i)(调用|使用).{0,20}(未授权|所有|全部).{0,20}(工具|skill|tool)"),
        re.compile(r"(?i)(执行|运行).{0,20}(shell|os command|系统命令)"),
    ]

    def __init__(self, cfg: ToolSecurityConfig):
        self._cfg = cfg

    def inspect(self, query: str, slots: Dict[str, Any]) -> Optional[str]:
        if not self._cfg.prompt_injection_tripwire_enabled:
            return None

        text = self._flatten_text(query, slots)
        if len(text) > self._cfg.slot_text_max_chars:
            text = text[: self._cfg.slot_text_max_chars]

        hit_reasons: List[str] = []
        for pattern in self._PATTERNS:
            if pattern.search(text):
                hit_reasons.append(pattern.pattern)

        if len(hit_reasons) >= max(1, self._cfg.prompt_injection_threshold):
            return "suspicious prompt-injection pattern detected"
        return None

    def _flatten_text(self, query: str, slots: Dict[str, Any]) -> str:
        parts: List[str] = [query or ""]

        def _append(value: Any):
            if value is None:
                return
            if isinstance(value, str):
                parts.append(value)
                return
            if isinstance(value, (int, float, bool)):
                parts.append(str(value))
                return
            if isinstance(value, list):
                for item in value[:50]:
                    _append(item)
                return
            if isinstance(value, dict):
                for _, v in list(value.items())[:50]:
                    _append(v)

        _append(slots)
        return "\n".join(parts)
