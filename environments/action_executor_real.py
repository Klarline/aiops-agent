"""Real action executor for live infrastructure remediation.

Dispatches agent remediation actions to actual
infrastructure APIs: kubectl for Kubernetes workloads, configurable
webhooks for firewall/WAF rules and rate limiting.

Safety: AIOPS_DRY_RUN=true (the default) logs all actions without
executing them. Set to false only when the agent has been validated
in shadow mode and credentials are properly mounted.

Credentials must be injected as mounted secrets or environment variables,
NOT committed to .env files.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

import httpx

from environments.types import ActionResult

logger = logging.getLogger(__name__)


class RealActionExecutor:
    """Dispatches remediation actions to real infrastructure.

    Usage:
        executor = RealActionExecutor.from_env()
        env.attach_action_executor(executor)
    """

    def __init__(
        self,
        dry_run: bool = True,
        kubectl_context: str | None = None,
        kubectl_namespace: str = "default",
        block_ip_webhook: str | None = None,
        rate_limit_webhook: str | None = None,
        http_timeout: float = 30.0,
    ) -> None:
        self.dry_run = dry_run
        self._kubectl_context = kubectl_context
        self._kubectl_namespace = kubectl_namespace
        self._block_ip_webhook = block_ip_webhook
        self._rate_limit_webhook = rate_limit_webhook
        self._http_timeout = http_timeout

        if dry_run:
            logger.info("RealActionExecutor: DRY RUN mode — actions will be logged but not executed")
        else:
            logger.warning("RealActionExecutor: LIVE mode — actions will be executed on real infrastructure")

    @classmethod
    def from_env(cls) -> "RealActionExecutor":
        """Build executor from environment variables."""
        return cls(
            dry_run=os.getenv("AIOPS_DRY_RUN", "true").lower() != "false",
            kubectl_context=os.getenv("KUBECTL_CONTEXT"),
            kubectl_namespace=os.getenv("KUBECTL_NAMESPACE", "default"),
            block_ip_webhook=os.getenv("BLOCK_IP_WEBHOOK_URL"),
            rate_limit_webhook=os.getenv("RATE_LIMIT_WEBHOOK_URL"),
        )

    def execute(self, action: str, target: str, **kwargs: Any) -> ActionResult:
        """Execute a remediation action, respecting dry_run mode."""
        if self.dry_run:
            return ActionResult(
                True, action, target,
                f"[dry-run] Would execute: {action} on {target} kwargs={kwargs}",
            )

        dispatch = {
            "restart_service": self._restart,
            "scale_out": self._scale,
            "rollback": self._rollback,
            "block_ip": self._block_ip,
            "rate_limit": self._rate_limit,
            "alert_human": self._alert_human,
            "continue_monitoring": lambda t, **kw: ActionResult(True, "continue_monitoring", t, "Monitoring"),
        }
        handler = dispatch.get(action)
        if handler is None:
            return ActionResult(False, action, target, f"Unknown action: {action}")
        try:
            return handler(target, **kwargs)
        except Exception as exc:
            logger.error("action %s on %s failed: %s", action, target, exc)
            return ActionResult(False, action, target, str(exc))

    # ------------------------------------------------------------------
    # Kubernetes actions
    # ------------------------------------------------------------------

    def _restart(self, service: str, **_: Any) -> ActionResult:
        ok, out, err = self._kubectl("rollout", "restart", f"deployment/{service}")
        return ActionResult(ok, "restart_service", service, (out or err).strip())

    def _scale(self, service: str, replicas: int = 3, **_: Any) -> ActionResult:
        ok, out, err = self._kubectl("scale", f"deployment/{service}", f"--replicas={replicas}")
        return ActionResult(ok, "scale_out", service, (out or err).strip())

    def _rollback(self, service: str, **_: Any) -> ActionResult:
        ok, out, err = self._kubectl("rollout", "undo", f"deployment/{service}")
        return ActionResult(ok, "rollback", service, (out or err).strip())

    def _kubectl(self, *args: str) -> tuple[bool, str, str]:
        """Run a kubectl command and return (success, stdout, stderr)."""
        cmd = ["kubectl"]
        if self._kubectl_context:
            cmd += ["--context", self._kubectl_context]
        cmd += ["--namespace", self._kubectl_namespace]
        cmd += list(args)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "kubectl timed out after 60s"
        except FileNotFoundError:
            return False, "", "kubectl not found — is it installed and on PATH?"

    # ------------------------------------------------------------------
    # Webhook actions
    # ------------------------------------------------------------------

    def _block_ip(self, target: str, ip: str = "", **_: Any) -> ActionResult:
        if not self._block_ip_webhook:
            return ActionResult(
                True, "block_ip", target,
                f"[no webhook] IP {ip} would be blocked on {target}. "
                "Set BLOCK_IP_WEBHOOK_URL to enable.",
            )
        try:
            resp = httpx.post(
                self._block_ip_webhook,
                json={"ip": ip, "service": target},
                timeout=self._http_timeout,
            )
            resp.raise_for_status()
            return ActionResult(True, "block_ip", target, f"IP {ip} blocked via webhook")
        except Exception as exc:
            return ActionResult(False, "block_ip", target, str(exc))

    def _rate_limit(self, target: str, **_: Any) -> ActionResult:
        if not self._rate_limit_webhook:
            return ActionResult(
                True, "rate_limit", target,
                f"[no webhook] Rate limit would be applied to {target}. "
                "Set RATE_LIMIT_WEBHOOK_URL to enable.",
            )
        try:
            resp = httpx.post(
                self._rate_limit_webhook,
                json={"service": target},
                timeout=self._http_timeout,
            )
            resp.raise_for_status()
            return ActionResult(True, "rate_limit", target, "Rate limit applied via webhook")
        except Exception as exc:
            return ActionResult(False, "rate_limit", target, str(exc))

    def _alert_human(self, target: str, message: str = "", **_: Any) -> ActionResult:
        logger.warning("HUMAN ESCALATION: service=%s message=%s", target, message)
        return ActionResult(True, "alert_human", target, f"Escalated to human operator: {message}")
