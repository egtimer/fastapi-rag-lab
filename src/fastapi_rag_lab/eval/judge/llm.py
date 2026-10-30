"""Claude client for the answer judge: token usage, cost and a hard budget cap.

Adapted from agentic-sql-assistant (``src/llm/``), without the Ollama path and
the tracing. Credentials come from the environment (``ANTHROPIC_API_KEY``);
every call is checked against the budget first and appended to a JSONL log.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

FAILURE_STOP_REASONS = frozenset({"refusal", "max_tokens"})


@dataclass(frozen=True)
class JudgeModel:
    """Model id, request settings and prices in USD per million tokens."""

    model: str
    input_per_mtok: float
    output_per_mtok: float
    max_tokens: int = 16000
    effort: str | None = "low"


# Prices from the platform.claude.com pricing page, as recorded in
# agentic-sql-assistant/configs/llm/ (checked 2026-10-18 and 2026-10-19).
# Judge prompts are far below Haiku's 100K long-context threshold.
MODELS = {
    "haiku": JudgeModel("claude-haiku-5-5", input_per_mtok=0.10, output_per_mtok=0.50),
    "sonnet": JudgeModel(
        "claude-sonnet-5-5", input_per_mtok=2.00, output_per_mtok=10.00
    ),
}


class BudgetExceeded(Exception):  # noqa: N818
    """Raised before a call when spending has reached the budget."""

    def __init__(self, spent: float, budget: float) -> None:
        super().__init__(f"budget of ${budget:.4f} reached (spent ${spent:.4f})")
        self.spent = spent
        self.budget = budget


@dataclass
class CallUsage:
    """Usage and cost of one call."""

    item_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def cost_of(usage: CallUsage, model: JudgeModel) -> float:
    return (
        usage.input_tokens * model.input_per_mtok
        + usage.output_tokens * model.output_per_mtok
    ) / 1_000_000


class BudgetedClient:
    """One ``complete`` method over the Anthropic SDK, with the budget checked first.

    The check happens before each call, so a run overshoots its budget by at most
    the cost of one call.
    """

    def __init__(
        self,
        model: JudgeModel,
        budget_usd: float,
        log_path: Path | None = None,
        sdk_client: Any = None,
    ) -> None:
        self.model = model
        self.budget_usd = budget_usd
        self.log_path = log_path
        self.spent = 0.0
        self._sdk = sdk_client

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            import anthropic

            self._sdk = anthropic.Anthropic()
        return self._sdk

    def complete(
        self,
        system: str,
        user: str,
        *,
        item_id: str,
        tool: dict[str, Any] | None = None,
    ) -> tuple[str, CallUsage]:
        """One call. With ``tool``, the model is forced to call it (structured
        output) and the returned text is that call's input as JSON.

        The reply text is written to the log whole, so every verdict, parsed
        or not, can be read back from ``llm_calls.jsonl``.
        """
        if self.spent >= self.budget_usd:
            raise BudgetExceeded(self.spent, self.budget_usd)
        kwargs: dict[str, Any] = {
            "model": self.model.model,
            "max_tokens": self.model.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.model.effort:
            kwargs["output_config"] = {"effort": self.model.effort}
        if tool is not None:
            kwargs["tools"] = [tool]
            kwargs["tool_choice"] = {"type": "tool", "name": tool["name"]}
        start = time.perf_counter()
        response = self._get_sdk().messages.create(**kwargs)
        if tool is not None:
            inputs = [
                b.input
                for b in response.content
                if getattr(b, "type", None) == "tool_use"
                and getattr(b, "name", None) == tool["name"]
            ]
            text = json.dumps(inputs[0], ensure_ascii=False) if inputs else ""
        else:
            text = "".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
        usage = CallUsage(
            item_id=item_id,
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
            latency_ms=(time.perf_counter() - start) * 1000,
            stop_reason=response.stop_reason,
        )
        usage.cost_usd = cost_of(usage, self.model)
        self.spent += usage.cost_usd
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"model": self.model.model, **asdict(usage), "text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return text, usage
