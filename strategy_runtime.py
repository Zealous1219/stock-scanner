"""Runtime data structures shared by the scanner and strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class StrategyContext:
    """Runtime context shared with each strategy."""

    now: datetime
    stock_pool: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyDecision:
    """Whether a strategy should run for the current invocation."""

    should_run: bool
    reason_code: str
    reason_text: str


@dataclass(frozen=True)
class StrategyResult:
    """Structured scan result for a single symbol."""

    matched: bool
    reason_code: str
    reason_text: str
    details: Dict[str, Any] = field(default_factory=dict)
