"""运行时数据结构：Context / Decision / Result。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class StrategyContext:
    """策略运行时上下文。"""

    now: datetime
    stock_pool: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyDecision:
    """策略是否应执行本次扫描。"""

    should_run: bool
    reason_code: str
    reason_text: str


@dataclass(frozen=True)
class StrategyResult:
    """单标的扫描结果。"""

    matched: bool
    reason_code: str
    reason_text: str
    details: Dict[str, Any] = field(default_factory=dict)
