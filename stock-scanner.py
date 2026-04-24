"""Thin launcher for stock scanner.

Single source of runtime logic:
- scanner_app.main

This file only provides optional local overrides for quick manual runs.
"""

from __future__ import annotations

import os

from scanner_app import main as scanner_main

# Optional local overrides for manual runs.
# Set to None to use config.json values.
#默认保持None，仅用于临时调试
STOCK_POOL = None  # hs300 | zz500 | sz50 | all | None
STRATEGY_NAME = None  # black_horse | moving_average | None


def apply_local_overrides() -> None:
    """Expose optional local constants as environment overrides."""
    if STOCK_POOL:
        os.environ["STOCK_POOL"] = STOCK_POOL

    if STRATEGY_NAME:
        os.environ["STRATEGY_NAME"] = STRATEGY_NAME


if __name__ == "__main__":
    apply_local_overrides()
    scanner_main()
