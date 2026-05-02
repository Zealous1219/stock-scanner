"""薄启动器，调用 scanner_app.main()，支持本地覆盖。"""

from __future__ import annotations

import os

from scanner_app import main as scanner_main

# 本地覆盖常量，设为 None 则使用 config.json,"black_horse","momentum_reversal_13"
STOCK_POOL = None
STRATEGY_NAME = None


def apply_local_overrides() -> None:
    """将本地常量设为环境变量，供 scanner 运行时读取。"""
    if STOCK_POOL:
        os.environ["STOCK_POOL"] = STOCK_POOL
    if STRATEGY_NAME:
        os.environ["STRATEGY_NAME"] = STRATEGY_NAME


if __name__ == "__main__":
    apply_local_overrides()
    scanner_main()
