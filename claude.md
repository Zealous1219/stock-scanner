# A-Share Quant Framework

## Purpose
This is a configuration-driven A-share stock screening and backtesting system.

Focus: correctness > reproducibility > maintainability > performance.

---

## Architecture (DO NOT CHANGE)
- Entry: stock-scanner.py
- App: scanner_app.py
- Config: config_loader.py
- Strategies: strategies/ (BaseStrategy + compute())
- Data: data_utils.py
- Runtime: strategy_runtime.py

Separation of concerns is critical:
data / strategy / backtest / output must stay independent.

---

## Strategy System
- All strategies inherit from BaseStrategy
- Main interface: compute()
- Strategies must be stateless and independent
- Strategy switching is config-driven via create_strategy_from_config()
- Do NOT hardcode strategy selection

---

## Data Rules
- Data source: baostock
- Cached data stored in /data
- Cache must be reused when possible
- Do NOT modify data pipeline or cache logic without explicit reason

---

## Backtest Rules
- Must be deterministic and reproducible
- No hidden randomness
- Do NOT change execution, fee, or slippage logic silently

---

## Critical Constraints (IMPORTANT)
- Do NOT change existing strategy behavior
- Do NOT refactor multiple layers at once
- Do NOT rewrite architecture in one pass
- Keep changes minimal and incremental
- Always explain impact before modifying code

---

## Working Method
1. Understand code first
2. Propose plan
3. Execute step-by-step (one change at a time)
4. Verify result after each step

---

## Known Risks
- baostock API instability
- missing tests
- potential coupling between strategy and backtest
- cache inconsistency risk

Treat these as constraints, not reasons for redesigning everything.

---

## Goal
Make strategies pluggable, safe, and easy to extend without breaking existing logic.