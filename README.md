# Stock Scanner

## Entry Point

Current runtime entry is `scanner_app.py` -> `main()`.

`stock-scanner.py` is now a thin launcher only. It keeps optional local overrides and then calls `scanner_app.main()`.

## Runtime Precedence

When scanner starts, values are resolved in this order:

1. Environment override (highest)
   - `STOCK_POOL`
   - `STRATEGY_NAME`
2. Launcher local constants in `stock-scanner.py`
   - `STOCK_POOL`
   - `STRATEGY_NAME`
3. `config.json` (default)
   - `stock_pool.type`
   - `strategy.name`

So if you set `STOCK_POOL = "zz500"` in `stock-scanner.py`, it overrides `config.json`.

## How To Switch Strategy

Recommended way (stable and explicit): edit `config.json`.

```json
{
  "strategy": {
    "name": "black_horse",
    "params": {
      "required_weeks": 3,
      "min_weekly_bars": 12
    }
  },
  "stock_pool": {
    "type": "hs300"
  },
  "data": {
    "lookback_days": 180,
    "initial_days": 400,
    "request_interval": 0.5
  }
}
```

Valid built-in strategy names:

- `black_horse` (alias: `bh`)
- `moving_average` (alias: `ma`)

## How To Switch Stock Pool

Valid pool values:

- `hs300`
- `zz500`
- `sz50`
- `all`

You can switch by either:

1. Editing `config.json` -> `stock_pool.type`
2. Editing `stock-scanner.py` local `STOCK_POOL` override
3. Setting env variable before run: `set STOCK_POOL=zz500`

## Run Commands

Preferred interpreter is Python 3.13.

```powershell
.\install_requirements.ps1
.\check_runtime_env.ps1
.\run_scanner.ps1
```

Equivalent direct run:

```powershell
py -3.13 stock-scanner.py
```

## Output Files

Scanner writes into `output/`:

- Candidate file: `<strategy>_candidates_YYYY-MM-DD.csv`
- Run log: `strategy_runs_YYYY-MM-DD.csv`

## Notes

`black_horse` strategy uses the latest 3 completed weekly bars and ignores the current unfinished week.
