# Weekly Replay Audit Follow-Up

## Background

This document records the follow-up work completed after a high-risk engineering audit of the weekly replay flow. The goal of the audit was to improve replay correctness, resume safety, multi-strategy isolation, and weekly-bar semantic consistency without rewriting the replay architecture.

The changes summarized here were intentionally kept local and incremental. Strategy logic, forward-return logic, checkpoint granularity, and the broader replay architecture were preserved unless a specific defect required adjustment.

## Completed Remediation Items

### 1. Per-snapshot replay outputs replaced aggregate append-based CSV writing

Previously, weekly replay appended all snapshot results and errors into shared aggregate CSV files. This allowed duplicate rows when a snapshot wrote to CSV successfully but crashed before checkpoint completion.

This was changed to:

- one result file per snapshot
- one error file per snapshot
- overwrite-on-rerun semantics for the same snapshot
- checkpoint remaining the only progress truth

Result:

- snapshot reruns no longer create duplicate rows
- output isolation is cleaner
- aggregate files are no longer treated as runtime truth

### 2. Empty-result snapshots now write stable carrier files

A regression appeared during the per-snapshot refactor: a snapshot with zero matches could be marked completed in checkpoint without a corresponding result file, which broke resume consistency checks.

This was fixed by always writing a per-snapshot result carrier file, even when the snapshot has zero matched records.

Result:

- completed snapshots always have a stable result file
- resume consistency checks remain valid

### 3. Failed symbols are now tracked explicitly

Originally, symbol-level failures inside a snapshot were only written to error CSV output. The snapshot could still be marked completed, and the missing symbols were not tracked structurally in checkpoint state.

Checkpoint schema was extended with:

- `failed_symbols_per_snapshot`

Result:

- replay still completes at snapshot granularity
- unresolved failed symbols are explicitly recorded
- startup and completion logs now surface unresolved omissions
- failures are visible rather than silently disappearing

### 4. `black_horse` replay now produces valid forward returns

`black_horse` originally did not populate `signal_date`, which caused replay forward-return calculation to produce `NaN`.

This was fixed by defining `signal_date` as the third and latest completed weekly bar end date, which matches the intended strategy semantics.

Result:

- `black_horse` now reuses the shared replay and forward-return pipeline correctly
- no strategy-specific replay special case was introduced

### 5. MR13 volume MA20 no longer treats row count as calendar-week continuity

`momentum_reversal_13` used `weekly.iloc[idx-20:idx]` for volume MA20, which implicitly assumed that 20 weekly rows represented 20 calendar weeks. This could pull in stale history if weekly data contained gaps.

This was fixed by adding a continuity check before computing MA20:

- still uses the prior 20 weekly rows
- but first verifies their calendar span is within a conservative threshold
- otherwise returns `None`

Result:

- normal continuous weekly data behaves as before
- clearly gapped weekly windows no longer pollute MA20 with overly old bars

### 6. `load_or_update_data()` now uses a single captured `now`

The function previously called `datetime.now()` multiple times, creating a theoretical midnight-boundary inconsistency where `today`, `yesterday`, and refresh decisions could disagree if execution crossed 00:00.

This was fixed by capturing `now` once at function entry and deriving all related values from it.

Result:

- intra-call time semantics are consistent
- behavior remains otherwise unchanged

### 7. Replay file paths are now strategy-isolated

Replay `experiment_tag` already included strategy identity, but checkpoint paths and per-snapshot replay files did not. Different strategies could therefore collide in the same replay directory.

All replay-related file naming was updated to include `strategy_slug`:

- checkpoint files
- per-snapshot result files
- per-snapshot error files
- merged helper outputs

Result:

- multiple replay strategies can coexist safely
- `mr13` and `black_horse` no longer share checkpoint or result namespaces

### 8. Replay strategy selection now comes from config

Replay entry previously hardcoded `momentum_reversal_13`, even after strategy-specific output isolation was implemented.

This was changed so `run_weekly_replay_validation()` now reads:

- `config["strategy"]["name"]`

with safe fallback to:

- `momentum_reversal_13`

Result:

- switching replay strategy now only requires a config change
- MR13 slug compatibility remains intact

### 9. Historical snapshot trading-week metadata is now explicit

Historical snapshot generation still uses natural-week anchors, which can include a week that had no trading days at all. This usually did not break replay, but it weakened sample explainability.

A new helper was added:

- `get_snapshot_trading_week_info()`

It distinguishes among:

- a valid historical trading week
- a fully non-trading week
- a trading-calendar query failure

Replay now logs this metadata lightly without changing replay behavior.

Result:

- historical snapshot semantics are more transparent
- analysis can distinguish true trading weeks from calendar-only anchors

### 10. Calendar-query-failure guard is now aligned between main and weekly fallback replay paths

The optimized weekly replay path already handled unknown calendar state conservatively by dropping the snapshot-week last bar when completion could not be confirmed. The fallback weekly path could theoretically diverge.

A shared guard was introduced and applied consistently to:

- the main precomputed weekly slice path
- the weekly fallback replay path

The fallback logic was explicitly narrowed so that:

- weekly strategies may receive guarded `precomputed_weekly`
- non-weekly strategies continue to use the original `strategy.scan(symbol, df, context)` path without `precomputed_weekly`

Result:

- weekly replay semantics are aligned when calendar query state is unknown
- non-weekly strategy call semantics remain unchanged

## Current State

The weekly replay flow is now materially more stable and interpretable than before the audit.

Key properties now in place:

- per-snapshot output truth instead of aggregate append truth
- checkpoint-driven resume behavior
- explicit visibility into failed symbols
- multi-strategy replay coexistence
- config-driven replay strategy selection
- clearer historical trading-week metadata
- better semantic alignment across weekly replay execution paths

## Remaining Non-Blocking Observations

No major audit item remains open on the original high-risk path.

The remaining observations are lower priority and mainly concern future polish rather than correctness blockers:

- some tests could be strengthened further over time for behavior-level confidence
- replay experiment parameters such as universe, lookback window, and version are still partially code-defined rather than fully config-driven
- trading-week metadata currently improves logging and interpretation, but is not yet systematically persisted into replay result schema

These are reasonable future improvements, but they are not required to consider the current audit follow-up complete.

## Conclusion

The replay-related audit follow-up can now be considered substantially complete.

The system has moved from a state where replay correctness and recoverability depended on several fragile assumptions to one where:

- replay state is more explicit
- reruns are safer
- strategy isolation is cleaner
- weekly semantics are better defined
- edge-case behavior is more conservative and more visible

This document should be updated only if future replay architecture changes intentionally alter the assumptions summarized above.
