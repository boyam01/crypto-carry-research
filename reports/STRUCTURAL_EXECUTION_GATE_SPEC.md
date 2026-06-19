# Structural execution qualification gate

## Purpose

This gate closes the gap between a **research-pass edge** and a **deployable edge** without pretending that a backtest or one current order-book snapshot proves execution.

The probe is intentionally read-only. It consumes locked JSON artifacts, applies structural sleeve weights and hedge multipliers, and produces a portfolio-level execution capacity plus an explicit blocker matrix. It does not fetch exchange data, rerun research, place orders, or mutate source artifacts.

A passing research statistic is necessary but not sufficient. The deploy gate opens only when every required sleeve/leg has certified selected-window depth and fill evidence, every operational gate passes, and the lower-bound portfolio capacity clears the configured safety multiple.

## Three capacity layers

1. **Current-depth proxy**
   - Diagnostic only.
   - Useful for identifying the likely binding sleeve or leg.
   - Never opens the deploy gate by itself.

2. **Timed-depth proxy**
   - Must match the strategy's selected execution window.
   - Still diagnostic unless its artifact is certified and all configured sample/coverage requirements pass.

3. **Certified execution capacity**
   - Uses the lower confidence bound from every required evidence family.
   - Requires both timed-depth and fill evidence by default.
   - Is the only capacity used by `deploy_gate`.

## Capacity identity

For portfolio notional `N`, structural sleeve weight `w_s`, and leg hedge multiplier `h_sl`, leg `l` must execute:

```text
leg_notional = N * abs(w_s) * abs(h_sl)
```

If the certified leg capacity is `C_sl`, that leg supports portfolio capacity:

```text
portfolio_capacity_sl = C_sl / (abs(w_s) * abs(h_sl))
```

The portfolio capacity is the bottleneck across all required sleeves and legs:

```text
portfolio_capacity = min(portfolio_capacity_sl)
```

For multiple evidence families, the certified leg capacity is itself fail-closed:

```text
certified_leg_capacity = min(timed_depth_lcb, fill_lcb, ...)
```

For a target portfolio `N_target` and safety multiple `H`, the qualification hurdle is:

```text
required_capacity = H * N_target
```

A useful sizing diagnostic is:

```text
max_structural_weight_s = certified_sleeve_capacity_s / required_capacity
```

This is a diagnostic ceiling, not permission to silently reweight. Any material reweighting creates a new portfolio variant and must return through the research/statistical gate.

## Qualification contract

The default contract requires:

- portfolio structural weights sum to the configured target;
- every sleeve's `research_gate` is `PASS`;
- required operational gates are `PASS`;
- every required execution record has `status=PASS`;
- capacity is stated in USD, not raw contracts or coins;
- the record is explicitly `certified=true`;
- a lower-confidence-bound capacity is present;
- selected-window depth matches the configured execution window;
- minimum sample count and coverage pass;
- fill evidence comes from an allowed mode such as replay, shadow, or live observation;
- certified portfolio capacity is at least `target_notional_usd * capacity_safety_multiple`.

Any missing, unknown, stale, mismatched, uncertified, or under-capacity record leaves the gate `CLOSED` and `deployable_count=0`.

## Candidate-specific interpretation

### CVaR funding carry

This is the shortest path to an independently qualified sleeve if its research gate already passes. It still needs execution evidence at the actual funding-entry/exit window, including both legs, realistic partial fills, fees, slippage, and any borrow requirement. A current-depth pass is not a substitute for timed depth or fills.

### COIN-M basis

COIN-M capacity must be normalized to USD using the contract multiplier and the actual inverse-contract hedge multiplier before structural weighting. Both spot and futures legs must independently pass. The weaker leg binds. Contract-count capacity is rejected by the gate.

A venue/schema gate should also be required so that exchange endpoint, symbol metadata, multiplier, and stream semantics used by the artifacts match the production venue version.

### Residual spectral TWAP

TWAP is treated as an execution policy or capacity-improvement overlay, not an independent alpha source. It becomes qualification evidence only after selected-window replay/shadow fills show that it improves certified lower-bound capacity without degrading implementation shortfall, adverse selection, completion probability, or timing risk beyond the locked limits.

## Artifact shape

Evidence may be embedded inline or read from an existing JSON artifact:

```json
{
  "artifact": "artifacts/cvar_timed_depth.json",
  "json_pointer": "/qualification/BTCUSDT",
  "status": "PASS",
  "certified": true,
  "capacity_lcb_usd": 1250000,
  "window": "00:00-08:00 UTC",
  "sample_count": 240,
  "coverage": 0.98,
  "unit": "USD"
}
```

Inline fields override fields loaded from the artifact. This permits a locked artifact to expose a larger result object while the manifest selects one precise qualification record.

## Required output

The report contains:

- `current_depth_proxy` portfolio capacity and binding sleeve/leg;
- `timed_depth_proxy` portfolio capacity and binding sleeve/leg;
- certified capacity by sleeve and leg;
- maximum supported structural weight at the required capacity;
- the final binding sleeve and leg;
- an execution blocker matrix with deterministic codes;
- `deploy_gate` and `deployable_count`.

## Run

```bash
python engine/structural_execution_probe.py \
  reports/structural_execution_manifest.json \
  --json-out artifacts/structural_execution_gate.json \
  --md-out reports/STRUCTURAL_EXECUTION_GATE_REPORT.md \
  --require-open
```

Exit codes:

- `0`: evaluation completed; and, with `--require-open`, the gate is open;
- `2`: malformed manifest or unreadable artifact;
- `3`: evaluation completed but the deploy gate remains closed.

## Non-negotiable semantics

This probe can convert **existing valid evidence** into a portfolio qualification verdict. It cannot convert absent evidence into an edge. Until timed-depth, fill, watch, risk, operations, and venue-schema artifacts satisfy the contract, the honest output remains:

```text
deploy_gate = CLOSED
deployable_count = 0
```
