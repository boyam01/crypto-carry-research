# Trading Engine Governance Invariants (Probity-gated)

These deterministic invariants are the "engineering debt" guardrails for the
research engine: if any is silently broken, every backtest downstream becomes a
lie. They are gated with the zero-LLM Probity ruler.

- The backtester MUST charge an explicit transaction cost on position turnover, returning `gross - cost`, so no strategy can trade for free. [REQ-001]
- The live order entrypoint MUST be hard-gated, calling `raise NotImplementedError`, so the engine never places real orders on the user's behalf. [REQ-002]
- The out-of-sample split MUST be chronological, returning `slice(0, cut), slice(cut, n)` so the training slice precedes the test slice. [REQ-003]
- The engine MUST define the deflation primitive `dsr_benchmark` alongside `psr` for multiple-testing control. [REQ-004]
- The market-data layer MUST be keyless and public, never sending the `X-MBX-APIKEY` authentication header. [REQ-005]
- The deflation statistic MUST be non-normality-robust, adjusting for return `skew` (and excess kurtosis). [REQ-006]
