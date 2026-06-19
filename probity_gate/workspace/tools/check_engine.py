"""Deterministic, zero-LLM checker for the engine's governance invariants.
For each invariant that HOLDS, prints the spec's backticked code literal. A
mutation that breaks an invariant removes that literal from stdout -> the Probity
check (stdout_contains the literal) fails -> mutant killed. Exits 0 always."""
import pathlib, re

root = pathlib.Path(__file__).resolve().parent.parent
src = root / "src"
bt = (src / "backtest.py").read_text(encoding="utf-8")
bk = (src / "book.py").read_text(encoding="utf-8")
fb = (src / "fetch_binance.py").read_text(encoding="utf-8")

# REQ-001: explicit transaction cost charged on position turnover -> `gross - cost`
if re.search(r"np\.abs\(np\.diff\(position", bt) and "cost_bps / 1e4" in bt and "return gross - cost" in bt:
    print("REQ-001 ok: gross - cost")

# REQ-002: live order entrypoint raises NotImplementedError -> `raise NotImplementedError`
m = re.search(r"def place_live_order\(.*?\):(.*?)(\ndef |\Z)", bk, re.S)
if m and "raise NotImplementedError" in m.group(1):
    print("REQ-002 ok: place_live_order will raise NotImplementedError")

# REQ-003: chronological OOS split -> `slice(0, cut), slice(cut, n)`
if re.search(r"return\s+slice\(0,\s*cut\),\s*slice\(cut,\s*n\)", bt):
    print("REQ-003 ok: slice(0, cut), slice(cut, n)")

# REQ-004: deflation primitives present -> `psr` + `dsr_benchmark`
if "def psr(" in bt and "def dsr_benchmark(" in bt:
    print("REQ-004 ok: psr and dsr_benchmark")

# REQ-005: data layer is keyless/public -> the auth header `X-MBX-APIKEY` is ABSENT
if "X-MBX-APIKEY" not in fb and "api_key" not in fb.lower():
    print("REQ-005 ok: keyless public data layer, no X-MBX-APIKEY auth header")

# REQ-006: deflation is non-normality-robust -> PSR adjusts for `skew` and `kurt`
if "skew * sr_pp" in bt and "(kurt - 1)" in bt:
    print("REQ-006 ok: PSR adjusts for skew and kurt")
