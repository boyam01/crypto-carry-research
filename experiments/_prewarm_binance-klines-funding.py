"""Prewarm Binance parquet cache: 1d/8h futures klines, 1d spot klines, funding.
slice = binance-klines-funding. No strategy computation here."""
import sys, time
from datetime import datetime, timezone
sys.path.insert(0, '.')
from engine import fetch_binance as fb

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
         "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
         "MATICUSDT", "NEARUSDT", "ATOMUSDT"]

START = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END = int(datetime.now(timezone.utc).timestamp() * 1000)

results = {}
for c in COINS:
    row = {}
    for label, fn in [
        ("fut_1d", lambda c=c: fb.klines(c, "1d", START, END, futures=True)),
        ("fut_8h", lambda c=c: fb.klines(c, "8h", START, END, futures=True)),
        ("spot_1d", lambda c=c: fb.klines(c, "1d", START, END, futures=False)),
        ("funding", lambda c=c: fb.funding_rate(c, START, END)),
    ]:
        try:
            df = fn()
            row[label] = 0 if df is None else len(df)
        except Exception as e:
            row[label] = f"ERR:{type(e).__name__}:{e}"
        time.sleep(0.05)
    results[c] = row
    print(f"{c:10s} | fut_1d={row['fut_1d']!s:>6} fut_8h={row['fut_8h']!s:>6} "
          f"spot_1d={row['spot_1d']!s:>6} funding={row['funding']!s:>6}")
    sys.stdout.flush()

def ok(v):
    return isinstance(v, int) and v > 0

succeeded = [c for c, r in results.items() if all(ok(r[k]) for k in r)]
partial = [c for c, r in results.items()
           if c not in succeeded and any(ok(r[k]) for k in r)]
failed = [c for c, r in results.items() if not any(ok(r[k]) for k in r)]

print("\n=== SUMMARY ===")
print(f"FULL_OK ({len(succeeded)}): {succeeded}")
print(f"PARTIAL ({len(partial)}): {partial}")
print(f"FAILED  ({len(failed)}): {failed}")
for c in partial + failed:
    print(f"  {c}: {results[c]}")
