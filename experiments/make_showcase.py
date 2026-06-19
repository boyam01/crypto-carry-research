"""Portfolio showcase figure: (1) honest research funnel, (2) deployable basis-carry equity.

Reuses experiments/basis_carry_backtest.py logic (coin_stream) so the equity curve is the
exact end-to-end playbook track record (BTC/ETH quarterly cash-and-carry + funding satellite),
read offline from the warm parquet cache. Saves reports/showcase.png at dpi=170.
"""
from __future__ import annotations
import sys, time, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.dates as mdates

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import backtest as bt
from engine.basis_carry_live import PARAMS as P
import experiments.basis_carry_backtest as bcb

COINS = bcb.COINS


def build_equity():
    """Recompute the end-to-end book daily return + equity (unlev & 3x) with OOS split."""
    end = int(time.time() * 1000)
    start = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
    streams, flags = {}, {}
    for c in COINS:
        s, fl = bcb.coin_stream(c, start, end)
        streams[c], flags[c] = s, fl
    df = pd.DataFrame(streams).dropna(how="all").fillna(0.0)
    book = df.mean(axis=1)                              # equal-weight BTC/ETH daily return
    idx = df.index
    n = len(book)
    cut = int(n * 0.6)                                  # chronological 60/40 IS/OOS split
    lev = P["max_gross_leverage"]
    eq1 = np.cumprod(1 + book.values)
    eq3 = np.cumprod(1 + book.values * lev)
    oos = bt.metrics((book.values)[cut:], 365)
    oos_sr = oos["sharpe_ann"]
    full_sr = bt.metrics(book.values, 365)["sharpe_ann"]
    frac_basis = pd.DataFrame(flags).reindex(idx).fillna(False).mean(axis=1).mean()
    return dict(idx=idx, eq1=eq1, eq3=eq3, cut=cut, lev=lev, oos_sr=oos_sr,
                full_sr=full_sr, frac_basis=frac_basis,
                oos_ret=oos["ret_ann"], oos_dd=oos["maxdd"])


def main():
    d = build_equity()

    # palette
    INK = "#1b1f24"; MUT = "#6b7280"; GRID = "#e6e8eb"
    KILL = "#c0392b"; SURV = "#1f8a4c"; BLUE = "#2563cb"; GOLD = "#d39a00"
    OOSF = "#eef6f0"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.edgecolor": "#cfd3d8", "axes.linewidth": 0.9,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUT, "ytick.color": MUT, "axes.titlecolor": INK,
    })

    fig = plt.figure(figsize=(15.0, 6.6), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.16,
                          left=0.045, right=0.975, top=0.84, bottom=0.10)

    fig.suptitle("Crypto Quant Research  --  intellectual honesty over hype",
                 x=0.045, ha="left", fontsize=17, fontweight="bold", color=INK, y=0.955)
    fig.text(0.045, 0.895,
             "~40 methods stress-tested across 3 waves with adversarial + cross-engine (Codex) verification.  "
             "Almost all killed. One robust edge survived.",
             ha="left", fontsize=10.3, color=MUT)

    # ============================ PANEL 1: research funnel ============================
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("1  --  The research funnel (what survived honesty)",
                 loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # funnel stages: (label, sublabel, count_for_width, color)
    stages = [
        ("~40 methods tested", "3 waves: formula battery / exotic math / 4,120-formula mining", 40, "#3b4654"),
        ("Survive in-sample", "many looked great on IS Sharpe alone", 24, "#5b6b7d"),
        ("Survive cost + OOS", "formula alphas: gross-of-cost = 0 across 11 angles", 6, GOLD),
        ("Survive stationarity", "IS-vs-OOS shape corr = -0.507  (alpha flips sign OOS)", 2, KILL),
        ("Robust survivor", "quarterly futures cash-and-carry basis", 1, SURV),
    ]
    # geometry: stacked horizontal bands, centered, width ~ sqrt-ish of count
    y0, y1 = 92, 8
    band_h = (y0 - y1) / len(stages)
    maxw, minw = 92.0, 20.0
    counts = np.array([s[2] for s in stages], float)
    widths = minw + (maxw - minw) * (np.sqrt(counts) - np.sqrt(counts.min())) / \
        (np.sqrt(counts.max()) - np.sqrt(counts.min()))
    cx = 50.0
    for i, ((lab, sub, cnt, col), w) in enumerate(zip(stages, widths)):
        top = y0 - i * band_h
        bot = top - band_h * 0.78
        # trapezoid: this band's top width -> next band's width
        wt = w
        wb = widths[i + 1] if i + 1 < len(widths) else w * 0.62
        poly = plt.Polygon([(cx - wt / 2, top), (cx + wt / 2, top),
                            (cx + wb / 2, bot), (cx - wb / 2, bot)],
                           closed=True, facecolor=col, edgecolor="white", linewidth=2.0,
                           alpha=0.93, zorder=2)
        ax.add_patch(poly)
        midy = (top + bot) / 2
        ax.text(cx, midy + 2.4, lab, ha="center", va="center", color="white",
                fontsize=11.0, fontweight="bold", zorder=3)
        ax.text(cx, midy - 3.0, sub, ha="center", va="center", color="white",
                fontsize=7.7, zorder=3, alpha=0.95)
        # count chip on the right
        chip = "1" if cnt == 1 else ("~40" if cnt == 40 else str(int(cnt)))
        ax.text(cx + wt / 2 + 3.5, top - band_h * 0.39, chip, ha="left", va="center",
                color=col, fontsize=12.5, fontweight="bold", zorder=3)

    # kill annotations on the left for the two big drops
    ax.annotate("KILLED:\ncosts eat the\nentire edge",
                xy=(cx - widths[2] / 2, y0 - 2 * band_h - band_h * 0.39),
                xytext=(3, y0 - 2 * band_h - band_h * 0.39),
                ha="left", va="center", fontsize=8.0, color=KILL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=KILL, lw=1.4))
    ax.annotate("KILLED:\nnon-stationary\n(alpha flips OOS)",
                xy=(cx - widths[3] / 2, y0 - 3 * band_h - band_h * 0.39),
                xytext=(3, y0 - 3 * band_h - band_h * 0.39),
                ha="left", va="center", fontsize=8.0, color=KILL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=KILL, lw=1.4))
    ax.text(50, 2.0, "Width ~ count of methods. The point of the project: kill the hype, keep what is real.",
            ha="center", va="center", fontsize=8.0, color=MUT, style="italic")

    # ============================ PANEL 2: deployable equity ============================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("2  --  The deployable edge: basis cash-and-carry equity",
                  loc="left", fontsize=12.5, fontweight="bold", pad=10)

    idx = d["idx"]; eq1 = d["eq1"]; eq3 = d["eq3"]; cut = d["cut"]
    oos_start = idx[cut]

    # OOS shaded region
    ax2.axvspan(oos_start, idx[-1], color=OOSF, zorder=0)
    ax2.axvline(oos_start, color=SURV, lw=1.1, ls="--", alpha=0.7, zorder=1)

    ax2.plot(idx, eq3, color=GOLD, lw=2.3, zorder=4,
             label=f"3x leveraged  (+{(eq3[-1]-1)*100:.0f}% total)")
    ax2.plot(idx, eq1, color=BLUE, lw=2.3, zorder=4,
             label=f"unleveraged  (+{(eq1[-1]-1)*100:.1f}% total)")
    ax2.axhline(1.0, color=MUT, lw=0.8, ls=":", alpha=0.6, zorder=1)

    ax2.set_ylabel("growth of $1 (cumulative equity)")
    ax2.grid(True, color=GRID, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    # x axis dates
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    # in-sample / out-of-sample labels
    ymax = eq3.max()
    ax2.text(idx[cut // 2], ymax * 0.995, "in-sample (60%)",
             ha="center", va="top", fontsize=9, color=MUT, fontweight="bold")
    ax2.text(idx[cut + (len(idx) - cut) // 2], ymax * 0.995, "OUT-OF-SAMPLE (40%)",
             ha="center", va="top", fontsize=9, color=SURV, fontweight="bold")

    # OOS Sharpe annotation box, anchored cleanly inside the OOS band (lower-right)
    txt = (f"OOS Sharpe $\\approx$ {d['oos_sr']:.0f}\n"
           f"OOS return = +{d['oos_ret']*100:.1f}%/yr (unlev)\n"
           f"OOS max DD = {d['oos_dd']*100:.2f}%")
    ax2.text(0.965, 0.045, txt, transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=9.6, color=INK, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=SURV, lw=1.4, alpha=0.96),
             zorder=6)

    leg = ax2.legend(loc="upper left", bbox_to_anchor=(0.012, 0.86), frameon=True,
                     fontsize=9.6, framealpha=0.95, edgecolor="#d0d4d9")
    leg.get_frame().set_linewidth(0.8)

    ax2.text(0.5, -0.165,
             "BTC + ETH quarterly cash-and-carry (delta-neutral) + funding-carry satellite | costs charged | "
             f"{idx[0].date()}..{idx[-1].date()} | settlement-locked convergence, 12/12 contracts positive",
             transform=ax2.transAxes, ha="center", va="top", fontsize=7.9, color=MUT)

    out = ROOT / "reports" / "showcase.png"
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"SAVED {out}")
    print(f"full Sharpe={d['full_sr']:.2f}  OOS Sharpe={d['oos_sr']:.2f}  "
          f"OOS ret={d['oos_ret']*100:.2f}%/yr  OOS maxDD={d['oos_dd']*100:.2f}%")
    print(f"unlev final={eq1[-1]:.3f}x  3x final={eq3[-1]:.3f}x  "
          f"basis-day frac={d['frac_basis']*100:.0f}%  n_days={len(idx)}")


if __name__ == "__main__":
    main()
