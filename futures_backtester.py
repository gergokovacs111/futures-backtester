#!/usr/bin/env python3
# backtester.py — simple pure-Python backtester (SMA crossover strategy)
# Major updates:
#   • Signals are shifted 1 day forward → execution happens at NEXT DAY OPEN (if available), otherwise CLOSE
#   • Execution price selection prefers Open price, with fallback to Close
#
# Usage:
#   python backtester.py --csv data.csv --fast 20 --slow 50 --fee-bps 5 --start 2019-01-01 --end 2024-12-31
#
# CSV minimum: Date, Close   (optional: Open, High, Low, Volume)
# Date format: YYYY-MM-DD

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Dict
import argparse, csv, math, statistics, sys
from datetime import datetime, date

# ---------- Data Structures ----------

@dataclass
class Bar:
    dt: date
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None

@dataclass
class Trade:
    entry_dt: date
    exit_dt: Optional[date]
    entry_px: float
    exit_px: Optional[float]
    pnl: Optional[float]  # relative PnL (not absolute profit)


# ---------- Helpers ----------

def parse_date(s: str) -> date:
    """Parse a date string in common formats."""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {s!r}")

def read_csv_bars(path: str, start: Optional[str], end: Optional[str]) -> List[Bar]:
    """Read CSV with at least 'Date' and 'Close'. If present, also reads Open/High/Low/Volume."""
    start_d = parse_date(start) if start else None
    end_d = parse_date(end) if end else None
    out: List[Bar] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise ValueError("Empty or invalid CSV: no header found.")
        hdr = {k.lower().strip(): k for k in rdr.fieldnames}
        if "date" not in hdr or "close" not in hdr:
            raise ValueError("CSV must contain at least 'Date' and 'Close' columns.")
        for row in rdr:
            dt = parse_date(row[hdr["date"]])
            if start_d and dt < start_d:
                continue
            if end_d and dt > end_d:
                continue
            close = float(row[hdr["close"]])
            open_v = float(row[hdr["open"]]) if "open" in hdr and row[hdr["open"]] else None
            high_v = float(row[hdr["high"]]) if "high" in hdr and row[hdr["high"]] else None
            low_v  = float(row[hdr["low"] ]) if "low"  in hdr and row[hdr["low" ]] else None
            vol_v  = float(row[hdr["volume"]]) if "volume" in hdr and row[hdr["volume"]] else None
            out.append(Bar(dt=dt, close=close, open=open_v, high=high_v, low=low_v, volume=vol_v))
    out.sort(key=lambda b: b.dt)
    return out

def rolling_sma(values: List[float], window: int) -> List[Optional[float]]:
    """Compute a simple moving average in O(n). Returns None until enough samples."""
    if window <= 0:
        raise ValueError("window must be > 0")
    q = deque()
    s = 0.0
    out: List[Optional[float]] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) < window:
            out.append(None)
        else:
            out.append(s / window)
            if len(q) > window:
                s -= q.popleft()
    return out


# ---------- Strategy & Signals ----------

def generate_signals_shifted_for_next_open(prices: List[float], fast: int, slow: int) -> List[int]:
    """
    Generate SMA crossover signals.
    • Raw signal: 1 if SMA_fast > SMA_slow else 0.
    • Execution is shifted to T+1 → we shift the position forward by 1 day:
        pos_shifted[t] = pos_raw[t-1], pos_shifted[0] = 0
    """
    if fast >= slow:
        raise ValueError("Fast period must be less than slow period (e.g., 20/50).")
    sma_f = rolling_sma(prices, fast)
    sma_s = rolling_sma(prices, slow)

    pos_raw: List[int] = []
    for i in range(len(prices)):
        if sma_f[i] is None or sma_s[i] is None:
            pos_raw.append(0)
        else:
            pos_raw.append(1 if sma_f[i] > sma_s[i] else 0)

    # Shift signals forward for T+1 execution
    pos_shifted = [0] + pos_raw[:-1]
    return pos_shifted


# ---------- Backtest ----------

def backtest(
    bars: List[Bar],
    positions: List[int],
    start_cash: float = 10_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0
) -> Dict[str, object]:
    """
    Simple long/flat backtest:
      • Execution uses next day's OPEN if available, otherwise CLOSE.
      • All-in when long, all-out when flat.
      • Fees applied on notional each time you buy or sell.
    """
    if len(bars) != len(positions):
        raise ValueError("bars and positions length mismatch")

    def exec_price(bar: Bar) -> float:
        return bar.open if (bar.open is not None) else bar.close

    cash = start_cash
    shares = 0.0
    fee = (fee_bps + slippage_bps) / 10_000.0

    equity_curve: List[float] = []
    daily_ret: List[float] = []
    trades: List[Trade] = []

    prev_pos = 0
    entry_px = None
    entry_dt = None
    last_equity = start_cash

    for i, b in enumerate(bars):
        target = positions[i]

        # Rebalance if signal changes → execute on T+1 OPEN (already shifted)
        if target != prev_pos:
            px = exec_price(b)
            if target == 1 and prev_pos == 0:
                # Buy
                spendable = cash * (1.0 - fee)
                shares = spendable / px
                cash = 0.0
                entry_px = px
                entry_dt = b.dt
                trades.append(Trade(entry_dt=entry_dt, exit_dt=None, entry_px=entry_px, exit_px=None, pnl=None))
            elif target == 0 and prev_pos == 1:
                # Sell
                proceeds = shares * px * (1.0 - fee)
                cash = proceeds
                if trades and trades[-1].exit_dt is None:
                    trades[-1].exit_dt = b.dt
                    trades[-1].exit_px = px
                    trades[-1].pnl = (px - trades[-1].entry_px) / trades[-1].entry_px
                shares = 0.0
                entry_px = None
                entry_dt = None

        # Mark-to-market on close
        equity = cash + shares * b.close
        equity_curve.append(equity)
        if i > 0:
            r = (equity - last_equity) / last_equity if last_equity > 0 else 0.0
            daily_ret.append(r)
        last_equity = equity
        prev_pos = target

    # Liquidate at the end if still long
    if prev_pos == 1:
        final_bar = bars[-1]
        px = exec_price(final_bar)
        proceeds = shares * px * (1.0 - fee)
        cash = proceeds
        if trades and trades[-1].exit_dt is None:
            trades[-1].exit_dt = final_bar.dt
            trades[-1].exit_px = px
            trades[-1].pnl = (px - trades[-1].entry_px) / trades[-1].entry_px
        shares = 0.0
        last_equity = cash
        equity_curve[-1] = last_equity

    # Performance metrics
    start_dt, end_dt = bars[0].dt, bars[-1].dt
    total_years = max(1e-9, (end_dt - start_dt).days / 365.25)
    total_return = (equity_curve[-1] / equity_curve[0]) - 1.0 if equity_curve[0] > 0 else 0.0
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1.0 / total_years) - 1.0 if equity_curve[0] > 0 else 0.0

    if len(daily_ret) > 1 and statistics.pstdev(daily_ret) > 0:
        mean_d = statistics.mean(daily_ret)
        std_d = statistics.pstdev(daily_ret)
        sharpe = (mean_d / std_d) * math.sqrt(252.0)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = -1e18
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    max_drawdown = max_dd

    # Trade stats
    closed = [t for t in trades if t.pnl is not None]
    wins = [t for t in closed if t.pnl > 0]
    win_rate = (len(wins) / len(closed)) if closed else 0.0

    return {
        "equity_end": equity_curve[-1],
        "equity_start": equity_curve[0],
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "bars": len(bars),
        "trades": trades,
        "closed_trades": len(closed),
        "win_rate": win_rate,
        "equity_curve": equity_curve,
        "start": start_dt,
        "end": end_dt,
    }


# ---------- CLI ----------

def run(csv_path: str, fast: int, slow: int, fee_bps: float, slippage_bps: float,
        start: Optional[str], end: Optional[str]) -> None:
    bars = read_csv_bars(csv_path, start, end)
    closes = [b.close for b in bars]

    positions = generate_signals_shifted_for_next_open(closes, fast, slow)
    res = backtest(bars, positions, start_cash=10_000.0, fee_bps=fee_bps, slippage_bps=slippage_bps)

    # Print results
    print("\n=== Futures Strategy Backtest (SMA crossover; T+1 OPEN execution if available) ===")
    print(f"File: {csv_path}")
    print(f"Range: {res['start']} → {res['end']}  |  Bars: {res['bars']}")
    print(f"Parameters: fast={fast}, slow={slow}, fee={fee_bps} bps, slippage={slippage_bps} bps")
    print(f"Start equity: ${res['equity_start']:.2f}  →  End equity: ${res['equity_end']:.2f}")
    print(f"Total Return: {res['total_return']*100:.2f}%  |  CAGR: {res['cagr']*100:.2f}%")
    print(f"Sharpe (annualized): {res['sharpe']:.2f}  |  Max Drawdown: {res['max_drawdown']*100:.2f}%")
    print(f"Closed trades: {res['closed_trades']}  |  Win rate: {res['win_rate']*100:.1f}%")

    closed = [t for t in res["trades"] if t.pnl is not None]
    if closed:
        print("\nLast trades:")
        for t in closed[-5:]:
            pnl_pct = t.pnl * 100 if t.pnl is not None else float('nan')
            print(f"  {t.entry_dt} @ {t.entry_px:.2f}  ->  {t.exit_dt} @ {t.exit_px:.2f}   PnL: {pnl_pct:.2f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple Python backtester (SMA crossover, T+1 OPEN execution).")
    p.add_argument("--csv", required=True, help="CSV path (Date,Close[,Open,High,Low,Volume])")
    p.add_argument("--fast", type=int, default=20, help="Fast SMA window (e.g., 20)")
    p.add_argument("--slow", type=int, default=50, help="Slow SMA window (e.g., 50)")
    p.add_argument("--fee-bps", type=float, default=5.0, help="Per-side fee in basis points (5 = 0.05%)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Optional slippage in bps")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (optional)")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (optional)")
    args = p.parse_args()
    try:
        run(args.csv, args.fast, args.slow, args.fee_bps, args.slippage_bps, args.start, args.end)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
