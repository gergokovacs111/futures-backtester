# 📈 Futures Backtester – Python SMA Crossover Strategy

A Python-based backtesting engine for a moving-average crossover trading strategy.  
It supports **T+1 execution**, trade analytics, and key performance metrics on multi-year OHLCV data.

This project was created to demonstrate algorithmic trading logic, quantitative finance concepts, and Python programming skills.  
It can be used as a foundation for more advanced trading systems or machine learning-based strategies.

---

## 🚀 Project Overview

This tool simulates a simple moving average (SMA) crossover trading strategy using historical price data (OHLCV).  
It allows you to:

- Load historical market data from CSV  
- Generate buy/sell signals based on fast & slow SMA crossovers  
- Simulate trades using **T+1 execution** (entry/exit on the next day’s open)  
- Analyze strategy performance with detailed metrics  

---

## ⚙️ Features

✅ SMA Crossover strategy (fast vs slow moving average)  
✅ T+1 execution simulation (more realistic than same-day closes)  
✅ Automatic trade tracking and PnL calculation  
✅ Performance metrics:

- Total Return  
- CAGR (Annualized Growth Rate)  
- Sharpe Ratio  
- Max Drawdown  
- Win Rate & number of trades  

✅ Works with both demo and multi-year historical datasets  

---

## 📁 Files in this Repository

| File | Description |
|------|------------|
| `futures_backtester.py` | Main Python script containing the backtester |
| `data_ohlcv.csv` | Short sample dataset (~20 days) for quick testing |
| `ohlcv_5.csv` | Full 5-year dataset (2019–2024) for realistic results |
| `README.md` | Project documentation |

---

## ▶️ How to Run

### Run from terminal:
```bash
python futures_backtester.py --csv ohlcv_5.csv --fast 20 --slow 50 --fee-bps 5
