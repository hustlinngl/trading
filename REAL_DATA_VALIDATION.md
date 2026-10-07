# Real-data validation — 0.9.11

## What was actually checked

This release was checked against public market data retrieved from Kraken's public REST API and against a current public ticker snapshot.

### Historical window

- Instrument: BTC/USD (Kraken pair `XBTUSD`)
- Timeframe: 4h
- Window: 2026-06-09 through 2026-10-06 UTC (four 30-day pulls)
- Candles collected: 720 completed candles
- First observed close: 62,851.4
- Last observed close: 85,541.8
- Market move over the window: +36.10%

This is **market performance, not strategy performance**. The project's model was not declared validated from this bounded slice alone.

## Source semantics

Kraken documents `/0/public/OHLC` as an unauthenticated public endpoint with a maximum of 720 candles per call and a final current/uncommitted candle. The adapter removes the uncommitted candle by default and stores provenance plus a content fingerprint.

Because of that hard cap, Kraken is used here as a cross-exchange/recent-history validator. Long-history training remains on the versioned Binance Vision path already present in the project.

## Live market snapshot

The public Kraken ticker endpoints were queried for BTC/USD, ETH/USD and SOL/USD during the research run. The resulting snapshot is intended for live-market health checks and cross-exchange monitoring, not for fitting the historical model.

## Reproducible commands

```bash
python scripts_real_data.py --symbol BTC/USDT --timeframe 4h --limit 700
python run_full_research.py --symbol BTC/USDT --timeframe 15m --start 2018-01-01 --end 2026-10-07
python -m ai_trading_lab.main real-history --symbol BTC/USDT --timeframe 4h --limit 700
python -m ai_trading_lab.main real-ticker --symbols BTC/USDT,ETH/USDT,SOL/USDT
```

## Research conclusion

The real market window confirms that the data adapters are pointed at real, changing market conditions and that the current period contains materially different volatility/trend regimes. It does **not** establish that the current strategy has an edge.

The next production-research gate is a multi-year, versioned Binance Vision dataset across several assets, followed by purged walk-forward, untouched holdout, cost stress, placebo controls and promotion governance.
