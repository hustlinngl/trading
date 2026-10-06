# V6 real-data playbook

The goal is a broad historical dataset without pretending that one vendor covers every market or every historical instrument.

## Universe hierarchy

1. Binance Vision: broad crypto spot and USD-M futures archives.
2. FRED/ALFRED: macroeconomic time series and vintage-aware releases.
3. SEC EDGAR: issuer filings / XBRL event context.
4. CFTC COT: futures positioning history.
5. U.S. Treasury: Treasury curve / rates.
6. Alpha Vantage (optional): long daily global equity history where a key is available.

## Efficient research order

Do not download 15m history for thousands of instruments first. Download daily history broadly, audit coverage and liquidity, select an intraday research universe, then spend expensive compute on purged walk-forward + pristine holdout validation.

The universe runner is paper-only and does not place orders.
