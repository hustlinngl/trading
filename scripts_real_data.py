from __future__ import annotations
"""Real-market data utility: bounded Kraken cross-check + live snapshot."""
import argparse,json
from pathlib import Path
from ai_trading_lab.kraken_data import fetch_ohlcv,fetch_ticker,fingerprint_frame,save_provenance

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--symbol",default="BTC/USDT"); p.add_argument("--timeframe",default="4h"); p.add_argument("--limit",type=int,default=700); p.add_argument("--out",default="data/real_crosscheck"); args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df,provenance=fetch_ohlcv(args.symbol,args.timeframe,limit=args.limit); ticker=fetch_ticker(args.symbol)
    data_path=out/f"{args.symbol.replace('/','_')}_{args.timeframe}_kraken.csv"; df.to_csv(data_path,index=True)
    save_provenance(provenance,out/f"{args.symbol.replace('/','_')}_{args.timeframe}_kraken.provenance.json")
    snapshot={"ticker":ticker,"history":provenance.to_dict(),"history_fingerprint":fingerprint_frame(df),"data_path":str(data_path)}
    (out/"live_snapshot.json").write_text(json.dumps(snapshot,indent=2),encoding="utf-8"); print(json.dumps(snapshot,indent=2))
if __name__=="__main__": main()
