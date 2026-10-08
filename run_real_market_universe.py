"""Compatibility entrypoint for real-universe discovery/research.
The canonical long-running CI path is .github/workflows/intensive-real-research.yml.
"""
import argparse, json
from pathlib import Path
from ai_trading_lab.real_universe import build_seed_universe, discover_binance_symbols, expand_binance_universe, write_manifest, summarize_manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--discover-binance',action='store_true'); p.add_argument('--start',default='2017-08-01'); p.add_argument('--end',default='2026-10-06'); p.add_argument('--max-binance-symbols',type=int,default=500); p.add_argument('--research-top',type=int,default=30); p.add_argument('--manifest',default=None); p.add_argument('--workers',type=int,default=8); args=p.parse_args()
    if args.manifest:
        data=json.loads(Path(args.manifest).read_text(encoding='utf-8')); print(json.dumps({'manifest':args.manifest,'items':len(data)},indent=2)); return
    specs=build_seed_universe(args.start,args.end)
    if args.discover_binance:
        symbols=discover_binance_symbols()[:max(1,args.max_binance_symbols)]; specs=expand_binance_universe(symbols,args.start,args.end)+specs
    out=Path('data/real_universe/manifest.json'); write_manifest(specs,out); print(json.dumps({'manifest':str(out),'summary':summarize_manifest(specs)},indent=2))
if __name__=='__main__': main()
