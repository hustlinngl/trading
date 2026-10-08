"""Build deterministic real-market manifests for broad discovery and intraday research.

The broad manifest describes the full discovered universe; the research manifest is
a deterministic priority slice used by downstream research jobs. Runtime settings
provide the defaults so CLI and config cannot silently drift.
"""
import argparse
import json
from pathlib import Path

from ai_trading_lab.config import load_settings
from ai_trading_lab.real_universe import (
    build_seed_universe,
    discover_binance_symbols,
    expand_binance_universe,
    write_manifest,
    summarize_manifest,
)


def main():
    settings = load_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--discover-binance", action="store_true")
    p.add_argument("--start", default="2017-08-01")
    p.add_argument("--end", default="2026-10-06")
    p.add_argument("--max-binance-symbols", type=int, default=None)
    p.add_argument("--research-top", type=int, default=None)
    p.add_argument("--manifest", default=None)
    args = p.parse_args()

    if args.manifest:
        data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        print(json.dumps({"manifest": args.manifest, "items": len(data)}, indent=2))
        return

    max_symbols = max(
        1,
        int(
            args.max_binance_symbols
            if args.max_binance_symbols is not None
            else getattr(settings, "broad_crypto_symbol_cap", 500)
        ),
    )
    research_top = max(
        1,
        int(
            args.research_top
            if args.research_top is not None
            else getattr(settings, "intraday_research_top_n", 30)
        ),
    )

    specs = build_seed_universe(args.start, args.end)
    discovered_specs = []
    if args.discover_binance:
        symbols = discover_binance_symbols()[:max_symbols]
        discovered_specs = expand_binance_universe(
            symbols, args.start, args.end, timeframe=settings.broad_timeframe
        )
        specs = discovered_specs + specs

    # Keep the broad manifest lossless. The research slice is a separate artifact,
    # preventing an intraday cap from accidentally shrinking the discovered universe.
    broad_path = Path("data/real_universe/manifest.json")
    write_manifest(specs, broad_path)

    ranked = sorted(
        discovered_specs,
        key=lambda item: (-int(item.priority), item.symbol),
    )
    research_specs = ranked[:research_top]
    if not research_specs:
        research_specs = list(specs[:research_top])
    research_path = Path("data/real_universe/intraday_research_manifest.json")
    write_manifest(research_specs, research_path)

    print(
        json.dumps(
            {
                "manifest": str(broad_path),
                "research_manifest": str(research_path),
                "max_binance_symbols": max_symbols,
                "research_top": research_top,
                "summary": summarize_manifest(specs),
                "research_summary": summarize_manifest(research_specs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
