from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator


class UnsupportedStreamSymbolError(ValueError):
    """The unified symbol does not map to a supported Binance public stream."""


@dataclass
class StreamEvent:
    received_at: str
    stream: str
    event_time_ms: int
    symbol: str
    event_type: str
    payload: dict[str, Any]

    def row(self):
        return asdict(self)


class BinancePublicStream:
    """Binance public market-data adapter with unified-symbol routing.

    Spot/margin symbols use the spot combined-stream host. USDⓈ-M contracts use
    the USDⓈ-M futures public stream host, while COIN-M contracts use the
    COIN-M futures stream host. Dated futures are translated to Binance's
    underscore-delimited stream symbol form.
    """

    SPOT_BASE_URL = "wss://stream.binance.com:9443/stream"
    USDM_BASE_URL = "wss://fstream.binance.com/public/stream"
    COINM_BASE_URL = "wss://dstream.binance.com/stream"

    def __init__(
        self,
        symbol="BTC/USDT",
        streams=("bookTicker", "aggTrade"),
        base_url: str | None = None,
    ):
        self.original_symbol = str(symbol).strip().upper()
        self.symbol = self._stream_symbol(self.original_symbol)
        self.streams = tuple(str(x).strip().lower() for x in streams if str(x).strip())
        if not self.streams:
            raise ValueError("At least one Binance stream is required.")
        self.base_url = str(base_url).rstrip("/") if base_url else self._base_url(
            self.original_symbol
        )

    @classmethod
    def _route(cls, symbol: str) -> tuple[str, str]:
        raw = str(symbol).strip().upper()
        if not raw:
            raise UnsupportedStreamSymbolError("Empty Binance symbol.")

        parts = raw.split(":", 1)
        base = parts[0].replace("/", "")
        if not base:
            raise UnsupportedStreamSymbolError(f"Invalid Binance symbol: {symbol!r}")

        if len(parts) == 1:
            # Unified spot and margin symbols map to the spot public stream.
            return "spot", base.lower()

        contract = parts[1]
        suffix = contract.split("-")
        settlement = suffix[0]
        if len(suffix) >= 3:
            # Binance Options have a different public websocket namespace and
            # stream grammar; do not silently send them to a futures endpoint.
            raise UnsupportedStreamSymbolError(
                f"Binance option streams are not supported by this adapter: {symbol}"
            )

        kind = "usdm" if settlement in {"USDT", "USDC"} else "coinm"
        stream_symbol = base
        if len(suffix) == 2 and suffix[1]:
            stream_symbol += "_" + suffix[1]
        return kind, stream_symbol.lower()

    @classmethod
    def _stream_symbol(cls, symbol: str) -> str:
        return cls._route(symbol)[1]

    @classmethod
    def _base_url(cls, symbol: str) -> str:
        kind, _ = cls._route(symbol)
        return {
            "spot": cls.SPOT_BASE_URL,
            "usdm": cls.USDM_BASE_URL,
            "coinm": cls.COINM_BASE_URL,
        }[kind]

    def url(self):
        return self.base_url + "?streams=" + "/".join(
            f"{self.symbol}@{stream}" for stream in self.streams
        )

    async def events(self, reconnect_seconds=3.0) -> AsyncIterator[StreamEvent]:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets is not installed; install the stream extra"
            ) from exc

        while True:
            try:
                async with websockets.connect(
                    self.url(),
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=10_000,
                ) as ws:
                    async for raw in ws:
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        yield StreamEvent(
                            datetime.now(timezone.utc).isoformat(),
                            str(msg.get("stream", "")),
                            int(data.get("E") or data.get("T") or 0),
                            str(
                                data.get(
                                    "s",
                                    self.original_symbol.replace("/", ""),
                                )
                            ).upper(),
                            str(
                                data.get(
                                    "e",
                                    data.get("stream", "unknown"),
                                )
                            ),
                            data,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(max(1.0, reconnect_seconds))


async def collect_to_jsonl(collector, path, max_events=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("a", encoding="utf-8") as fh:
        async for event in collector.events():
            fh.write(json.dumps(event.row(), default=str) + "\n")
            count += 1
            if max_events is not None and count >= max_events:
                break
    return count
