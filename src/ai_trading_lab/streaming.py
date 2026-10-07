from __future__ import annotations

import asyncio
import json
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

    Spot/margin use the spot combined stream. USDⓈ-M bookTicker and aggTrade
    use their distinct public/market websocket namespaces. COIN-M currently
    uses its combined market-stream endpoint. Dated futures are translated to
    Binance's underscore-delimited stream symbol form.
    """

    SPOT_BASE_URL = "wss://stream.binance.com:9443/stream"
    USDM_PUBLIC_BASE_URL = "wss://fstream.binance.com/public/stream"
    USDM_MARKET_BASE_URL = "wss://fstream.binance.com/market/stream"
    COINM_BASE_URL = "wss://dstream.binance.com/stream"

    _USDM_PUBLIC_STREAMS = frozenset({"bookticker"})
    _USDM_MARKET_STREAMS = frozenset(
        {"aggtrade", "kline", "markprice", "miniTicker".lower(), "ticker"}
    )

    def __init__(
        self,
        symbol="BTC/USDT",
        streams=("bookTicker", "aggTrade"),
        base_url: str | None = None,
    ):
        self.original_symbol = str(symbol).strip().upper()
        self.symbol = self._stream_symbol(self.original_symbol)
        self.streams = tuple(
            str(x).strip().lower() for x in streams if str(x).strip()
        )
        if not self.streams:
            raise ValueError("At least one Binance stream is required.")
        self.base_url = str(base_url).rstrip("/") if base_url else None
        if self.base_url is None:
            # Fail at construction for an actually unsupported unified symbol,
            # but keep URL routing deferred so mixed USDⓈ-M streams can split.
            self._base_url(self.original_symbol)

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
            # Binance Options have a different symbol/stream grammar.
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
            "usdm": cls.USDM_PUBLIC_BASE_URL,
            "coinm": cls.COINM_BASE_URL,
        }[kind]

    @classmethod
    def _usdm_base_url(cls, stream: str) -> str:
        if stream in cls._USDM_PUBLIC_STREAMS:
            return cls.USDM_PUBLIC_BASE_URL
        if stream in cls._USDM_MARKET_STREAMS:
            return cls.USDM_MARKET_BASE_URL
        # Unknown channels are treated conservatively as regular market data;
        # callers can still override base_url explicitly for custom channels.
        return cls.USDM_MARKET_BASE_URL

    def urls(self) -> tuple[str, ...]:
        """Return one URL per websocket namespace required by this subscription."""
        kind, _ = self._route(self.original_symbol)
        if self.base_url is not None:
            return (
                self.base_url
                + "?streams="
                + "/".join(f"{self.symbol}@{stream}" for stream in self.streams),
            )

        grouped: dict[str, list[str]] = {}
        for stream in self.streams:
            if kind == "usdm":
                base_url = self._usdm_base_url(stream)
            elif kind == "coinm":
                base_url = self.COINM_BASE_URL
            else:
                base_url = self.SPOT_BASE_URL
            grouped.setdefault(base_url, []).append(stream)

        return tuple(
            base_url + "?streams=" + "/".join(
                f"{self.symbol}@{stream}" for stream in streams
            )
            for base_url, streams in grouped.items()
        )

    def url(self):
        """Return the single URL when this subscription has one namespace."""
        urls = self.urls()
        if len(urls) != 1:
            raise ValueError(
                "This subscription spans multiple Binance websocket namespaces; "
                "use urls() or events()."
            )
        return urls[0]

    @staticmethod
    def _event_from_message(
        raw: str | bytes,
        *,
        original_symbol: str,
        fallback_stream: str,
    ) -> StreamEvent:
        msg = json.loads(raw)
        data = msg.get("data", msg)
        return StreamEvent(
            datetime.now(timezone.utc).isoformat(),
            str(msg.get("stream", fallback_stream)),
            int(data.get("E") or data.get("T") or 0),
            str(data.get("s", original_symbol.replace("/", ""))).upper(),
            str(data.get("e", data.get("stream", "unknown"))),
            data,
        )

    async def _reader_loop(
        self,
        websocket,
        url: str,
        queue: asyncio.Queue[StreamEvent],
        reconnect_seconds: float,
    ) -> None:
        backoff = max(1.0, float(reconnect_seconds))
        stream_hint = url.split("?streams=", 1)[-1].split("/", 1)[0]
        while True:
            try:
                async with websocket.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=10_000,
                ) as ws:
                    backoff = max(1.0, float(reconnect_seconds))
                    async for raw in ws:
                        event = self._event_from_message(
                            raw,
                            original_symbol=self.original_symbol,
                            fallback_stream=stream_hint,
                        )
                        try:
                            queue.put_nowait(event)
                        except asyncio.QueueFull:
                            try:
                                queue.get_nowait()
                                queue.task_done()
                            except asyncio.QueueEmpty:
                                pass
                            try:
                                queue.put_nowait(event)
                            except asyncio.QueueFull:
                                pass
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def events(self, reconnect_seconds=3.0) -> AsyncIterator[StreamEvent]:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets is not installed; install the project [full] extra"
            ) from exc

        queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=10_000)
        tasks = [
            asyncio.create_task(
                self._reader_loop(
                    websockets,
                    url,
                    queue,
                    reconnect_seconds,
                ),
                name=f"binance-stream-{index}",
            )
            for index, url in enumerate(self.urls())
        ]
        try:
            while True:
                yield await queue.get()
                queue.task_done()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


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
