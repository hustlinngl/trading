from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data import exchange_client, fetch_ohlcv
from .streaming import BinancePublicStream, StreamEvent, UnsupportedStreamSymbolError
from .trade_window import _atr, timeframe_minutes

HISTORY_NAME = "live_signal_history.jsonl"
_EMPTY_STATE: Mapping[str, Any] = MappingProxyType({})


class LiveTracker:
    """Low-latency market state cache fed by asyncio stream consumers.

    Websocket producers never touch the dashboard/rendering path directly.
    Events are buffered in an asyncio.Queue and one merger task updates an
    O(1) per-symbol state index. get_current_state returns an immutable view
    without waiting on network or model work.
    """

    def __init__(
        self,
        symbols: tuple[str, ...] | list[str] = (),
        *,
        streams: tuple[str, ...] = ("bookTicker", "aggTrade"),
        max_queue: int = 20_000,
        max_symbols: int = 64,
        reconnect_seconds: float = 3.0,
        logger: logging.Logger | None = None,
    ):
        self._symbols: list[str] = []
        self._max_symbols = max(1, int(max_symbols))
        self._streams = tuple(streams)
        self._max_queue = max(256, int(max_queue))
        self._reconnect_seconds = max(1.0, float(reconnect_seconds))
        self._logger = logger or logging.getLogger(__name__)

        self._state_lock = threading.RLock()
        self._states: dict[str, Mapping[str, Any]] = {}
        self._latest: Mapping[str, Any] = _EMPTY_STATE

        self._lifecycle_lock = threading.RLock()
        self._stop_thread = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[StreamEvent] | None = None
        self._stop_async: asyncio.Event | None = None
        self._stream_tasks: dict[str, asyncio.Task[Any]] = {}

        self.add_symbols(symbols)

    @staticmethod
    def _normalize_symbols(symbols: Any) -> list[str]:
        if symbols is None:
            return []
        if isinstance(symbols, str):
            symbols = symbols.split(",")
        out: list[str] = []
        for symbol in symbols:
            value = str(symbol).strip().upper()
            if value and value not in out:
                out.append(value)
        return out

    @property
    def symbols(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(self._symbols)

    def add_symbols(
        self,
        symbols: tuple[str, ...] | list[str] | str,
    ) -> tuple[str, ...]:
        """Register symbols and subscribe them immediately when the loop is live."""
        requested = self._normalize_symbols(symbols)
        with self._state_lock:
            for symbol in requested:
                if symbol not in self._symbols and len(self._symbols) < self._max_symbols:
                    self._symbols.append(symbol)
            active = tuple(self._symbols)

        loop = self._loop
        if loop is not None and loop.is_running() and not self._stop_thread.is_set():
            new_symbols = [s for s in requested if s in active]
            loop.call_soon_threadsafe(self._schedule_new_streams, new_symbols)
        return active

    def start(
        self,
        symbols: tuple[str, ...] | list[str] | str | None = None,
    ) -> None:
        if symbols is not None:
            self.add_symbols(symbols)
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_thread.clear()
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="live-tracker-event-loop",
                daemon=True,
            )
            self._thread.start()
        self._ready.wait(timeout=2.0)

    def stop(self, timeout: float = 2.0) -> None:
        with self._lifecycle_lock:
            self._stop_thread.set()
            loop = self._loop
            stop_async = self._stop_async
            thread = self._thread
        if loop is not None and loop.is_running() and stop_async is not None:
            loop.call_soon_threadsafe(stop_async.set)
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, float(timeout)))
        with self._lifecycle_lock:
            if thread is self._thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def close(self) -> None:
        self.stop()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_loop())
        except Exception:
            self._logger.exception("live tracker event loop crashed")
        finally:
            self._loop = None
            self._queue = None
            self._stop_async = None
            self._ready.set()

    async def _run_loop(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._stop_async = asyncio.Event()
        self._ready.set()

        for symbol in self.symbols:
            self._schedule_new_streams([symbol])
        merger = asyncio.create_task(self._merge_loop(), name="live-tracker-merger")
        try:
            await self._stop_async.wait()
        finally:
            tasks = [merger, *self._stream_tasks.values()]
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    self._logger.error("live tracker task failure: %s", result)
            self._stream_tasks.clear()

    def _schedule_new_streams(self, symbols: list[str]) -> None:
        if self._stop_async is None:
            return
        for symbol in symbols:
            if symbol in self._stream_tasks:
                continue
            if symbol not in self.symbols:
                continue
            self._stream_tasks[symbol] = asyncio.create_task(
                self._stream_worker(symbol),
                name=f"live-stream-{symbol.replace('/', '_')}",
            )

    async def _enqueue(self, event: StreamEvent) -> None:
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        # High-frequency backpressure: retain newest data by evicting oldest.
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            self._logger.warning("live tracker queue remained full; dropping event")

    async def _stream_worker(self, symbol: str) -> None:
        backoff = self._reconnect_seconds
        while not self._stop_thread.is_set() and not (
            self._stop_async is not None and self._stop_async.is_set()
        ):
            try:
                collector = BinancePublicStream(
                    symbol=symbol,
                    streams=self._streams,
                )
                async for event in collector.events(reconnect_seconds=backoff):
                    if self._stop_thread.is_set():
                        break
                    await self._enqueue(event)
                    backoff = self._reconnect_seconds
            except asyncio.CancelledError:
                raise
            except UnsupportedStreamSymbolError as exc:
                self._logger.warning(
                    "live stream unsupported for %s: %s",
                    symbol,
                    exc,
                )
                return
            except Exception as exc:
                self._logger.warning(
                    "live stream failed for %s: %s",
                    symbol,
                    exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _merge_loop(self) -> None:
        queue = self._queue
        if queue is None:
            return
        while not self._stop_thread.is_set():
            try:
                event = await queue.get()
            except asyncio.CancelledError:
                raise
            try:
                self._merge_event(event)
            except Exception:
                self._logger.exception(
                    "failed to merge live event for %s",
                    getattr(event, "symbol", "unknown"),
                )
            finally:
                queue.task_done()

    @staticmethod
    def _event_price(
        event: StreamEvent,
        previous: Mapping[str, Any],
    ) -> float | None:
        payload = event.payload
        for key in ("p", "price", "lastPrice", "c", "last"):
            value = payload.get(key)
            if value is not None:
                try:
                    candidate = float(value)
                    if np.isfinite(candidate) and candidate > 0:
                        return candidate
                except (TypeError, ValueError):
                    continue
        previous_price = previous.get("price") if previous else None
        try:
            return (
                float(previous_price)
                if previous_price is not None and np.isfinite(float(previous_price))
                else None
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _valid_quote(bid: Any, ask: Any) -> tuple[float, float] | None:
        try:
            bid_value = float(bid)
            ask_value = float(ask)
        except (TypeError, ValueError):
            return None
        if not (
            np.isfinite(bid_value)
            and np.isfinite(ask_value)
            and bid_value > 0
            and ask_value > 0
            and ask_value >= bid_value
        ):
            return None
        return bid_value, ask_value

    def _merge_event(self, event: StreamEvent) -> None:
        symbol = str(event.symbol or "").strip().upper()
        if not symbol:
            return

        now = datetime.now(timezone.utc)
        event_time = (
            datetime.fromtimestamp(
                event.event_time_ms / 1000.0,
                tz=timezone.utc,
            )
            if int(event.event_time_ms or 0) > 0
            else now
        )
        transport_latency_ms = max(
            0.0,
            (now - event_time).total_seconds() * 1000.0,
        )

        with self._state_lock:
            previous = self._states.get(symbol, _EMPTY_STATE)
            merged: dict[str, Any] = dict(previous)
            merged.update(
                {
                    "symbol": symbol,
                    "stream": event.stream,
                    "event_type": event.event_type,
                    "event_time_ms": int(event.event_time_ms or 0),
                    "received_at": event.received_at,
                    "updated_at": now.isoformat(),
                    "latency_ms": float(transport_latency_ms),
                    "sequence": int(previous.get("sequence", 0)) + 1,
                }
            )

            payload = event.payload or {}
            event_kind = event.event_type.lower()
            event_ms = int(event.event_time_ms or 0)
            quote_time = int(previous.get("quote_event_time_ms", 0) or 0)
            trade_time = int(previous.get("trade_event_time_ms", 0) or 0)
            accepted_quote = event_kind == "bookticker" and (
                event_ms <= 0 or quote_time <= 0 or event_ms > quote_time
            )
            accepted_trade = event_kind in {"aggtrade", "trade"} and (
                event_ms <= 0 or trade_time <= 0 or event_ms > trade_time
            )

            if event_kind == "bookticker":
                if accepted_quote:
                    for target, source in (("bid", "b"), ("ask", "a")):
                        if source in payload:
                            try:
                                value = float(payload[source])
                                if np.isfinite(value) and value > 0:
                                    merged[target] = value
                            except (TypeError, ValueError):
                                pass
                    merged["quote_event_time_ms"] = event_ms or quote_time
                else:
                    merged["stale_event_count"] = int(previous.get("stale_event_count", 0) or 0) + 1

            event_price = self._event_price(event, previous)
            if accepted_trade and event_price is not None and np.isfinite(event_price):
                merged["last_trade_price"] = float(event_price)
                merged["trade_event_time_ms"] = event_ms or trade_time

            quote = self._valid_quote(merged.get("bid"), merged.get("ask"))
            if quote is not None:
                bid, ask = quote
                merged["mid"] = (bid + ask) / 2.0
                # The dashboard's canonical realtime price is the executable quote
                # midpoint, not the last trade, whenever a valid book exists.
                merged["price"] = merged["mid"]
            elif accepted_trade and event_price is not None and np.isfinite(event_price):
                merged["price"] = float(event_price)

            if accepted_trade:
                try:
                    if payload.get("q") is not None:
                        qty = float(payload["q"])
                        if np.isfinite(qty) and qty >= 0:
                            merged["last_qty"] = qty
                except (TypeError, ValueError):
                    pass
                merged["trade_side"] = (
                    "SELL" if bool(payload.get("m")) else "BUY"
                )

            view = MappingProxyType(merged)
            self._states[symbol] = view
            self._latest = view

    def submit_update(self, event: StreamEvent) -> bool:
        """Schedule an event without blocking; True means the hand-off was accepted."""
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        loop = self._loop
        if loop is None or not loop.is_running() or self._queue is None:
            self._merge_event(event)
            return True

        # The queue operation executes on the event-loop thread, so the caller
        # cannot know synchronously whether a later queue-full eviction occurs.
        # Returning True here means the hand-off was scheduled successfully.
        loop.call_soon_threadsafe(self._publish_threadsafe, event)
        return True

    def _publish_threadsafe(self, event: StreamEvent) -> None:
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass
        try:
            queue.get_nowait()
            queue.task_done()
            queue.put_nowait(event)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            self._logger.warning("live tracker queue remained full; dropping submitted event")

    def get_current_state(self, symbol: str | None = None) -> Mapping[str, Any]:
        """Return the latest immutable state view; average lookup is O(1)."""
        key = str(symbol or "").strip().upper()
        with self._state_lock:
            if key:
                return self._states.get(key, _EMPTY_STATE)
            return self._latest


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
            rows.append(obj) if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return rows


def _write_history(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def _resolve_result(
    df, signal, timeframe, pt_atr, sl_atr, fee_bps, slippage_bps, max_bars,
    impact_bps_per_sqrt=0.0, max_participation_pct=0.10,
    short_borrow_bps_per_bar=0.0,
):
    """Resolve a published signal with the same entry, barrier and time-stop semantics as training."""
    # New history records store an explicit data_timestamp. Older records only
    # have timestamp, which LiveAssessment sets from the last OHLCV candle index.
    stamp_value = signal.get("data_timestamp")
    if stamp_value is None or not str(stamp_value).strip():
        stamp_value = signal.get("timestamp")
    if stamp_value is None or not str(stamp_value).strip():
        return None
    try:
        stamp = pd.Timestamp(stamp_value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    stamp = (
        stamp.tz_localize("UTC")
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )
    idx = df.index.searchsorted(stamp)
    if idx >= len(df) or pd.Timestamp(df.index[idx]) != stamp:
        return None

    entry_i = idx + 1
    horizon = int(max_bars)
    if horizon < 1 or entry_i >= len(df):
        return None
    exit_open_i = entry_i + horizon

    opens = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    entry = float(opens[entry_i])
    atr = float(_atr(df).iloc[idx])
    if (
        not np.isfinite(atr)
        or atr <= 0
        or not np.isfinite(entry)
        or entry <= 0
    ):
        return None

    side = str(signal.get("signal", "")).upper()
    if side not in {"LONG", "SHORT"}:
        return None

    upper = entry + float(pt_atr) * atr
    lower = entry - float(sl_atr) * atr
    if not np.isfinite(upper) or not np.isfinite(lower) or lower <= 0 or upper <= lower:
        return None

    cost_parameters = np.asarray(
        [
            fee_bps,
            slippage_bps,
            impact_bps_per_sqrt,
            max_participation_pct,
            short_borrow_bps_per_bar,
        ],
        dtype=float,
    )
    if not np.isfinite(cost_parameters).all() or np.any(cost_parameters < 0.0):
        return None
    participation = float(np.clip(float(max_participation_pct), 0.0, 1.0))
    fee_bps = float(fee_bps)
    slippage_bps = float(slippage_bps)
    impact_bps_per_sqrt = float(impact_bps_per_sqrt)
    short_borrow_bps_per_bar = float(short_borrow_bps_per_bar)
    round_trip_cost_bps = 2.0 * (
        fee_bps + slippage_bps + impact_bps_per_sqrt * np.sqrt(participation)
    )
    min_bars = max(1, int(signal.get("_min_bars", 1) or 1))
    interval_minutes = timeframe_minutes(timeframe)

    def cost_bps_for(held: int) -> float:
        borrow = short_borrow_bps_per_bar * max(0, int(held)) if side == "SHORT" else 0.0
        return float(round_trip_cost_bps + borrow)

    def result_for(barrier: str, fill_price: float, held: int) -> dict:
        early = held < min_bars
        if early:
            outcome = "EARLY"
        elif barrier == "upper":
            outcome = "WIN" if side == "LONG" else "LOSS"
        else:
            outcome = "LOSS" if side == "LONG" else "WIN"
        gross = (
            fill_price / entry - 1.0
            if side == "LONG"
            else entry / fill_price - 1.0
        )
        estimated_cost_bps = cost_bps_for(held)
        return {
            "outcome": outcome,
            "realized_return": float(gross - estimated_cost_bps / 10000.0),
            "estimated_cost_bps": estimated_cost_bps,
            "holding_hours": float(held * interval_minutes / 60.0),
        }

    # Inspect only holding candles. The final time-stop candle is excluded, and
    # the open of each observed candle takes precedence over its intrabar range.
    # Elapsed intervals are measured from the executable entry open (held=0 there).
    for j in range(entry_i, min(exit_open_i, len(df))):
        bar_open = float(opens[j])
        if not np.isfinite(bar_open) or bar_open <= 0:
            return None
        held = j - entry_i

        if bar_open >= upper:
            return result_for("upper", bar_open, held)
        if bar_open <= lower:
            return result_for("lower", bar_open, held)

        high = float(highs[j])
        low = float(lows[j])
        if not np.isfinite(high) or not np.isfinite(low) or high < low:
            return None

        hit_up = high >= upper
        hit_down = low <= lower
        if hit_up and hit_down:
            # OHLC cannot establish which barrier was first; do not invent a win/loss.
            # Without intrabar ordering the exit fill and net return are unknown.
            return {
                "outcome": "AMBIGUOUS",
                "realized_return": None,
                "estimated_cost_bps": cost_bps_for(held),
                "holding_hours": float(held * interval_minutes / 60.0),
            }
        if hit_up:
            return result_for("upper", upper, held)
        if hit_down:
            return result_for("lower", lower, held)

    # An early barrier event can be resolved without waiting for the full horizon.
    # Otherwise, fail closed until the time-stop open is actually available.
    if exit_open_i >= len(df):
        return None
    exit_price = float(opens[exit_open_i])
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None

    gross = (
        exit_price / entry - 1.0
        if side == "LONG"
        else entry / exit_price - 1.0
    )
    estimated_cost_bps = cost_bps_for(horizon)
    return {
        "outcome": "TIMEOUT",
        "realized_return": float(gross - estimated_cost_bps / 10000.0),
        "estimated_cost_bps": estimated_cost_bps,
        "holding_hours": float(horizon * interval_minutes / 60.0),
    }


def update_live_signal_outcomes(
    settings,
    root=".",
    exchange=None,
    max_records=500,
):
    root = Path(root)
    path = root / "logs" / HISTORY_NAME
    rows = _load_history(path)
    open_rows = [
        r for r in rows
        if r.get("status") == "SIGNAL" and not r.get("outcome")
    ]
    if not open_rows:
        return {
            "updated": 0,
            "open": 0,
            "closed": sum(bool(r.get("outcome")) for r in rows),
        }
    ex = exchange or exchange_client(
        getattr(settings, "exchange", "binance"),
        sandbox=False,
    )
    updated = 0
    for rec in open_rows:
        try:
            df = fetch_ohlcv(
                ex,
                str(rec["symbol"]),
                settings.timeframe,
                max(300, int(getattr(settings, "live_lookback_bars", 600))),
            )
            df = df.sort_index()
            rec["_min_bars"] = max(
                1,
                int(
                    round(
                        float(getattr(settings, "trade_window_min_hours", 3))
                        * 60
                        / timeframe_minutes(settings.timeframe)
                    )
                ),
            )
            resolved = _resolve_result(
                df,
                rec,
                settings.timeframe,
                float(getattr(settings, "trade_window_pt_atr", 1.25)),
                float(getattr(settings, "trade_window_sl_atr", 0.90)),
                float(getattr(settings, "fee_bps", 7)),
                float(getattr(settings, "slippage_bps", 5)),
                int(
                    round(
                        float(getattr(settings, "trade_window_max_hours", 24))
                        * 60
                        / timeframe_minutes(settings.timeframe)
                    )
                ),
                impact_bps_per_sqrt=float(getattr(settings, "impact_bps_per_sqrt", 1.5)),
                max_participation_pct=float(getattr(settings, "max_participation_pct", 0.10)),
                short_borrow_bps_per_bar=float(getattr(settings, "short_borrow_bps_per_bar", 0.0)),
            )
            if resolved:
                rec.update(resolved)
                rec.pop("_min_bars", None)
                rec["resolved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
                stamp_value = rec.get("data_timestamp") or rec.get("timestamp")
                stamp = pd.Timestamp(stamp_value)
                stamp = (
                    stamp.tz_localize("UTC")
                    if stamp.tzinfo is None
                    else stamp.tz_convert("UTC")
                )
                ei = df.index.searchsorted(stamp) + 1
                if ei < len(df):
                    rec["entry_timestamp"] = str(df.index[ei])
                    rec["entry_price"] = float(df["open"].iloc[ei])
                updated += 1
        except Exception as exc:
            rec.pop("_min_bars", None)
            rec["tracking_error"] = str(exc)
    for rec in rows:
        rec.pop("_min_bars", None)
    rows = rows[-max_records:]
    _write_history(path, rows)
    return {
        "updated": updated,
        "open": sum(
            1 for r in rows
            if r.get("status") == "SIGNAL" and not r.get("outcome")
        ),
        "closed": sum(bool(r.get("outcome")) for r in rows),
    }
