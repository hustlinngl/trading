from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class SearchDocument:
    provider: str
    retrieved_at: str
    published_at: str | None
    title: str
    url: str
    text: str
    score: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request(method: str, url: str, *, timeout: float = 25.0, headers: dict[str, str] | None = None,
             json: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        timeout=float(timeout),
        headers=headers or {},
        json=json,
        params=params,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON response from {url}")
    return data


def _text_from_result(item: dict[str, Any]) -> str:
    for key in ("text", "content", "snippet", "raw_content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    highlights = item.get("highlights")
    if isinstance(highlights, list):
        return " ".join(str(x) for x in highlights if str(x).strip())
    return ""


class ExaClient:
    endpoint = "https://api.exa.ai/search"

    def __init__(self, api_key: str | None = None, timeout: float = 25.0):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.timeout = float(timeout)

    def search(self, query: str, num_results: int = 6, deep: bool = True) -> list[SearchDocument]:
        if not self.api_key:
            return []
        payload: dict[str, Any] = {
            "query": str(query),
            "numResults": max(1, min(100, int(num_results))),
            "contents": {"text": {"maxCharacters": 5000 if deep else 1800}},
        }
        data = _request(
            "POST",
            self.endpoint,
            timeout=self.timeout,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        out = []
        for item in data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            out.append(
                SearchDocument(
                    "exa",
                    _now(),
                    item.get("publishedDate") or item.get("published_date"),
                    str(item.get("title") or ""),
                    str(item.get("url") or ""),
                    _text_from_result(item),
                    float(item.get("score") or 0.0),
                )
            )
        return out


class TavilyClient:
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, timeout: float = 25.0):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = float(timeout)

    def search(self, query: str, num_results: int = 6, deep: bool = True) -> list[SearchDocument]:
        if not self.api_key:
            return []
        payload = {
            "api_key": self.api_key,
            "query": str(query),
            "max_results": max(1, min(20, int(num_results))),
            "search_depth": "advanced" if deep else "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        data = _request(
            "POST",
            self.endpoint,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        out = []
        for item in data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            out.append(
                SearchDocument(
                    "tavily",
                    _now(),
                    item.get("publishedDate") or item.get("published_date"),
                    str(item.get("title") or ""),
                    str(item.get("url") or ""),
                    _text_from_result(item),
                    float(item.get("score") or 0.0),
                )
            )
        return out


class FredClient:
    endpoint = "https://api.stlouisfed.org/fred/series/observations"
    public_endpoint = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, api_key: str | None = None, timeout: float = 25.0):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.timeout = float(timeout)

    def observations(self, series_id: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        sid = str(series_id).strip().upper()
        if not sid:
            return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])
        if self.api_key:
            params: dict[str, Any] = {
                "series_id": sid,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start,
                "observation_end": end,
                "sort_order": "asc",
            }
            data = _request("GET", self.endpoint, timeout=self.timeout, params={k: v for k, v in params.items() if v is not None})
            rows = data.get("observations", []) or []
            frame = pd.DataFrame(rows)
            if frame.empty:
                return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
            return frame.dropna(subset=["date", "value"]).reset_index(drop=True)
        params = {"id": sid}
        if start:
            params["cosd"] = str(start)
        if end:
            params["coed"] = str(end)
        response = requests.get(self.public_endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        from io import StringIO
        frame = pd.read_csv(StringIO(response.text))
        if frame.empty:
            return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])
        date_col = next((c for c in frame.columns if str(c).lower().startswith("observation")), frame.columns[0])
        value_col = sid if sid in frame.columns else frame.columns[-1]
        out = pd.DataFrame({"date": pd.to_datetime(frame[date_col], utc=True, errors="coerce"), "value": pd.to_numeric(frame[value_col], errors="coerce")})
        return out.dropna(subset=["date", "value"]).reset_index(drop=True)


class SecClient:
    endpoint = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, user_agent: str | None = None, timeout: float = 25.0):
        self.user_agent = user_agent or os.getenv(
            "SEC_USER_AGENT",
            "AdaptiveAITradingLab/0.9 research@example.com",
        )
        self.timeout = float(timeout)

    def submissions(self, cik: str) -> pd.DataFrame:
        digits = re.sub(r"\D", "", str(cik)).zfill(10)
        data = _request(
            "GET",
            self.endpoint.format(cik=digits),
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
        )
        recent = (data.get("filings") or {}).get("recent") or {}
        if not recent or not recent.get("form"):
            return pd.DataFrame()
        n = len(recent.get("form", []))
        columns = {}
        for key, values in recent.items():
            if isinstance(values, list) and len(values) == n:
                columns[key] = values
        frame = pd.DataFrame(columns)
        if "filingDate" in frame:
            frame["filingDate"] = pd.to_datetime(frame["filingDate"], utc=True, errors="coerce")
        return frame


_EVENT_TERMS = {
    "risk_on": ("approval", "inflow", "adoption", "beat", "upgrade", "growth", "easing", "dovish"),
    "risk_off": ("hack", "exploit", "ban", "lawsuit", "sanction", "tariff", "miss", "downgrade", "hawkish"),
    "rates": ("fed", "fomc", "interest rate", "yield", "treasury", "powell"),
    "inflation": ("inflation", "cpi", "pce", "price pressure"),
    "flows": ("etf", "flow", "inflow", "outflow", "fund"),
    "regulation": ("sec", "regulation", "approval", "lawsuit", "compliance"),
    "security": ("hack", "exploit", "breach", "vulnerability"),
}


def extract_event_terms(text: str) -> dict[str, float]:
    value = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", value)
    token_set = set(tokens)
    result: dict[str, float] = {}
    for family, terms in _EVENT_TERMS.items():
        hits = sum(1 for term in terms if " " not in term and term in token_set)
        hits += sum(1 for term in terms if " " in term and term in value)
        result[f"event_{family}_hits"] = float(hits)
    result["event_risk_balance"] = float(result["event_risk_on_hits"] - result["event_risk_off_hits"])
    result["event_pressure"] = float(min(1.0, sum(v for k, v in result.items() if k.endswith("_hits")) / 12.0))
    return result
