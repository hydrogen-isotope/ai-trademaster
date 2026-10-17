"""
Alpaca Market Data API v2 client for the research scripts (official, documented API; read-only).

Keys: env APCA_API_KEY_ID / APCA_API_SECRET_KEY, or private/config/alpaca.json {"key_id", "secret_key", "feed"}
(same lookup as the dashboard). Keys are never printed or logged.

Feeds on the free Basic plan:
  sip  all US exchanges, historical only (the most recent 15 minutes are not available)  -> research downloads
  iex  one exchange, real time                                                           -> live loggers
Bars are stamped with their START time, like the Yahoo data these scripts used before, and filtered to the regular
session (09:30-16:00 ET) so the downstream datasets keep the same shape.
"""
import json, os, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BASE = "https://data.alpaca.markets"
NY = ZoneInfo("America/New_York")
SESSION_OPEN, SESSION_CLOSE = 9 * 60 + 30, 16 * 60


class AlpacaConfigError(RuntimeError):
    pass


def credentials(root="."):
    kid, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    feed = os.environ.get("APCA_FEED")
    path = os.path.join(root, "private", "config", "alpaca.json")
    if (not kid or not sec) and os.path.exists(path):
        cfg = json.load(open(path, encoding="utf-8"))
        kid, sec, feed = cfg.get("key_id"), cfg.get("secret_key"), feed or cfg.get("feed")
    if not kid or not sec:
        raise AlpacaConfigError("no Alpaca market-data key: set APCA_API_KEY_ID / APCA_API_SECRET_KEY or create "
                                "private/config/alpaca.json (see docs/MARKET-DATA.md)")
    return kid, sec, feed or "iex"


def parse_ts(s):
    """RFC-3339 (e.g. 2026-10-16T13:30:00Z, with or without fractional seconds) -> unix seconds."""
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, rest = s.split(".", 1)
        tz = rest[rest.find("+"):] if "+" in rest else rest[rest.find("-"):] if "-" in rest else ""
        s = head + tz
    return int(datetime.fromisoformat(s).timestamp())


def rfc3339(unix):
    return datetime.fromtimestamp(int(unix), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(path, params, creds, retries=5):
    kid, sec, _ = creds
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:          # rate limited: back off and retry
                time.sleep(2 + 3 * attempt)
                continue
            body = e.read()[:200].decode("utf-8", "replace")
            raise RuntimeError(f"Alpaca HTTP {e.code}: {body}") from None


def get_bars(symbols, timeframe, start, end=None, feed="sip", adjustment="split", creds=None, pause=0.32, log=None):
    """All bars for `symbols` between unix times start..end (end None = latest allowed).
    Returns {symbol: [(t, o, h, l, c, v), ...]} sorted by time. Paginates; stays under 200 requests/minute."""
    creds = creds or credentials()
    out = {s: [] for s in symbols}
    for k in range(0, len(symbols), 50):
        group = symbols[k:k + 50]
        params = {"symbols": ",".join(group), "timeframe": timeframe, "start": rfc3339(start), "limit": 10000,
                  "adjustment": adjustment, "feed": feed, "sort": "asc"}
        if end is not None:
            params["end"] = rfc3339(end)
        token, pages = None, 0
        while True:
            if token:
                params["page_token"] = token
            doc = _get("/v2/stocks/bars", params, creds)
            for s, bars in (doc.get("bars") or {}).items():
                out.setdefault(s, []).extend((parse_ts(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"]) for b in bars)
            token, pages = doc.get("next_page_token"), pages + 1
            time.sleep(pause)
            if not token:
                break
        if log:
            log(f"  {group[0]}..{group[-1]}: {pages} page(s)")
    for s in out:
        out[s] = sorted(set(out[s]))
    return out


def session_minute(t):
    d = datetime.fromtimestamp(t, NY)
    return d.hour * 60 + d.minute


def regular_session(rows):
    """Keep bars that start inside 09:30-16:00 ET."""
    return [r for r in rows if SESSION_OPEN <= session_minute(r[0]) < SESSION_CLOSE]


def session_hours(rows30):
    """Aggregate 30-minute bars into session-aligned hours (09:30, 10:30, ... 15:30), like Yahoo's 1h bars."""
    out, cur, key = [], None, None
    for t, o, h, l, c, v in regular_session(rows30):
        m = session_minute(t)
        k = (datetime.fromtimestamp(t, NY).date(), (m - SESSION_OPEN) // 60)
        start = t - ((m - SESSION_OPEN) % 60) * 60
        if k != key:
            if cur:
                out.append(tuple(cur))
            key, cur = k, [start, o, h, l, c, v]
        else:
            cur[2], cur[3], cur[4], cur[5] = max(cur[2], h), min(cur[3], l), c, cur[5] + v
    if cur:
        out.append(tuple(cur))
    return out


def chart_json(rows):
    """Yahoo-chart-compatible JSON, so dataset.build_panel() reads Alpaca data unchanged."""
    return {"chart": {"result": [{"timestamp": [r[0] for r in rows], "indicators": {"quote": [{
        "open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows],
        "close": [r[4] for r in rows], "volume": [r[5] for r in rows]}]}}],
        "source": "Alpaca Market Data API v2"}}
