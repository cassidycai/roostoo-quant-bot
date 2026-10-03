"""Signed REST access to the Roostoo mock exchange.

Three things in here matter more than the rest:

1. **Signature exactness.** ``MSG-SIGNATURE`` is HMAC-SHA256 over the
   parameters sorted by key and joined as ``k=v&k=v``. Only the parameters the
   endpoint documents may be included -- the server rebuilds the string from
   the parameters *it* recognises, so one extra field invalidates the whole
   request.
2. **``Success: false`` inside HTTP 200.** Application errors arrive with a
   normal status code, so a client that only inspects the status code will keep
   trading straight through a rejection.
3. **Order placement is not idempotent.** A transport failure during
   ``place_order`` leaves us genuinely unsure whether the order exists. We never
   blind-retry it; we surface ``status="UNKNOWN"`` and let the engine reconcile
   via ``query_order``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Optional, Protocol

from .config import Config
from .errors import APIError, RateLimitError, TransportError
from .models import (
    ExchangeInfo,
    OrderResult,
    ShortPosition,
    Ticker,
    WalletBalance,
)

log = logging.getLogger(__name__)

# Endpoints, kept in one place so the paths cannot drift between call sites.
PATH_SERVER_TIME = "/v3/serverTime"
PATH_EXCHANGE_INFO = "/v3/exchangeInfo"
PATH_TICKER = "/v3/ticker"
PATH_BALANCE = "/v3/balance"
PATH_PENDING_COUNT = "/v3/pending_count"
PATH_PLACE_ORDER = "/v3/place_order"
PATH_QUERY_ORDER = "/v3/query_order"
PATH_CANCEL_ORDER = "/v3/cancel_order"
PATH_SHORT_OPEN = "/v6/short_open"
PATH_SHORT_CLOSE = "/v6/short_close"
PATH_SHORT_POSITIONS = "/v6/short_positions"

# Endpoints that are documented to answer `Success: false` for a normal,
# empty result rather than a genuine failure.
_EMPTY_IS_NORMAL = {
    PATH_PENDING_COUNT: "no pending order",
    PATH_QUERY_ORDER: "no order matched",
}
# Public endpoints whose successful responses do not contain a `Success` flag.
_NO_SUCCESS_FLAG = {
    PATH_SERVER_TIME,
    PATH_EXCHANGE_INFO,
}

# ---------------------------------------------------------------------------
# Pure signing helpers (unit-tested against the published vector)
# ---------------------------------------------------------------------------


def canonical_params(params: dict[str, Any]) -> str:
    """Build the string that is both signed and sent.

    Keys sorted lexicographically, values stringified, joined by ``&``.
    """
    return "&".join(f"{key}={params[key]}" for key in sorted(params))


#: Values the venue is accepted to send for a successful call. Anything else --
#: including a missing flag -- is a failure. `is False` was too narrow: a payload
#: carrying `0`, `"false"` or no `Success` key at all was read as a success, so a
#: rejected order reached the engine as a completed fill.
_SUCCESS_VALUES = (True, 1, "1", "true", "TRUE", "True", "yes")


def _is_success(payload: dict[str, Any]) -> bool:
    """Has the venue affirmatively reported success?"""
    return payload.get("Success") in _SUCCESS_VALUES


def is_success(payload: dict[str, Any]) -> bool:
    """Public form of :func:`_is_success`, for callers checking raw payloads.

    The v6 short endpoints return their own dicts rather than an ``OrderResult``,
    so the engine needs the same rule here instead of testing truthiness itself.
    """
    return _is_success(payload)


def sign_payload(canonical: str, secret_key: str) -> str:
    """Hex HMAC-SHA256 of ``canonical`` keyed by the account's secret."""
    return hmac.new(secret_key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Transport seam (swappable so tests never touch the network)
# ---------------------------------------------------------------------------


class Transport(Protocol):  # pragma: no cover - structural typing only
    def send(
        self, method: str, url: str, body: Optional[bytes], headers: dict[str, str], timeout: float
    ) -> tuple[int, str]:
        """Return ``(http_status, response_text)``."""


class UrllibTransport:
    """Standard-library HTTP transport: no third-party dependency to audit."""

    def send(
        self, method: str, url: str, body: Optional[bytes], headers: dict[str, str], timeout: float
    ) -> tuple[int, str]:
        request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a body
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return exc.code, detail
        except urllib.error.URLError as exc:
            raise TransportError(f"{method} {url} failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TransportError(f"{method} {url} timed out after {timeout}s") from exc
        except OSError as exc:
            raise TransportError(f"{method} {url} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Live client
# ---------------------------------------------------------------------------


class RoostooClient:
    """Signed client for ``https://mock-api.roostoo.com``."""

    def __init__(
        self,
        config: Config,
        transport: Optional[Transport] = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not config.api_key or not config.secret_key:
            raise ValueError("RoostooClient requires ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY")
        self.cfg = config
        self.base_url = config.base_url
        self._transport = transport or UrllibTransport()
        self._clock = clock
        self._sleep = sleeper
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        self._server_offset_ms = 0
        self._consecutive_failures = 0
        self._request_count = 0

    # -- time -----------------------------------------------------------
    def timestamp_ms(self) -> str:
        """Server-corrected millisecond timestamp.

        Roostoo rejects any signed request whose timestamp is more than 60s away
        from server time, so we measure the offset once and keep applying it
        instead of trusting the host clock.
        """
        return str(int(self._clock() * 1000) + self._server_offset_ms)

    def sync_time(self) -> int:
        """Measure and store the offset between this host and the exchange."""
        local_before = self._clock() * 1000
        payload = self._call("GET", PATH_SERVER_TIME, {"timestamp": str(int(local_before))}, signed=False, retries=2)
        server_ms = int(payload.get("ServerTime", 0) or 0)
        local_after = self._clock() * 1000
        if server_ms:
            midpoint = (local_before + local_after) / 2.0
            self._server_offset_ms = int(server_ms - midpoint)
            log.info("clock synced: offset=%+dms", self._server_offset_ms)
        return self._server_offset_ms

    @property
    def server_offset_ms(self) -> int:
        return self._server_offset_ms

    def sign_headers(self, params: dict[str, Any], canonical: Optional[str] = None) -> dict[str, str]:
        """The auth headers for ``params``.

        Public rather than inlined in ``_call`` so that auxiliary signed
        endpoints can reuse the exact same signing path. ``RoostooDepthProvider``
        used to probe for a private ``_sign_headers`` that never existed, so it
        silently sent *unsigned* requests to a credential-requiring path.

        ``canonical`` lets the caller pass the very string it is about to put on
        the wire, which removes any chance of signing one thing and sending
        another.
        """
        body = canonical if canonical is not None else canonical_params(params)
        return {
            "RST-API-KEY": self.cfg.api_key,
            "MSG-SIGNATURE": sign_payload(body, self.cfg.secret_key),
        }

    # -- plumbing -------------------------------------------------------
    def _throttle(self) -> None:
        """Enforce a floor between requests.

        The rulebook bans high-frequency trading and warns that excessive
        requests earn failed responses. A local floor is cheaper than being
        throttled by the exchange mid-session.
        """
        with self._lock:
            elapsed = self._clock() - self._last_request_at
            wait = self.cfg.min_request_interval_sec - elapsed
            if wait > 0:
                self._sleep(wait)
            self._last_request_at = self._clock()

    def _prepare(
        self, method: str, path: str, params: dict[str, Any], signed: bool
    ) -> tuple[str, Optional[bytes], dict[str, str]]:
        """Build the URL, body and headers for one attempt.

        Called inside the retry loop rather than once outside it. The signed
        timestamp is part of what is signed, and the venue rejects anything more
        than 60 s from its own clock, so a retry that re-sent the first attempt's
        signature would be sending a stale credential -- the backoff, the throttle
        floor and the original round trip all add up between attempts.
        """
        params = dict(params)
        headers = {"Accept": "application/json", "User-Agent": "roostoo-quant-bot/0.1"}
        if signed:
            params["timestamp"] = self.timestamp_ms()
            body_str: Optional[str] = canonical_params(params)
            headers.update(self.sign_headers(params, canonical=body_str))
        else:
            body_str = None

        if method == "GET":
            return f"{self.base_url}{path}?{canonical_params(params)}", None, headers
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return f"{self.base_url}{path}", (body_str or canonical_params(params)).encode("utf-8"), headers

    def _call(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
        *,
        signed: bool,
        retries: Optional[int] = None,
    ) -> dict[str, Any]:
        # `MAX_RETRIES` used to be configuration in name only: every call site
        # hard-coded its own budget, so an operator raising it changed nothing.
        # Non-idempotent endpoints still pass an explicit 0.
        if retries is None:
            retries = max(0, int(self.cfg.max_retries))
        attempt = 0
        while True:
            url, body, headers = self._prepare(method, path, params, signed)
            self._throttle()
            self._request_count += 1
            try:
                status, text = self._transport.send(method, url, body, headers, self.cfg.request_timeout_sec)
            except TransportError:
                if attempt < retries:
                    attempt += 1
                    self._backoff(attempt)
                    continue
                self._consecutive_failures += 1
                raise

            retryable = status >= 500 or status in (408, 425, 429)
            if retryable:
                if attempt < retries:
                    attempt += 1
                    # 429 is the rate limiter, not a malformed request: the venue
                    # warns that bursts earn failed responses, so back off and try
                    # again instead of losing the whole cycle. (`Retry-After` is a
                    # response *header*, which the Transport seam does not surface;
                    # the exponential backoff below stands in for it.)
                    log.warning("%s %s -> HTTP %s, retry %d/%d", method, path, status, attempt, retries)
                    self._backoff(attempt)
                    continue
                self._consecutive_failures += 1
                raise TransportError(f"{method} {path} -> HTTP {status}: {text[:300]}")

            if status >= 400:
                # Any other 4xx is our fault (bad signature, bad params), which
                # retrying cannot fix.
                raise TransportError(f"{method} {path} -> HTTP {status}: {text[:300]}")

            self._consecutive_failures = 0
            return self._parse(path, text)

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(0.5 * (2 ** (attempt - 1)), 8.0) * (0.7 + 0.6 * random.random()))

    @staticmethod
    def _parse(path: str, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            raise TransportError(f"{path} returned an empty body")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TransportError(f"{path} returned non-JSON payload: {text[:300]!r}") from exc
        if not isinstance(payload, dict):
            raise TransportError(f"{path} returned unexpected payload type {type(payload).__name__}")
        if path not in _NO_SUCCESS_FLAG and not _is_success(payload):
            err_msg = str(payload.get("ErrMsg", "unknown error") or "unknown error")
            if "Success" not in payload:
                err_msg = "response carried no Success flag"
         
            empty_token = _EMPTY_IS_NORMAL.get(path)
            if empty_token and empty_token in err_msg.lower():
                # Documented empty state, not an error.
                return payload
         
          raise APIError(err_msg, path, payload)
              return payload

    @property
    def request_count(self) -> int:
        return self._request_count

    # -- public (unsigned) ---------------------------------------------
    def exchange_info(self, retries: Optional[int] = None) -> ExchangeInfo:
        return ExchangeInfo.from_api(self._call("GET", PATH_EXCHANGE_INFO, {}, signed=False, retries=retries))

    def ticker(self, pair: Optional[str] = None, retries: Optional[int] = None) -> dict[str, Ticker]:
        params: dict[str, Any] = {}
        if pair:
            params["pair"] = pair
        payload = self._call("GET", PATH_TICKER, params, signed=False, retries=retries)
        server_ms = int(payload.get("ServerTime", 0) or 0)
        data = payload.get("Data") or {}
        return {p: Ticker.from_api(p, row, server_ms) for p, row in data.items()}

    # -- private (signed) ----------------------------------------------
    def balance(self) -> dict[str, WalletBalance]:
        payload = self._call("GET", PATH_BALANCE, {}, signed=True)
        wallet = payload.get("Wallet") or {}
        return {
            asset: WalletBalance(asset=asset, free=float(v.get("Free", 0.0) or 0.0), locked=float(v.get("Lock", 0.0) or 0.0))
            for asset, v in wallet.items()
        }

    def pending_count(self) -> tuple[int, dict[str, int]]:
        payload = self._call("GET", PATH_PENDING_COUNT, {}, signed=True)
        total = int(payload.get("TotalPending", 0) or 0)
        pairs = {str(k): int(v) for k, v in (payload.get("OrderPairs") or {}).items()}
        return total, pairs

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: str | float,
        order_type: str = "MARKET",
        price: Optional[str | float] = None,
    ) -> OrderResult:
        """Place one order. Never auto-retried -- see the module docstring."""
        order_type = order_type.upper()
        side = side.upper()
        if order_type == "LIMIT" and price is None:
            raise ValueError("LIMIT orders require a price")
        params: dict[str, Any] = {"pair": pair, "side": side, "type": order_type, "quantity": str(quantity)}
        if order_type == "LIMIT":
            params["price"] = str(price)
        try:
            payload = self._call("POST", PATH_PLACE_ORDER, params, signed=True, retries=0)
        except TransportError as exc:
            log.error("place_order transport failure (order state UNKNOWN): %s", exc)
            return OrderResult(
                pair=pair,
                side=side,
                order_type=order_type,
                quantity=float(quantity),
                price=float(price or 0.0),
                status="UNKNOWN",
                err_msg=str(exc),
            )
        return OrderResult.from_api(pair, side, order_type, float(quantity), payload)

    def query_orders(
        self,
        order_id: Optional[int | str] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if order_id is not None and (pair or pending_only is not None or offset is not None or limit is not None):
            raise ValueError("order_id is exclusive: no other optional parameter may be sent with it")
        params: dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = str(order_id)
        else:
            if pair:
                params["pair"] = pair
                if pending_only is not None:
                    params["pending_only"] = "TRUE" if pending_only else "FALSE"
            if offset is not None:
                params["offset"] = str(offset)
            if limit is not None:
                params["limit"] = str(limit)
        payload = self._call("POST", PATH_QUERY_ORDER, params, signed=True)
        return list(payload.get("OrderMatched") or [])

    def cancel_order(self, order_id: Optional[int | str] = None, pair: Optional[str] = None) -> list[int]:
        if order_id is not None and pair:
            raise ValueError("only one of order_id and pair may be sent")
        params: dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = str(order_id)
        elif pair:
            params["pair"] = pair
        payload = self._call("POST", PATH_CANCEL_ORDER, params, signed=True)
        return [int(x) for x in (payload.get("CanceledList") or [])]

    # -- short side (v6) ------------------------------------------------
    def short_open(self, pair: str, collateral: str | float, price: Optional[str | float] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"pair": pair, "collateral": str(collateral)}
        if price is not None:
            params["order_type"] = "LIMIT"
            params["price"] = str(price)
        return self._call("POST", PATH_SHORT_OPEN, params, signed=True, retries=0)

    def short_close(
        self, pair: str, close_qty: Optional[str | float] = None, close_pct: Optional[str | float] = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"pair": pair}
        if close_qty is not None:
            params["close_qty"] = str(close_qty)
        elif close_pct is not None:
            params["close_pct"] = str(close_pct)
        return self._call("POST", PATH_SHORT_CLOSE, params, signed=True, retries=0)

    def short_positions(self) -> list[ShortPosition]:
        payload = self._call("GET", PATH_SHORT_POSITIONS, {}, signed=True)
        return [ShortPosition.from_api(row) for row in (payload.get("Positions") or [])]


# ---------------------------------------------------------------------------
# Interface shared by the live client and the simulator
# ---------------------------------------------------------------------------


class MarketClient(Protocol):  # pragma: no cover - structural typing only
    """The surface the engine depends on, so the simulator can stand in."""

    def sync_time(self) -> int: ...
    def exchange_info(self, retries: int = 1) -> ExchangeInfo: ...
    def ticker(self, pair: Optional[str] = None, retries: int = 1) -> dict[str, Ticker]: ...
    def balance(self) -> dict[str, WalletBalance]: ...
    def pending_count(self) -> tuple[int, dict[str, int]]: ...
    def place_order(
        self, pair: str, side: str, quantity: Any, order_type: str = "MARKET", price: Any = None
    ) -> OrderResult: ...
    def query_orders(self, **kwargs: Any) -> list[dict[str, Any]]: ...
    def cancel_order(self, order_id: Any = None, pair: Optional[str] = None) -> list[int]: ...
    def short_positions(self) -> list[ShortPosition]: ...


def build_client(config: Config, transport: Optional[Transport] = None) -> Any:
    """Factory: a simulated client when ``ROOSTOO_MOCK=1``, else the live one."""
    if config.mock:
        from .simulator import MockRoostooClient

        return MockRoostooClient(config)
    return RoostooClient(config, transport=transport)


def normalise_pairs(pairs: Iterable[str]) -> list[str]:
    out = []
    for pair in pairs:
        p = pair.strip().upper()
        out.append(p if "/" in p else f"{p}/USD")
    return out
