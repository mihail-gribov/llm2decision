"""JSON over HTTP with the standard library: one kept-alive connection per thread and host, proxies
from the environment (`HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY`), no dependencies.

Network failures raise `NetworkError` (the transport retries them); an HTTP status is returned as is.
A kept-alive connection the server has closed in the meantime is reopened once, silently.
"""
from __future__ import annotations

import base64
import http.client
import json
import ssl
import threading
import urllib.request
from dataclasses import dataclass
from email.message import Message
from typing import Any
from urllib.parse import unquote, urlsplit


class NetworkError(Exception):
    """The request did not get an HTTP response: refused, reset, timed out, name not resolved."""


@dataclass
class Response:
    status: int
    headers: Message
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


def _proxy_for(scheme: str, host: str):
    """The proxy URL the environment names for this scheme and host, unless `NO_PROXY` exempts the host."""
    proxies = urllib.request.getproxies()
    url = proxies.get(scheme)
    if not url or urllib.request.proxy_bypass(host):
        return None
    return urlsplit(url if "://" in url else f"http://{url}")


def _proxy_auth(p) -> dict[str, str]:
    if not p.username:
        return {}
    token = base64.b64encode(f"{unquote(p.username)}:{unquote(p.password or '')}".encode()).decode()
    return {"Proxy-Authorization": f"Basic {token}"}


class Session:
    """Default headers plus per-thread connections. Thread-safe: threads never share a connection."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = dict(headers or {})
        self._local = threading.local()
        self._lock = threading.Lock()
        self._open: list[http.client.HTTPConnection] = []
        self._ssl = ssl.create_default_context()

    def _connection(self, scheme: str, host: str, port: int, timeout: float):
        conns = self._local.__dict__.setdefault("conns", {})
        key = (scheme, host, port)
        conn = conns.get(key)
        fresh = conn is None
        if fresh:
            proxy = _proxy_for(scheme, host)
            if scheme == "https":
                if proxy:
                    conn = http.client.HTTPSConnection(proxy.hostname, proxy.port or 8080, timeout=timeout,
                                                       context=self._ssl)
                    conn.set_tunnel(host, port, headers=_proxy_auth(proxy))
                else:
                    conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=self._ssl)
            else:
                target = proxy if proxy else None
                conn = http.client.HTTPConnection(target.hostname if target else host,
                                                  (target.port or 8080) if target else port, timeout=timeout)
                conn._llm2decision_proxy = proxy           # an absolute URL in the request line
            conns[key] = conn
            with self._lock:
                self._open.append(conn)
        conn.timeout = timeout
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        return conn, fresh

    def _drop(self, scheme: str, host: str, port: int) -> None:
        conn = self._local.__dict__.get("conns", {}).pop((scheme, host, port), None)
        if conn is not None:
            conn.close()

    def post(self, url: str, body: Any, *, timeout: float, headers: dict[str, str] | None = None) -> Response:
        u = urlsplit(url)
        scheme, host = u.scheme, u.hostname or ""
        port = u.port or (443 if scheme == "https" else 80)
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        data = json.dumps(body).encode("utf-8")
        hdrs = {**self.headers, **(headers or {}), "Content-Type": "application/json",
                "Content-Length": str(len(data)), "Accept": "application/json"}
        for _ in range(2):
            conn, fresh = self._connection(scheme, host, port, timeout)
            target = url if getattr(conn, "_llm2decision_proxy", None) else path
            if getattr(conn, "_llm2decision_proxy", None):
                hdrs.update(_proxy_auth(conn._llm2decision_proxy))
            try:
                conn.request("POST", target, body=data, headers=hdrs)
                r = conn.getresponse()
                raw = r.read()
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError) as e:
                self._drop(scheme, host, port)
                if fresh:                                   # a new connection failed: a real network error
                    raise NetworkError(type(e).__name__) from e
                continue                                    # a kept-alive one went stale: reopen once
            except (OSError, http.client.HTTPException) as e:
                self._drop(scheme, host, port)
                raise NetworkError(type(e).__name__) from e
            if (r.getheader("Connection") or "").lower() == "close":
                self._drop(scheme, host, port)
            return Response(r.status, r.headers, raw.decode("utf-8", "replace"))
        raise NetworkError("connection closed")              # pragma: no cover - two stale connections

    def close(self) -> None:
        with self._lock:
            conns, self._open = self._open, []
        for c in conns:
            c.close()
