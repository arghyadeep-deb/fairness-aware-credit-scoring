"""Keep-alive for hosted deployments (e.g. Render's free tier, which sleeps a service after 15 minutes idle).

Every KEEPALIVE_SECONDS (default 600 = 10 minutes) the app sends GET <public url>/health to itself, so the
request arrives through the public URL and counts as traffic. It is OFF unless a public URL is known:
  - KEEPALIVE_URL        set by you, e.g. https://my-app.onrender.com
  - RENDER_EXTERNAL_URL  set automatically by Render on every web service
Set KEEPALIVE_DISABLED=1 to turn it off. Runs locally and in tests only if one of the URLs is set.
No applicant data is sent; only the health check.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from collections.abc import Callable

log = logging.getLogger("api.keepalive")
DEFAULT_SECONDS = 600


def target_url(env: dict | None = None) -> str | None:
    env = os.environ if env is None else env
    if env.get("KEEPALIVE_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return None
    base = (env.get("KEEPALIVE_URL") or env.get("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        return None
    return base + "/health"


def interval_seconds(env: dict | None = None) -> int:
    env = os.environ if env is None else env
    try:
        return max(60, int(env.get("KEEPALIVE_SECONDS", DEFAULT_SECONDS)))
    except ValueError:
        return DEFAULT_SECONDS


def ping(url: str, timeout: float = 30.0) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "credit-scoring-keepalive"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL comes from our own env)
        return resp.status


async def run(url: str, seconds: int, pinger: Callable[[str], int] = ping, max_pings: int | None = None) -> None:
    """Sleep, ping, repeat. A failed ping is logged and never stops the loop or the app."""
    n = 0
    while max_pings is None or n < max_pings:
        await asyncio.sleep(seconds)
        try:
            status = await asyncio.to_thread(pinger, url)
            log.info("keep-alive ping %s -> %s", url, status)
        except Exception as exc:  # network hiccup, cold start, etc.
            log.warning("keep-alive ping failed: %s", type(exc).__name__)
        n += 1


def start() -> asyncio.Task | None:
    url = target_url()
    if url is None:
        return None
    seconds = interval_seconds()
    log.info("keep-alive on: %s every %ss", url, seconds)
    return asyncio.create_task(run(url, seconds))
