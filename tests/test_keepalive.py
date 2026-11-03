import asyncio

from api import keepalive


def test_off_without_a_public_url():
    assert keepalive.target_url({}) is None
    assert keepalive.target_url({"RENDER_EXTERNAL_URL": "not-a-url"}) is None


def test_url_from_render_or_override():
    assert keepalive.target_url({"RENDER_EXTERNAL_URL": "https://x.onrender.com/"}) == "https://x.onrender.com/health"
    env = {"RENDER_EXTERNAL_URL": "https://x.onrender.com", "KEEPALIVE_URL": "https://y.example.org"}
    assert keepalive.target_url(env) == "https://y.example.org/health"
    assert keepalive.target_url({**env, "KEEPALIVE_DISABLED": "1"}) is None


def test_interval_default_ten_minutes_and_floor():
    assert keepalive.interval_seconds({}) == 600
    assert keepalive.interval_seconds({"KEEPALIVE_SECONDS": "5"}) == 60
    assert keepalive.interval_seconds({"KEEPALIVE_SECONDS": "abc"}) == 600


def test_loop_pings_and_survives_failures():
    calls = []

    def pinger(url):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("cold start")
        return 200

    asyncio.run(keepalive.run("https://x/health", 0, pinger, max_pings=3))
    assert calls == ["https://x/health"] * 3
