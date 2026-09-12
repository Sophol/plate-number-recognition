"""Tests for the live PAS lookup: safety, fallback, and back-off.

_query is stubbed; what is under test is everything around it -- that only a
validated number can reach the SQL, that a PAS failure falls back to the cached
list instead of failing the read, and that a dead PAS is not retried on every
frame.
"""

import urllib.error

from apps.inference_worker.container import KnownContainers
from apps.inference_worker.pas import PasClient

CACHE = KnownContainers(["TCLU5437389"])


def test_malformed_input_never_reaches_the_query(monkeypatch):
    client = PasClient("http://pas.example")
    called = []
    monkeypatch.setattr(client, "_query", lambda n: called.append(n) or True)
    # An injection attempt, and simply too-short / too-long strings.
    for bad in ["TCLU'; DROP TABLE cCntnrInfo;--", "TCLU543738", "TCLU54373899", ""]:
        assert client.is_known(bad) is False
    assert called == []


def test_live_answer_is_used_and_cached(monkeypatch):
    client = PasClient("http://pas.example")
    calls = []
    monkeypatch.setattr(client, "_query", lambda n: calls.append(n) or (n == "TCLU5437389"))
    assert client.is_known("tclu 543738 9") is True
    assert client.is_known("TCLU5437389") is True      # served from cache
    assert client.is_known("CMAU7224270") is False
    assert calls == ["TCLU5437389", "CMAU7224270"]


def test_pas_failure_falls_back_to_the_cached_list(monkeypatch):
    client = PasClient("http://pas.example", fallback=CACHE)

    def boom(n):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(client, "_query", boom)

    assert client.is_known("TCLU5437389") is True
    assert client.is_known("CMAU7224270") is False


def test_pas_failure_without_cache_is_unknown_not_false(monkeypatch):
    client = PasClient("http://pas.example")
    monkeypatch.setattr(client, "_query", lambda n: (_ for _ in ()).throw(TimeoutError()))
    assert client.is_known("TCLU5437389") is None


def test_dead_pas_is_not_retried_every_read(monkeypatch):
    client = PasClient("http://pas.example", fallback=CACHE)
    calls = []

    def boom(n):
        calls.append(n)
        raise urllib.error.URLError("down")
    monkeypatch.setattr(client, "_query", boom)

    client.is_known("TCLU5437389")
    client.is_known("CMAU7224270")
    client.is_known("MRSU2818883")
    assert len(calls) == 1                   # backed off after the first failure
    assert client.available is False


def test_no_url_means_cache_only():
    client = PasClient("", fallback=CACHE)
    assert client.available is False
    assert client.is_known("TCLU5437389") is True
