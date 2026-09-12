"""Live lookup of a container number in the PAS port system.

The gate's definition of a valid container is "one the terminal has on record",
so the authoritative check is a live query of cCntnrInfo, not the checksum and
not the cached export. The cached list (KnownContainers) stays as the fallback
when PAS is unreachable, and it is still what the snap correction uses, because
that needs the whole set rather than one row.

SAFETY: the endpoint executes whatever SQL it is sent. The only value ever
interpolated is a container number that has already been normalised to
[A-Z0-9]{11}; anything else is refused before a query is built. That, not
escaping, is what makes this injection-proof.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

import structlog

from apps.inference_worker.container import KnownContainers, normalise

log = structlog.get_logger()

SAFE_NUMBER = re.compile(r"^[A-Z0-9]{11}$")


class PasClient:
    def __init__(self, base_url: str, timeout: float = 3.0,
                 fallback: KnownContainers | None = None) -> None:
        self.url = f"{base_url.rstrip('/')}/run-pas-sql" if base_url else ""
        self.timeout = timeout
        self.fallback = fallback
        self._down_until = 0.0
        # Small positive cache: a container that was known a minute ago still is.
        self._cache: dict[str, tuple[bool, float]] = {}
        self._cache_ttl = 300.0

    @property
    def available(self) -> bool:
        return bool(self.url) and time.monotonic() >= self._down_until

    def is_known(self, number: str) -> bool | None:
        """True/False from PAS; on failure the cached list; None if neither can answer."""
        number = normalise(number)
        if not SAFE_NUMBER.match(number):
            return False

        hit = self._cache.get(number)
        if hit and time.monotonic() - hit[1] < self._cache_ttl:
            return hit[0]

        if self.available:
            try:
                known = self._query(number)
                self._cache[number] = (known, time.monotonic())
                return known
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                # Back off for a minute so a dead PAS does not add a timeout to
                # every read at the gate.
                self._down_until = time.monotonic() + 60.0
                log.warning("pas_lookup_failed_using_cache", error=str(exc))

        if self.fallback is not None:
            return number in self.fallback
        return None

    def _query(self, number: str) -> bool:
        # number is guaranteed [A-Z0-9]{11} here; see SAFE_NUMBER above.
        statement = f"SELECT TOP 1 CntnrNo FROM cCntnrInfo WHERE CntnrNo = '{number}'"
        body = json.dumps({"sqlStatement": statement}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        if payload.get("status") != 200:
            raise ValueError(f"PAS status {payload.get('status')}: {payload.get('message')}")
        rows = payload["data"][0]
        return len(rows) > 0
