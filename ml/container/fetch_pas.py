"""Fetch every container number known to the PAS port system.

    python -m ml.container.fetch_pas --out dataset/pas/containers.json

Pages through cCntnrInfo grouped by container number, so each number appears
once with how many times it has been through the terminal and when it was last
seen. The list grows over time; re-run to refresh.

Every number is classified rather than trusted: the column holds some junk
("0126") and space padding, so each value is stripped and checked against the
ISO 6346 shape and check digit. Numbers that fail the checksum are kept but
flagged -- they are real entries in the port system and a gate read matching
one should still be recognised, but they must not be used as ground truth for
the check-digit logic.

Output feeds two things: a known-container lookup for the gate (a read that
matches a number the terminal has seen is far more trustworthy than one that
merely passes the checksum), and real owner-code frequencies for synthesising
training data that looks like this terminal's traffic.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from apps.inference_worker.container import parse
from config import get_settings

SHAPE = re.compile(r"^[A-Z]{4}\d{7}$")
PAGE = 2000
QUERY = (
    "SELECT CntnrNo, COUNT(*) AS visits, MAX(UpdtTime) AS last_seen "
    "FROM cCntnrInfo GROUP BY CntnrNo ORDER BY CntnrNo "
    "OFFSET {offset} ROWS FETCH NEXT {page} ROWS ONLY"
)


def run_sql(base_url: str, statement: str, retries: int = 4) -> list[dict]:
    body = json.dumps({"sqlStatement": statement}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/run-pas-sql", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            if payload.get("status") != 200:
                raise RuntimeError(f"PAS returned {payload.get('status')}: {payload.get('message')}")
            return payload["data"][0]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("dataset/pas/containers.json"))
    ap.add_argument("--url", default=None, help="override PAS_SQL_URL")
    args = ap.parse_args()

    base_url = args.url or get_settings().pas_sql_url
    if not base_url:
        raise SystemExit("set PAS_SQL_URL in .env (or pass --url)")

    containers: list[dict] = []
    stats = Counter()
    offset = 0
    while True:
        rows = run_sql(base_url, QUERY.format(offset=offset, page=PAGE))
        for row in rows:
            raw = (row.get("CntnrNo") or "").strip().upper()
            entry = {"number": raw, "visits": int(row.get("visits") or 0),
                     "last_seen": row.get("last_seen")}
            if SHAPE.match(raw):
                cn = parse(raw)
                entry["checksum_ok"] = bool(cn and cn.checksum_ok)
                stats["valid" if entry["checksum_ok"] else "bad_checksum"] += 1
            else:
                entry["checksum_ok"] = False
                stats["junk"] += 1
            containers.append(entry)
        offset += len(rows)
        print(f"  fetched {offset} distinct numbers...", flush=True)
        if len(rows) < PAGE:
            break

    owners = Counter(c["number"][:4] for c in containers if c.get("checksum_ok"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "PAS cCntnrInfo (grouped by CntnrNo)",
        "count": len(containers),
        "summary": dict(stats),
        "top_owner_codes": owners.most_common(30),
        "containers": containers,
    }, indent=1))

    print(f"\nsaved {len(containers)} distinct numbers -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  ISO 6346 valid      {stats['valid']}")
    print(f"  bad check digit     {stats['bad_checksum']}")
    print(f"  not a container no. {stats['junk']}")
    print(f"  top owner codes     {', '.join(f'{o}:{n}' for o, n in owners.most_common(8))}")


if __name__ == "__main__":
    main()
