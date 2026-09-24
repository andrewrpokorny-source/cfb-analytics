"""Disk-cached CFBD fetch.

Historical CFBD data is immutable once a season is over, so we cache
aggressively. This keeps walk-forward rebuilds fast and keeps us well
under the API rate limit.
"""
import hashlib
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HEADERS  # noqa: E402

BASE_URL = "https://api.collegefootballdata.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")

_MEM = {}


def _key(endpoint, params):
    blob = endpoint + "|" + json.dumps(params, sort_keys=True)
    slug = endpoint.strip("/").replace("/", "_")
    return slug + "_" + hashlib.md5(blob.encode()).hexdigest()[:12]


def fetch(endpoint, params=None, refresh=False):
    """GET an endpoint, memoised in-process and on disk."""
    params = params or {}
    k = _key(endpoint, params)

    if not refresh and k in _MEM:
        return _MEM[k]

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, k + ".json")

    if not refresh and os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            _MEM[k] = data
            return data
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry; refetch

    url = f"{BASE_URL}{endpoint}"
    data, ok, last_status = [], False, None
    for attempt in range(1, 5):
        try:
            res = requests.get(url, headers=HEADERS, params=params, timeout=60)
            last_status = res.status_code
            if res.status_code == 200:
                data, ok = res.json(), True
                break
            if res.status_code == 429:
                # A monthly-quota 429 will not clear by waiting; a burst 429 will.
                if "quota" in res.text.lower():
                    break
                time.sleep(8 * attempt)
                continue
            if res.status_code >= 500:
                time.sleep(3 * attempt)
                continue
            break
        except requests.RequestException:
            time.sleep(3 * attempt)

    if not ok:
        # NEVER cache a failure. Writing [] here once poisoned 77 historical
        # entries when the CFBD monthly quota ran out: every later read saw
        # "no data" and silently treated it as truth. Return [] for this call
        # only, loudly, and leave the disk alone so the next run refetches.
        print(f"   ! {endpoint} {params} -> HTTP {last_status} (not cached)")
        return []

    with open(path, "w") as f:
        json.dump(data, f)
    _MEM[k] = data
    return data
