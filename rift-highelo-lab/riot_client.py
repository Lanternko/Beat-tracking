"""Rate-limited Riot API client：多 key 輪替 + 429 退避 + 磁碟快取（可續傳）。

- 每把 key 各自一個 sliding-window 限流器；每個請求挑「最快可用」的 key。
- 兩把有效 key ≈ 2× 吞吐（瓶頸是 rate limit，單執行緒即可）。
- 只用標準庫（urllib），任何 python3 都能跑。
"""
import collections
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import config

# Riot API 在 Cloudflare 後面，預設的 "Python-urllib/x.y" UA 會被 1010 擋掉。
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class RateLimiter:
    """Sliding-window 限流：同時滿足多個 (次數, 秒數) 限制。"""

    def __init__(self, limits):
        self.limits = limits
        self.calls = collections.deque()
        self._horizon = max(s for _, s in limits)

    def wait_time(self):
        """還要等幾秒才能發下一個請求（0 = 現在就能發）。不記錄。"""
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self._horizon:
            self.calls.popleft()
        wait = 0.0
        for count, seconds in self.limits:
            in_window = [t for t in self.calls if now - t <= seconds]
            if len(in_window) >= count:
                wait = max(wait, seconds - (now - in_window[0]) + 0.02)
        return wait

    def record(self):
        self.calls.append(time.monotonic())


class RiotClient:
    def __init__(self, api_keys=None, cache_dir=None):
        keys = api_keys or config.load_api_keys()
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.limiters = [RateLimiter(config.RATE_LIMITS) for _ in keys]
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = 0
        self.n_cache_hits = 0

    def get(self, host, path, params=None, use_cache=True):
        qs = ("?" + urllib.parse.urlencode(params)) if params else ""
        url = f"https://{host}{path}{qs}"
        cache_file = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if use_cache and cache_file.exists():
            self.n_cache_hits += 1
            return json.loads(cache_file.read_text())
        data = self._fetch(url)
        if use_cache:
            cache_file.write_text(json.dumps(data))
        return data

    def _pick_key(self):
        waits = [lim.wait_time() for lim in self.limiters]
        i = min(range(len(waits)), key=lambda j: waits[j])
        return i, waits[i]

    def _fetch(self, url, max_retries=6):
        for attempt in range(max_retries):
            i, w = self._pick_key()
            if w > 0:
                time.sleep(w)
            self.limiters[i].record()
            req = urllib.request.Request(
                url, headers={"X-Riot-Token": self.keys[i], "User-Agent": USER_AGENT}
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    self.n_calls += 1
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(int(e.headers.get("Retry-After", "2")) + 0.5)
                    continue
                if e.code in (500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"HTTP {e.code} for {url}: {e.read().decode()[:200]}")
            except urllib.error.URLError:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"重試 {max_retries} 次仍失敗：{url}")
