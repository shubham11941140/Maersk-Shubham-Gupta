"""Block until a URL returns HTTP 200 (used by `make docker-up`). Stdlib only."""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request


def main(url: str = "http://localhost:8000/health", timeout_s: float = 120.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 - local health URL
                if resp.status == 200:
                    print(f"ready: {url}")
                    return 0
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    print(f"timed out waiting for {url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
