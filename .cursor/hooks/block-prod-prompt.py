#!/usr/bin/env python3
"""Block user prompts that reference production environments."""

from __future__ import annotations

import json
import re
import sys

PROD_PATTERN = re.compile(r"production|\bprod\b|\bprd\b", re.IGNORECASE)

DENY_RESPONSE = {
    "continue": False,
    "user_message": (
        "Prompt blocked: this message references production "
        "(matched prod, prd, or production). Use a non-production target instead."
    ),
}

ALLOW_RESPONSE = {"continue": True}


def main() -> int:
    payload = sys.stdin.read()
    if PROD_PATTERN.search(payload):
        print(json.dumps(DENY_RESPONSE))
        return 0

    print(json.dumps(ALLOW_RESPONSE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
