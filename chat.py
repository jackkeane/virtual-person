#!/usr/bin/env python3
from __future__ import annotations

import sys
import httpx

BASE_URL = "http://127.0.0.1:8080"
USER_ID = "liyang"


def main() -> int:
    print("Ani terminal chat. Type /exit to quit.")
    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not text:
            continue
        if text in {"/exit", "/quit"}:
            print("Bye.")
            return 0

        try:
            r = httpx.post(
                f"{BASE_URL}/chat/turn",
                json={"user_id": USER_ID, "message": text},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            print(f"Ani> {data.get('response', '')}")
        except Exception as e:
            print(f"[error] {e}")


if __name__ == "__main__":
    sys.exit(main())
