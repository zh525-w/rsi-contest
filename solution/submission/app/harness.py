"""平台固定入口：python3 /app/harness.py describe|train|predict"""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
