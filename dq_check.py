"""Convenience entry point matching the brief: ``python dq_check.py --input ./data/raw``."""

import sys

from dq.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
