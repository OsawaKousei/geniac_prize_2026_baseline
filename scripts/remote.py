#!/usr/bin/env python3
"""Run from a local checkout; no third-party Python dependencies required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_tutorial.client import main

if __name__ == "__main__":
    raise SystemExit(main())
