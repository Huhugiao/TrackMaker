#!/usr/bin/env python3
"""Run a command with an open PTY on stdin and preserve its process group."""

from __future__ import annotations

import os
import pty
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: pty_exec.py COMMAND [ARG ...]", file=sys.stderr)
        return 2
    master, slave = pty.openpty()
    try:
        process = subprocess.Popen(sys.argv[1:], stdin=slave)
    finally:
        os.close(slave)
    try:
        return process.wait()
    except KeyboardInterrupt:
        return process.wait()
    finally:
        os.close(master)


if __name__ == "__main__":
    raise SystemExit(main())
