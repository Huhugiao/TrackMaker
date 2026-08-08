"""Runtime process logging helpers."""

import os
import socket
import sys


def print_training_process_info(entrypoint: str) -> int:
    """Print the main trainer PID so it can be matched in nvidia-smi."""
    pid = int(os.getpid())
    ppid = int(os.getppid())
    host = socket.gethostname()
    print(
        f"[PID] {entrypoint}: pid={pid} ppid={ppid} host={host} python={sys.executable}",
        flush=True,
    )
    print(
        f"[PID] Query with: nvidia-smi | grep ' {pid} '",
        flush=True,
    )
    return pid
