"""Helpers for child processes InferenceX owns (llama-server, supervisor workers)."""

from __future__ import annotations

import ctypes
import signal
import socket

_PR_SET_PDEATHSIG = 1


def free_port() -> int:
    """A loopback port that was free a moment ago.

    ponytail: bind-then-release has a small race; losing it makes the child
    fail to bind and exit, which its owner reports as a startup failure.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def die_with_parent() -> None:
    """`preexec_fn` for a child: have the kernel SIGTERM it if its parent dies.

    The signal fires when the thread that spawned the child exits, so spawn
    from a thread that lives as long as the process (the startup thread or
    the event loop thread).
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except OSError:
        pass
