"""Base URL validation for playground clients (SSRF guard)."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def validate_base_url(base_url: str, *, allow_internal: bool = False) -> str | None:
    """Return an error message when *base_url* is not allowed, else None."""
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https"):
        return f"Unsupported URL scheme {parsed.scheme!r}; use http or https."

    hostname = parsed.hostname
    if not hostname:
        return "Base URL must include a hostname."

    if allow_internal:
        return None

    if _hostname_is_blocked(hostname):
        return (
            f"Refusing to connect to internal address {hostname!r}. "
            "Pass --allow-internal for local or private networks."
        )

    return None


def _hostname_is_blocked(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True

    try:
        addr = ipaddress.ip_address(hostname)
        return _ip_is_blocked(addr)
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for info in infos:
        ip_str = info[4][0]
        try:
            if _ip_is_blocked(ipaddress.ip_address(ip_str)):
                return True
        except ValueError:
            continue
    return False


def _ip_is_blocked(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(addr.is_private or addr.is_link_local or addr.is_loopback)
