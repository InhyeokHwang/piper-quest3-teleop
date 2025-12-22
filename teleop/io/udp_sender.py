# io/udp_sender.py

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Sequence, Optional


@dataclass
class UDPSenderConfig:
    ip: str = "127.0.0.1"
    port: int = 15000
    fmt: str = "{:.6f}"          # float formatting for each value
    sep: str = " "               # separator
    encoding: str = "utf-8"


class UDPSender:
    """
    Simple UDP sender for streaming joint solutions etc.
    """

    def __init__(self, config: UDPSenderConfig | None = None):
        self.cfg = config if config is not None else UDPSenderConfig()
        self._sock: Optional[socket.socket] = None

    def open(self):
        if self._sock is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self):
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def send_floats(self, values: Sequence[float]):
        """
        Send float list as a single space-separated UDP message.
        Example payload: "0.123456 -1.234500 ..."
        """
        if self._sock is None:
            self.open()

        msg = self.cfg.sep.join(self.cfg.fmt.format(float(x)) for x in values)
        self._sock.sendto(msg.encode(self.cfg.encoding), (self.cfg.ip, self.cfg.port))

    def send_text(self, text: str):
        if self._sock is None:
            self.open()
        self._sock.sendto(text.encode(self.cfg.encoding), (self.cfg.ip, self.cfg.port))
