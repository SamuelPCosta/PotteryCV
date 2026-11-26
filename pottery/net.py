"""Protocolo UDP v1 (specs/protocol.md): codificacao, sender e receiver."""
from __future__ import annotations

import random
import socket
import struct
import time

import numpy as np

MAGIC = b"PCV1"
VERSION = 1
T_STATE, T_ECHO = 1, 3
F_RESET, F_CALIB, F_PAUSED, F_SYNTH, F_VIEW = 1, 2, 4, 8, 16   # F_VIEW: pede ao Blender o JPEG da camera
VIEW_JPEG = "blender_view.jpg"   # em <projeto>/data/ (Blender grava, GUI le)
STATE_HDR = struct.Struct("<4sBBHIIQQHHHHBBHffffI")   # 64 bytes
ECHO = struct.Struct("<4sBBHIIQQQQ")                  # 48 bytes
assert STATE_HDR.size == 64 and ECHO.size == 48
MAX_UDP = 65507


def encode_state(R: np.ndarray, H: float, phi: float, omega: float, seq: int, frame_id: int, t_capture_ns: int,
                 version: int, flags: int = 0, r_scale: float = 0.30, max_datagram: int = 65000,
                 hfrac: np.ndarray | None = None, wetness: float = 0.0) -> list[bytearray]:
    """Payload de cada bloco: col_countxn_z raios (uint16) + col_count alturas relativas da coluna (uint16, 0..1 x H).
    wetness (0..1) vai no campo de 16 bits do cabecalho (offset 42)."""
    n_theta, n_z = R.shape
    q = np.round(np.clip(R / r_scale, 0, 1) * 65535).astype("<u2")
    hf = np.ones(n_theta) if hfrac is None else np.asarray(hfrac, dtype=np.float64)
    qh = np.round(np.clip(hf, 0, 1) * 65535).astype("<u2")
    qw = int(round(min(max(wetness, 0.0), 1.0) * 65535))
    cols_per = max(1, (min(max_datagram, MAX_UDP) - STATE_HDR.size) // (n_z * 2 + 2))
    chunks = list(range(0, n_theta, cols_per))
    out = []
    for ci, c0 in enumerate(chunks):
        cc = min(cols_per, n_theta - c0)
        nr = cc * n_z * 2
        buf = bytearray(STATE_HDR.size + nr + cc * 2)
        STATE_HDR.pack_into(buf, 0, MAGIC, VERSION, T_STATE, flags, seq & 0xFFFFFFFF, frame_id & 0xFFFFFFFF,
                            t_capture_ns, 0, n_theta, n_z, c0, cc, ci, len(chunks), qw,
                            H, r_scale, phi % (2 * np.pi), omega, version & 0xFFFFFFFF)
        buf[STATE_HDR.size:STATE_HDR.size + nr] = q[c0:c0 + cc].tobytes()
        buf[STATE_HDR.size + nr:] = qh[c0:c0 + cc].tobytes()
        out.append(buf)
    return out


def stamp_send(buf: bytearray, t_send_ns: int):
    struct.pack_into("<Q", buf, 24, t_send_ns)


def decode(data: bytes) -> dict | None:
    if len(data) < 8 or data[:4] != MAGIC or data[4] != VERSION:
        return None
    mt = data[5]
    if mt == T_STATE:
        if len(data) < STATE_HDR.size:
            return None
        (_, _, _, flags, seq, fid, tc, ts, nt, nz, c0, cc, ci, cn, qw, H, rs, phi, om, ver) = STATE_HDR.unpack_from(data)
        nr = cc * nz * 2
        if c0 + cc > nt or len(data) not in (STATE_HDR.size + nr, STATE_HDR.size + nr + cc * 2):
            return None
        q = np.frombuffer(data, "<u2", count=cc * nz, offset=STATE_HDR.size).reshape(cc, nz)
        if len(data) == STATE_HDR.size + nr + cc * 2:
            hf = np.frombuffer(data, "<u2", count=cc, offset=STATE_HDR.size + nr).astype(np.float32) / 65535.0
        else:                                    # formato antigo (sem alturas por coluna): topo uniforme
            hf = np.ones(cc, np.float32)
        return dict(type="state", flags=flags, seq=seq, frame_id=fid, t_capture_ns=tc, t_send_ns=ts, n_theta=nt,
                    n_z=nz, col_start=c0, col_count=cc, chunk_index=ci, chunk_count=cn, height=H, r_scale=rs,
                    phi=phi, omega=om, version=ver, wetness=qw / 65535.0, R=q.astype(np.float32) * (rs / 65535.0), hfrac=hf)
    if mt == T_ECHO and len(data) == ECHO.size:
        (_, _, _, _, seq, fid, tc, ts, tr, ta) = ECHO.unpack(data)
        return dict(type="echo", seq=seq, frame_id=fid, t_capture_ns=tc, t_send_ns=ts, t_recv_ns=tr, t_applied_ns=ta)
    return None


def encode_echo(seq, frame_id, t_capture_ns, t_send_ns, t_recv_ns, t_applied_ns) -> bytes:
    return ECHO.pack(MAGIC, VERSION, T_ECHO, 0, seq, frame_id, t_capture_ns, t_send_ns, t_recv_ns, t_applied_ns)


def seq_newer(a: int, b: int) -> bool:
    """a e mais novo (ou igual) que b, com wrap de 32 bits."""
    return ((a - b) & 0xFFFFFFFF) < 0x80000000


def _addr(s: str) -> tuple[str, int]:
    h, p = s.rsplit(":", 1)
    return h, int(p)


class Sender:
    def __init__(self, targets: list[str], rate_hz: float = 30.0, echo_port: int | None = 9001,
                 max_datagram: int = 65000, r_scale: float = 0.30):
        self.targets = [_addr(t) for t in targets]
        self.min_dt = 1.0 / rate_hz
        self.max_datagram, self.r_scale = max_datagram, r_scale
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.echo = None
        if echo_port:
            try:
                self.echo = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.echo.bind(("127.0.0.1", echo_port))
                self.echo.setblocking(False)
            except OSError:
                self.echo = None
        self.drop = 0.0          # fracao de datagramas descartados (teste de perda)
        self.seq = 0
        self._last = 0.0
        self._last_version = -1
        self.sent = 0

    def maybe_send(self, R, H, phi, omega, frame_id, t_capture_ns, version, flags=0, force=False, hfrac=None,
                   wetness=0.0) -> bool:
        now = time.perf_counter()   # monotonic no Windows tem ~15 ms de resolucao
        idle = version == self._last_version and omega == 0.0
        if not force and (now - self._last < (0.5 if idle else 0.9 * self.min_dt)):
            return False
        self._last, self._last_version = now, version
        self.seq += 1
        for buf in encode_state(R, H, phi, omega, self.seq, frame_id, t_capture_ns, version, flags,
                                self.r_scale, self.max_datagram, hfrac, wetness):
            stamp_send(buf, time.time_ns())
            if self.drop and random.random() < self.drop:
                continue
            for a in self.targets:
                try:
                    self.sock.sendto(buf, a)
                except OSError:
                    pass   # ninguem escutando: normal
        self.sent += 1
        return True

    def poll_echo(self) -> list[dict]:
        out = []
        if self.echo is None:
            return out
        while True:
            try:
                data, _ = self.echo.recvfrom(256)
            except (BlockingIOError, ConnectionResetError, OSError):
                break
            d = decode(data)
            if d and d["type"] == "echo":
                d["t_echo_rx_ns"] = time.time_ns()
                out.append(d)
        return out

    def close(self):
        self.sock.close()
        if self.echo:
            self.echo.close()


class Receiver:
    """Recebe STATE e mantem o mapa R mais recente (usado pelo --viewer e pelos testes)."""

    def __init__(self, port: int, echo_to: tuple[str, int] | None = None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.echo_to = echo_to
        self.R: np.ndarray | None = None
        self.HF: np.ndarray | None = None      # altura relativa de cada coluna (0..1 x H)
        self.meta: dict = {}
        self._chunk_seq: dict[int, int] = {}
        self.packets = 0
        self.last_rx = 0.0

    def poll(self) -> bool:
        """Drena o socket; True se algo mudou."""
        changed = False
        latest: dict[int, dict] = {}
        while True:
            try:
                data, _ = self.sock.recvfrom(MAX_UDP)
            except (BlockingIOError, ConnectionResetError, OSError):
                break
            d = decode(data)
            if not d or d["type"] != "state":
                continue
            t_recv = time.time_ns()
            d["t_recv_ns"] = t_recv
            ci = d["chunk_index"]
            if ci in latest and not seq_newer(d["seq"], latest[ci]["seq"]):
                continue
            latest[ci] = d
        for ci, d in latest.items():
            if ci in self._chunk_seq and not seq_newer(d["seq"], self._chunk_seq[ci]):
                continue
            self._chunk_seq[ci] = d["seq"]
            if self.R is None or self.R.shape != (d["n_theta"], d["n_z"]):
                self.R = np.zeros((d["n_theta"], d["n_z"]), np.float32)
                self.HF = np.ones(d["n_theta"], np.float32)
            self.R[d["col_start"]:d["col_start"] + d["col_count"]] = d["R"]
            self.HF[d["col_start"]:d["col_start"] + d["col_count"]] = d["hfrac"]
            self.meta = {k: v for k, v in d.items() if k not in ("R", "hfrac")}
            self.packets += 1
            self.last_rx = time.monotonic()
            changed = True
            if self.echo_to:
                try:
                    self.sock.sendto(encode_echo(d["seq"], d["frame_id"], d["t_capture_ns"], d["t_send_ns"],
                                                 d["t_recv_ns"], time.time_ns()), self.echo_to)
                except OSError:
                    pass
        return changed

    def close(self):
        self.sock.close()
