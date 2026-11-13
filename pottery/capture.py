"""Captura de webcam em thread, gravação e replay de sessões."""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}


@dataclass
class Frame:
    id: int
    ts_ns: int          # time.time_ns() na captura
    image: np.ndarray   # BGR, já espelhada


class Camera:
    """Mantém só o quadro mais recente (descarta atrasados)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.index, _BACKENDS.get(cfg.backend, cv2.CAP_ANY))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)   # (BUFFERSIZE=1 derruba o MSMF para ~11 FPS; a thread já descarta atrasados)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não abriu a câmera {cfg.index}")
        self._latest: Frame | None = None
        self._last_read_id = -1
        self._cond = threading.Condition()
        self._running = True
        self._n = 0
        self.last_frame_time = time.monotonic()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while self._running:
            ok, img = self.cap.read()
            ts = time.time_ns()
            if not ok:
                time.sleep(0.01)
                continue
            if self.cfg.mirror:
                img = cv2.flip(img, 1)
            with self._cond:
                self._latest = Frame(self._n, ts, img)
                self._n += 1
                self.last_frame_time = time.monotonic()
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None and self._latest.id != self._last_read_id, timeout):
                return None
            self._last_read_id = self._latest.id
            return self._latest

    def stalled(self, secs=1.0) -> bool:
        return time.monotonic() - self.last_frame_time > secs

    def close(self):
        self._running = False
        self._t.join(timeout=1.0)
        self.cap.release()


class Replay:
    """Reproduz uma sessão gravada (video.mp4 + frames_ts.csv). realtime=False roda o mais rápido possível."""

    def __init__(self, folder: str | Path, realtime: bool = True):
        folder = Path(folder)
        self.cap = cv2.VideoCapture(str(folder / "video.mp4"))
        with open(folder / "frames_ts.csv", newline="") as f:
            self.ts = [int(r["ts_ns"]) for r in csv.DictReader(f)]
        self.realtime = realtime
        self.i = 0
        self.eof = False
        self._t0 = None

    def read(self, timeout: float = 1.0) -> Frame | None:
        if self.i >= len(self.ts):
            self.eof = True
            return None
        ok, img = self.cap.read()
        if not ok:
            self.eof = True
            return None
        ts = self.ts[self.i]
        if self.realtime:
            if self._t0 is None:
                self._t0 = (time.monotonic_ns(), ts)
            wait = (ts - self._t0[1]) - (time.monotonic_ns() - self._t0[0])
            if wait > 0:
                time.sleep(wait / 1e9)
        fr = Frame(self.i, ts, img)
        self.i += 1
        return fr

    def stalled(self, secs=1.0) -> bool:
        return False

    def close(self):
        self.cap.release()


class Recorder:
    def __init__(self, folder: str | Path, size: tuple[int, int], fps: float = 30.0):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.vw = cv2.VideoWriter(str(self.folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.f = open(self.folder / "frames_ts.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["frame_id", "ts_ns"])

    def write(self, fr: Frame):
        self.vw.write(fr.image)
        self.w.writerow([fr.id, fr.ts_ns])

    def close(self):
        self.vw.release()
        self.f.close()
