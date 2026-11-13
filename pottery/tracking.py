"""MediaPipe (mãos + rosto), filtro One Euro e atribuição das mãos às bordas."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mpv
from mediapipe.tasks.python.core.base_options import BaseOptions


# ---------------------------------------------------------------- filtros
class OneEuro:
    """One Euro filter vetorizado (Casiez et al.). t em segundos."""

    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


# ---------------------------------------------------------------- mãos
PALM = [0, 5, 9, 13, 17]


@dataclass
class HandObs:
    track_id: int
    handedness: str          # "Left" | "Right" (rótulo do MediaPipe)
    score: float
    lm: np.ndarray           # (21,2) px, filtrado
    lm_raw: np.ndarray       # (21,2) px
    d09: float               # |pulso - nó médio| px
    d517: float              # |nó indicador - nó mínimo| px
    center: np.ndarray = field(default=None)

    def scale(self, kappa: float) -> float:
        return max(self.d09, kappa * self.d517)


class HandTracker:
    def __init__(self, models_dir: str, cfg_hand):
        self.cfg = cfg_hand
        self.det = mpv.HandLandmarker.create_from_options(mpv.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(models_dir) / "hand_landmarker.task")),
            running_mode=mpv.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        self._tracks: dict[int, tuple[np.ndarray, OneEuro]] = {}
        self._next_id = 0
        self._last_ms = -1

    def process(self, rgb: np.ndarray, ts_ns: int) -> list[HandObs]:
        ms = max(int(ts_ns // 1_000_000), self._last_ms + 1)
        self._last_ms = ms
        res = self.det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        h, w = rgb.shape[:2]
        t = ts_ns / 1e9
        out, used = [], set()
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            score = hd[0].score
            if score < self.cfg.min_score:
                continue
            raw = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)
            c = raw[PALM].mean(0)
            d09 = float(np.linalg.norm(raw[0] - raw[9]))
            # associa à track mais próxima
            best, bd = None, 0.6 * d09
            for tid, (pc, _) in self._tracks.items():
                if tid in used:
                    continue
                d = np.linalg.norm(pc - c)
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self._next_id
                self._next_id += 1
                self._tracks[best] = (c, OneEuro(self.cfg.one_euro_min_cutoff, self.cfg.one_euro_beta))
            used.add(best)
            filt = self._tracks[best][1]
            lm = filt(raw, t) if self.cfg.smoothing else raw
            self._tracks[best] = (c, filt)
            out.append(HandObs(best, hd[0].category_name, score, lm, raw,
                               float(np.linalg.norm(lm[0] - lm[9])), float(np.linalg.norm(lm[5] - lm[17])),
                               lm[PALM].mean(0)))
        for tid in list(self._tracks):
            if tid not in used:
                del self._tracks[tid]
        return out

    def close(self):
        self.det.close()


# ---------------------------------------------------------------- cabeça
@dataclass
class HeadObs:
    ok: bool
    pitch: float = 0.0   # graus
    yaw: float = 0.0
    roll: float = 0.0
    bbox: tuple | None = None


def matrix_to_euler(m: np.ndarray) -> tuple[float, float, float]:
    """Rotação 3x3 -> (pitch, yaw, roll) em graus (convenção x-y-z)."""
    r = m[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


class HeadTracker:
    def __init__(self, models_dir: str):
        self.det = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(models_dir) / "face_landmarker.task")),
            running_mode=mpv.RunningMode.VIDEO, num_faces=1, output_facial_transformation_matrixes=True))
        self._last_ms = -1

    def process(self, rgb: np.ndarray, ts_ns: int) -> HeadObs:
        ms = max(int(ts_ns // 1_000_000), self._last_ms + 1)
        self._last_ms = ms
        res = self.det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not res.facial_transformation_matrixes:
            return HeadObs(False)
        p, y, r = matrix_to_euler(np.asarray(res.facial_transformation_matrixes[0]))
        h, w = rgb.shape[:2]
        pts = np.array([[q.x * w, q.y * h] for q in res.face_landmarks[0]])
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        return HeadObs(True, p, y, r, (int(x0), int(y0), int(x1), int(y1)))

    def close(self):
        self.det.close()


# ---------------------------------------------------------------- bordas
def assign_edges(hands: list[HandObs], axis_u: float, sticky: dict[str, int]) -> dict[str, HandObs | None]:
    """Borda pela posição em relação ao eixo ('R' = direita da tela, 'L' = esquerda).
    sticky: {edge: track_id} das bordas em contato — mantém a mesma mão mesmo se cruzar o eixo."""
    out: dict[str, HandObs | None] = {"R": None, "L": None}
    free = list(hands)
    for e, tid in sticky.items():
        for h in free:
            if h.track_id == tid:
                out[e] = h
                free.remove(h)
                break
    for h in sorted(free, key=lambda h: abs(h.center[0] - axis_u)):  # a mais perto do eixo ganha
        e = "R" if h.center[0] >= axis_u else "L"
        if out[e] is None:
            out[e] = h
    return out
