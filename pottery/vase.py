"""Vaso: mapa R(theta,z), varredura de arco, contato (so por fora / agarrar), nivel 2 e altura por volume."""
from __future__ import annotations

import math

import cv2
import numpy as np

EDGE_THETA = {"R": 0.0, "L": math.pi}   # angulo de mundo da borda (direita/esquerda da tela)
TWO_PI = 2 * math.pi


def sweep(theta_e: float, phi0: float, phi1: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colunas locais varridas pela borda entre phi0 e phi1. Intervalo semiaberto (a, b]: cada coluna 1x por passagem.
    Retorna (indices i, fracoes s em (0,1], inteiros n nao reduzidos)."""
    a = (theta_e - phi0) * n / TWO_PI
    b = (theta_e - phi1) * n / TWO_PI
    if abs(b - a) >= n:
        ns = np.arange(n)
        return ns, np.ones(n), ns
    if abs(b - a) < 1e-12:
        ns = np.array([int(round(b))])
        return ns % n, np.ones(1), ns
    if b > a:
        ns = np.arange(math.floor(a) + 1, math.floor(b) + 1)
    else:
        ns = np.arange(math.ceil(a) - 1, math.ceil(b) - 1, -1)
    if len(ns) == 0:          # girou menos de 1 coluna: nada novo sob a borda neste quadro
        e = np.zeros(0, int)
        return e, np.zeros(0), e
    s = (ns - a) / (b - a)
    return ns % n, s, ns


class EdgeContact:
    """Maquina de estados do contato de uma borda: a mao so empurra por fora (com histerese)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "FREE"
        self.track_id: int | None = None
        self._below = 0
        self._last_seen = 0.0
        self.pen = float("nan")

    def update(self, prof, track_id, wall: np.ndarray, now: float) -> str:
        c = self.cfg
        if prof is None:
            self.pen = float("nan")
            if self.state != "FREE" and now - self._last_seen > c.lost_ms / 1000:
                self._free()
            return "NONE"
        self._last_seen = now
        cov = prof.cover & ~np.isnan(wall)          # so onde existe parede (abaixo do topo desta coluna)
        if not cov.any():
            self.pen = float("nan")
            return "PUSH" if self.state == "PUSH" else "NONE"
        pen = float(np.max(wall[cov] - prof.r_near[cov]))
        self.pen = pen
        if self.state == "FREE":
            outside = prof.centroid_r >= float(np.mean(wall[cov]))   # so engata quem vem de fora
            if pen >= -c.engage_m and outside:
                self.state, self.track_id, self._below = "PUSH", track_id, 0
        if self.state == "PUSH":
            if -pen > c.release_m:
                self._below += 1
                if self._below >= c.release_frames:
                    self._free()
                    return "NONE"
            else:
                self._below = 0
            return "PUSH"
        return "NONE"

    def _free(self):
        self.state, self.track_id, self._below = "FREE", None, 0


class Vase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_theta, self.n_z = cfg.n_theta, cfg.n_z
        self.version = 0
        self.reset()

    """Mapa R(θ, z) com ALTURA POR COLUNA: a coluna i vai de z = 0 até Hc[i]; o bin j fica em z = Hc[i]·j/(N_z−1).
    Assim pressionar o topo abaixa só as colunas que passam sob a mão (o torno gira e vai baixando a borda)."""

    def reset(self):
        c = self.cfg
        self.R = np.full((self.n_theta, self.n_z), c.r0, np.float32)
        self.Hc = np.full(self.n_theta, min(c.h0, c.h_max), np.float64)
        self.V0 = math.pi * self.Hc[0] * c.r0 ** 2
        self.phi = 0.0
        self.omega = 0.0
        self.version += 1
        self.reset_flag = True
        w = np.ones(self.n_z)
        w[0] = w[-1] = 0.5
        self._wz = w / w.sum()                       # pesos de trapezio em z (media por coluna)
        self._zn = np.linspace(0.0, 1.0, self.n_z)

    # ---------------------------------------------------- consultas
    @property
    def H(self) -> float:
        """Altura maxima (topo mais alto) - escala global do vaso."""
        return float(self.Hc.max())

    def col_index(self, edge: str, phi: float | None = None) -> int:
        phi = self.phi if phi is None else phi
        return int(round((EDGE_THETA[edge] - phi) * self.n_theta / TWO_PI)) % self.n_theta

    def wall(self, edge: str, z_grid: np.ndarray | None = None) -> np.ndarray:
        """Raio da coluna sob a borda. Com z_grid (z global), reamostra para essas alturas (NaN acima do topo)."""
        i = self.col_index(edge)
        if z_grid is None:
            return self.R[i]
        zc = self.Hc[i] * self._zn
        return np.where(z_grid <= self.Hc[i] + 1e-9, np.interp(z_grid, zc, self.R[i]), np.nan)

    def top_at(self, edge: str) -> float:
        return float(self.Hc[self.col_index(edge)])

    def col_r2(self, R=None) -> np.ndarray:
        R = self.R if R is None else R
        return (R.astype(np.float64) ** 2) @ self._wz

    def mean_r2(self) -> float:
        return float(self.col_r2().mean())

    def volume(self) -> float:
        return float(math.pi * np.mean(self.Hc * self.col_r2()))

    # ---------------------------------------------------- dinamica
    def advance(self, dt: float) -> tuple[float, float]:
        phi0 = self.phi
        self.phi = phi0 + self.omega * dt
        return phi0, self.phi

    def apply(self, edge: str, phi0: float, phi1: float, p_prev, p_cur, action: str) -> np.ndarray | None:
        if action != "PUSH" or p_cur is None:
            return None
        c = self.cfg
        cols, s, ns = sweep(EDGE_THETA[edge], phi0, phi1, self.n_theta)
        if len(cols) == 0:
            return None
        # o perfil vem numa grade de z GLOBAL (0..H); cada coluna tem a sua altura -> reamostra para os z dela
        zg = np.linspace(0.0, self.H, self.n_z)
        zc = (self.Hc[cols][:, None] * self._zn[None]).ravel()

        def at(arr):
            return np.interp(zc, zg, arr).reshape(len(cols), self.n_z)

        t1, w1, c1 = at(np.nan_to_num(p_cur.r_near)), at(p_cur.weight), at(p_cur.cover.astype(float)) > 0.5
        if p_prev is not None:
            t0, w0, c0 = at(np.nan_to_num(p_prev.r_near)), at(p_prev.weight), at(p_prev.cover.astype(float)) > 0.5
            both = c0 & c1
            sc = s[:, None]
            T = np.where(both, (1 - sc) * t0 + sc * t1, t1)
            W = np.where(both, (1 - sc) * w0 + sc * w1, w1)
        else:
            T, W = t1, w1
        old = self.R.copy()
        sub = self.R[cols].astype(np.float64)
        m = c1 & (sub > T)
        sub = np.where(m, sub - c.alpha * W * (sub - T), sub)
        self.R[cols] = sub.astype(np.float32)
        self._spread(old, ns)
        np.clip(self.R, c.r_min, c.r_max, out=self.R)
        # altura POR COLUNA pela conservacao de volume (V_col = pi*Hc*media_z(R^2)), atenuada por side_gain:
        # 1 = conserva exatamente (apertar sobe muito), 0 = altura fixa
        if c.side_gain > 0:
            ch = np.flatnonzero(np.any(self.R != old, axis=1))
            mb, ma = self.col_r2(old[ch]), self.col_r2(self.R[ch])
            ok = ma > 0
            self.Hc[ch[ok]] = np.clip(self.Hc[ch[ok]] * (mb[ok] / ma[ok]) ** c.side_gain, c.h_min, c.h_max)
        self.version += 1
        return cols

    def press_top(self, edge: str, phi0: float, phi1: float, z_target: float) -> bool:
        """Mao empurrando o topo para baixo: so as colunas que passam sob a mao (varredura de arco) abaixam,
        alargam e fazem barriga (pico a 1/3 da altura), conservando o volume daquela coluna."""
        c = self.cfg
        cols, _, _ = sweep(EDGE_THETA[edge], phi0, phi1, self.n_theta)
        if len(cols) == 0:
            return False
        h = self.Hc[cols]
        hit = h > z_target
        if not hit.any():
            return False
        cols, h = cols[hit], h[hit]
        h_new = np.maximum(c.h_min, h - c.alpha * c.top_rate * (h - z_target))
        # o barro que sai da altura alarga a coluna toda e ainda faz barriga: soma em R^2 um perfil
        # = metade por igual + metade com pico em 1/3 da altura. Volume = pi*Hc*media_z(R^2) continua exato.
        R2 = self.R[cols].astype(np.float64) ** 2
        m = R2 @ self._wz                                    # media_z(R^2) antes
        w = 1.0 + 12.0 * self._zn * (1 - self._zn) ** 2     # 1 = alarga tudo; 12*zn*(1-zn)^2 = barriga (media ~1)
        add = (m * (h / h_new - 1) / (w @ self._wz))[:, None] * w[None]
        self.R[cols] = np.sqrt(R2 + add).astype(np.float32)
        np.clip(self.R, c.r_min, c.r_max, out=self.R)
        self.Hc[cols] = h_new
        self.version += 1
        return True

    def _spread(self, old: np.ndarray, ns: np.ndarray):
        """Espalha SO a deformacao deste quadro para os vizinhos (Gaussiana em theta e z).
        Nao mexe no que ja foi modelado: empurrar nunca devolve barro a um sulco antigo."""
        c = self.cfg
        lam = c.smooth_lambda
        if lam <= 0:
            return
        k = int(3 * max(c.smooth_sigma_theta, 1)) + 1
        N = self.n_theta
        if len(ns) + 2 * k >= N:
            idx = np.arange(-k, N + k) % N          # volta inteira: borda circular
            mid = slice(k, k + N)
            out_idx = np.arange(N)
        else:
            lo, hi = int(ns.min()), int(ns.max())
            idx = np.arange(lo - k, hi + k + 1) % N
            mid = slice(None)
            out_idx = idx
        D = (self.R[idx] - old[idx]).astype(np.float32)
        blur = cv2.GaussianBlur(D, (0, 0), sigmaX=c.smooth_sigma_z, sigmaY=c.smooth_sigma_theta, borderType=cv2.BORDER_CONSTANT)
        D = (1 - lam) * D + lam * blur
        self.R[out_idx] = old[out_idx] + D[mid]



class TopContact:
    """Mao por cima do vaso empurrando o topo para baixo (com histerese)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "FREE"
        self._above = 0
        self.z_low = float("nan")

    def update(self, z_low: float | None, centroid_z: float | None, H: float) -> bool:
        c = self.cfg
        if z_low is None:
            self.z_low = float("nan")
            self.state, self._above = "FREE", 0
            return False
        self.z_low = z_low
        if self.state == "FREE":
            if centroid_z is not None and centroid_z > H and z_low <= H + c.engage_m:   # so engata quem vem de cima
                self.state, self._above = "PRESS", 0
        elif z_low > H + c.release_m:
            self._above += 1
            if self._above >= c.release_frames:
                self.state = "FREE"
        else:
            self._above = 0
        return self.state == "PRESS"
