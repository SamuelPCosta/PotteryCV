"""Extratores de mascara da mao (landmarks | pele | MobileSAM) -> contorno real (com os dedos) -> perfil r_near(z), gesto."""
from __future__ import annotations

import queue
import time
from dataclasses import dataclass

import cv2
import numpy as np

EXTRACTORS = ["landmarks", "mobilesam", "sam3"]
_SAM_WEIGHTS = {"mobilesam": "mobile_sam.pt", "sam3": "sam3.pt"}   # sam3.pt: baixar do HF (acesso restrito)
_MIN_AGE = {"sam3": 24}   # SAM 3 leva ~460 ms (RTX 3060, fp16): a mascara vale por mais quadros, propagada pelos landmarks
FINGERS = [(0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALM_POLY = [0, 1, 2, 5, 9, 13, 17]


@dataclass
class HandMask:
    roi: tuple[int, int, int, int]   # x0, y0, x1, y1 (imagem)
    mask: np.ndarray                 # uint8 (y1-y0, x1-x0)
    source: str
    age: int = 0


def hand_roi(lm: np.ndarray, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    p = pad * max(x1 - x0, y1 - y0) + 4
    return (max(0, int(x0 - p)), max(0, int(y0 - p)), min(w, int(x1 + p) + 1), min(h, int(y1 + p) + 1))


# ------------------------------------------------------------- extratores sincronos
def landmarks_mask(hand, shape, finger_radius: float, kappa: float) -> HandMask:
    h, w = shape[:2]
    L = hand.scale(kappa)
    rad = max(2, int(finger_radius * L))
    x0, y0, x1, y1 = hand_roi(hand.lm, 0.0, w, h)
    x0, y0, x1, y1 = max(0, x0 - rad), max(0, y0 - rad), min(w, x1 + rad), min(h, y1 + rad)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = (hand.lm - [x0, y0]).astype(np.int32)
    cv2.fillPoly(m, [pts[PALM_POLY]], 255)              # palma, expandida (os landmarks ficam por dentro da borda)
    cv2.polylines(m, [pts[PALM_POLY]], True, 255, 2 * rad, cv2.LINE_AA)
    for chain in FINGERS:                                # dedos como "capsulas" ao longo de cada osso
        for a, b in zip(chain, chain[1:]):
            r = rad if b not in (4, 8, 12, 16, 20) else int(rad * 0.85)
            cv2.line(m, tuple(map(int, pts[a])), tuple(map(int, pts[b])), 255, 2 * r, cv2.LINE_AA)
            cv2.circle(m, tuple(map(int, pts[b])), r, 255, -1, cv2.LINE_AA)
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return HandMask((x0, y0, x1, y1), m, "landmarks")


# ------------------------------------------------------------- SAM (assincrono, GPU)
def _sam_worker(name, weights, device, q_in, q_out):
    """Processo separado: o CUDA do torch da access violation em threads secundarias (Py 3.15 + torch 2.14)."""
    try:
        from ultralytics import SAM
        model = SAM(weights)
        kw = dict(device=device, verbose=False, quantize="fp16")
        for _ in range(2):
            model(np.zeros((128, 128, 3), np.uint8), bboxes=[[16, 16, 100, 100]], points=[[[50, 50]]], labels=[[1]], **kw)
        q_out.put(("ready",))
        while True:
            item = q_in.get()
            if item is None:
                break
            fid, crop, boxes, pts = item
            t0 = time.perf_counter()
            # caixa + centro da palma (ponto positivo) por mao
            res = model(crop, bboxes=boxes, points=[[p] for p in pts], labels=[[1] for _ in pts], **kw)[0]
            xy = res.masks.xy if res.masks is not None else []
            polys = [xy[i] if i < len(xy) else None for i in range(len(boxes))]
            q_out.put(("res", fid, polys, (time.perf_counter() - t0) * 1000))
    except Exception as e:
        q_out.put(("error", f"{type(e).__name__}: {e}"))


class SamSegmenter:
    """Modelo da familia SAM (ultralytics) num processo a parte, assincrono. Prompt = caixa de cada mao (landmarks).
    Entre execucoes, a mascara e propagada pelo movimento dos landmarks (similaridade)."""

    def __init__(self, name: str, models_dir: str, device: str, pad: float, max_age: int, timeout: float = 180):
        import multiprocessing as mp
        from pathlib import Path
        self.name, self.pad, self.max_age = name, pad, max(max_age, _MIN_AGE.get(name, 0))
        wpath = Path(models_dir) / _SAM_WEIGHTS[name]
        weights = str(wpath) if wpath.exists() else _SAM_WEIGHTS[name]     # ultralytics baixa se faltar
        ctx = mp.get_context("spawn")
        self.q_in, self.q_out = ctx.Queue(maxsize=1), ctx.Queue()
        self.proc = ctx.Process(target=_sam_worker, args=(name, weights, device, self.q_in, self.q_out), daemon=True)
        self.proc.start()
        msg = self.q_out.get(timeout=timeout)       # espera carregar + aquecer
        if msg[0] != "ready":
            self.close()
            raise RuntimeError(msg[1] if len(msg) > 1 else "falha ao iniciar o SAM")
        self.pending: dict[int, list] = {}          # frame_id -> [(track, lm, roi)], crop origin
        self.results: dict[int, tuple[HandMask, np.ndarray, int]] = {}   # track -> (mask, lm_seg, frame_id)
        self.ms = 0.0
        self.error: str | None = None
        self._busy = False                          # ha um quadro sendo processado no worker

    def request(self, frame_id, image, hands):
        self._drain()
        if not hands or self._busy:   # worker ocupado: pula (pedir agora so faria o quadro esperar um ciclo a mais)
            return
        H, W = image.shape[:2]
        rois = [hand_roi(h.lm, self.pad, W, H) for h in hands]
        X0, Y0 = min(r[0] for r in rois), min(r[1] for r in rois)
        X1, Y1 = max(r[2] for r in rois), max(r[3] for r in rois)
        m = int(0.5 * max(X1 - X0, Y1 - Y0))   # contexto: com recorte justo o SAM devolve o recorte inteiro
        X0, Y0, X1, Y1 = max(0, X0 - m), max(0, Y0 - m), min(W, X1 + m), min(H, Y1 + m)
        crop = np.ascontiguousarray(image[Y0:Y1, X0:X1])
        boxes = [[r[0] - X0, r[1] - Y0, r[2] - X0, r[3] - Y0] for r in rois]
        pts = [(h.lm[[0, 5, 9, 13, 17]].mean(0) - [X0, Y0]).tolist() for h in hands]
        self.pending = {k: v for k, v in self.pending.items() if k > frame_id - 30}
        self.pending[frame_id] = ([(h.track_id, h.lm.copy(), r) for h, r in zip(hands, rois)], (X0, Y0))
        try:
            self.q_in.put_nowait((frame_id, crop, boxes, pts))
            self._busy = True
        except queue.Full:
            pass

    def _drain(self):
        while True:
            try:
                msg = self.q_out.get_nowait()
            except (queue.Empty, OSError, EOFError):
                break
            self._busy = False
            if msg[0] == "error":
                self.error = msg[1]
                continue
            _, fid, polys, ms = msg
            self.ms = ms
            info = self.pending.pop(fid, None)
            if info is None:
                continue
            hands, (X0, Y0) = info
            for (tid, lm, r), poly in zip(hands, polys):
                if poly is None or len(poly) < 3:
                    continue
                sub = np.zeros((r[3] - r[1], r[2] - r[0]), np.uint8)   # poligono (coords do crop) -> roi da mao
                cv2.fillPoly(sub, [(np.asarray(poly) - [r[0] - X0, r[1] - Y0]).astype(np.int32)], 255)
                self.results[tid] = (HandMask(r, sub, self.name), lm, fid)
        if not self.proc.is_alive() and self.error is None:
            self.error = "processo do SAM morreu"

    def get(self, hand, frame_id: int, shape) -> HandMask | None:
        self._drain()
        r = self.results.get(hand.track_id)
        if r is None:
            return None
        hm, lm_seg, fid = r
        age = frame_id - fid
        if age > self.max_age:
            return None
        if age == 0:
            return HandMask(hm.roi, hm.mask, hm.source, 0)
        M, inl = cv2.estimateAffinePartial2D(lm_seg.astype(np.float32), hand.lm.astype(np.float32))
        if M is None:
            return None
        x0, y0, x1, y1 = hm.roi
        corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], np.float32)
        nc = corners @ M[:, :2].T + M[:, 2]
        nx0, ny0 = (int(c) for c in np.floor(nc.min(0)))
        nx1, ny1 = (int(c) for c in np.ceil(nc.max(0)))
        nx0, ny0 = max(nx0, 0), max(ny0, 0)
        nx1, ny1 = min(nx1, shape[1]), min(ny1, shape[0])
        if nx1 <= nx0 or ny1 <= ny0:
            return None
        A = M.copy()   # roi antiga -> roi nova
        A[:, 2] = M[:, :2] @ np.array([x0, y0]) + M[:, 2] - [nx0, ny0]
        warped = cv2.warpAffine(hm.mask, A, (nx1 - nx0, ny1 - ny0), flags=cv2.INTER_NEAREST)
        return HandMask((nx0, ny0, nx1, ny1), warped, hm.source + "~", age)

    def close(self):
        try:
            self.q_in.put_nowait(None)
        except Exception:
            pass
        self.proc.join(timeout=2.0)
        if self.proc.is_alive():
            self.proc.terminate()


# ------------------------------------------------------------- perfil
@dataclass
class ToolProfile:
    r_near: np.ndarray      # (n_z,) m, raio da borda voltada para o eixo
    r_far: np.ndarray
    cover: np.ndarray       # bool
    weight: np.ndarray      # 0..1 (atenuacao nas pontas)
    centroid_r: float
    contour: np.ndarray     # (K,2) px na imagem - contorno real da mao (para desenhar)
    anchor: tuple | None = None   # px: ponto da mao mais perto do eixo (centro da correcao de profundidade)


def clean_mask(hm: HandMask) -> tuple[HandMask, np.ndarray] | None:
    """Maior componente, sem buracos, levemente suavizado. Devolve (mascara, contorno em px da imagem)."""
    cnts, _ = cv2.findContours(hm.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    m = np.zeros_like(hm.mask)
    cv2.drawContours(m, [c], -1, 255, -1)
    c = cv2.approxPolyDP(c, 1.0, True)
    return HandMask(hm.roi, m, hm.source, hm.age), c.reshape(-1, 2).astype(np.float64) + [hm.roi[0], hm.roi[1]]


def mask_row_spans(hm: HandMask, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada linha v (px da imagem): pixel mais a esquerda e mais a direita da mao (NaN se a linha nao cruza a mao).
    Nao e convexo: entre dedos separados a linha fica vazia -> o dedo deixa a sua marca."""
    x0, y0 = hm.roi[0], hm.roi[1]
    m = hm.mask > 0
    has = m.any(1)
    first = np.argmax(m, 1)
    last = m.shape[1] - 1 - np.argmax(m[:, ::-1], 1)
    r = np.round(v - y0).astype(int)
    ok = (r >= 0) & (r < m.shape[0])
    ok[ok] = has[r[ok]]
    umin = np.full(len(v), np.nan)
    umax = np.full(len(v), np.nan)
    umin[ok] = x0 + first[r[ok]]
    umax[ok] = x0 + last[r[ok]] + 1
    return umin, umax


def hand_bottom_over_vase(hm: HandMask, s: float, prof, r_top: float) -> tuple[float | None, float]:
    """Ponto mais baixo (z) da mao na faixa |x| <= r_top (sobre a boca do vaso) e z do centro da mao."""
    m = hm.mask > 0
    cols = m.any(0)
    if not cols.any():
        return None, float("nan")
    lowest = m.shape[0] - 1 - np.argmax(m[::-1, :], 0)            # linha mais baixa de cada coluna
    u = hm.roi[0] + np.arange(m.shape[1])
    lv = np.where(cols, lowest, -1)
    j = int(np.argmax(lv))                                         # ponto mais baixo da mao: centro da correcao
    anchor = (float(u[j]), float(hm.roi[1] + lv[j]))
    x, z = prof.to_vase(u, hm.roi[1] + lowest, s, anchor)
    sel = cols & (np.abs(x) <= r_top)
    ys, xs = np.nonzero(m)
    _, zc = prof.to_vase(hm.roi[0] + xs.mean(), hm.roi[1] + ys.mean(), s, anchor)
    return (float(z[sel].min()) if sel.any() else None), float(zc)


def feather(cover: np.ndarray, dz: float, width_m: float) -> np.ndarray:
    """Peso 0..1: rampa cosseno ate `width_m` de distancia das pontas da cobertura."""
    n = len(cover)
    idx = np.arange(n)
    unc = np.flatnonzero(~cover)
    if len(unc) == 0:
        return np.ones(n)
    if not cover.any():
        return np.zeros(n)
    pos = np.searchsorted(unc, idx)
    left = np.where(pos > 0, idx - unc[np.clip(pos - 1, 0, len(unc) - 1)], n)
    right = np.where(pos < len(unc), unc[np.clip(pos, 0, len(unc) - 1)] - idx, n)
    d = np.minimum(left, right) * dz
    w = 0.5 * (1 - np.cos(np.pi * np.clip(d / max(width_m, 1e-9), 0, 1)))
    return np.where(cover, w, 0.0)


def compute_profile(hm: HandMask, contour, edge: str, s: float, prof, H: float, n_z: int, feather_m: float) -> ToolProfile | None:
    z = np.linspace(0, H, n_z)
    c = np.asarray(contour, dtype=np.float64)
    a = c[np.argmin(np.abs(c[:, 0] - prof.axis_u))]               # ponto da mao mais perto do eixo (o contato)
    anchor = (float(a[0]), float(a[1]))
    v = prof.z_to_v(z, s, anchor)
    umin, umax = mask_row_spans(hm, v)
    cover = ~np.isnan(umin)
    if not cover.any():
        return None
    xmin, _ = prof.to_vase(np.nan_to_num(umin), v, s, anchor)
    xmax, _ = prof.to_vase(np.nan_to_num(umax), v, s, anchor)
    if edge == "R":
        rn, rf = xmin, xmax
    else:
        rn, rf = -xmax, -xmin
    rn = np.maximum(rn, 0.0)
    rf = np.maximum(rf, rn)
    w = feather(cover, H / (n_z - 1), feather_m)
    cr = float(np.mean((rn[cover] + rf[cover]) / 2))
    return ToolProfile(np.where(cover, rn, np.nan), np.where(cover, rf, np.nan), cover, w, cr, contour, anchor)


class ProfileEMA:
    def __init__(self, beta: float):
        self.beta = beta
        self.prev: dict[str, np.ndarray] = {}

    def __call__(self, edge: str, p: ToolProfile | None, enabled: bool) -> ToolProfile | None:
        if p is None:
            self.prev.pop(edge, None)
            return None
        old = self.prev.get(edge)
        if enabled and self.beta > 0 and old is not None:
            both = p.cover & ~np.isnan(old)
            p.r_near = p.r_near.copy()
            p.r_near[both] = self.beta * old[both] + (1 - self.beta) * p.r_near[both]
        self.prev[edge] = p.r_near.copy()
        return p
