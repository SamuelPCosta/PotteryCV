"""Calibracao por gestos (cabeca neutra -> maos de lado: escala + eixo + largura do vaso) e conversao pixel -> vaso."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

BASE_FRAC = 0.9     # base do vaso a 90% da altura da imagem (10% acima do rodape: mais facil de alcancar)


@dataclass
class CalibProfile:
    width: int
    height: int
    mirrored: bool = True
    axis_u: float = 0.0
    base_v: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k: float = 0.095 / 120     # m/px na profundidade de calibracao
    L_ref: dict = field(default_factory=dict)   # {"Left": px, "Right": px}
    kappa: dict = field(default_factory=dict)
    pitch0: float = 0.0
    yaw0: float = 0.0
    roll0: float = 0.0
    r0: float = 0.0            # raio inicial do vaso (m) medido pela distancia entre as maos; 0 = usa o config
    calibrated: bool = False
    created: str = ""

    @staticmethod
    def default(w: int, h: int) -> "CalibProfile":
        return CalibProfile(w, h, axis_u=w / 2, base_v=BASE_FRAC * h, cx=w / 2, cy=h / 2,   # base = BASE_FRAC da imagem
                            L_ref={"Left": 120.0, "Right": 120.0}, kappa={"Left": 1.3, "Right": 1.3})

    # ---- escala
    def L_ref_for(self, label: str) -> float:
        if label in self.L_ref:
            return self.L_ref[label]
        return float(np.mean(list(self.L_ref.values()))) if self.L_ref else 120.0

    def kappa_for(self, label: str, default=1.3) -> float:
        return self.kappa.get(label, default)

    # ---- conversoes (vetorizadas)
    # s = escala de profundidade da mao (> 1 = mais perto da camera). A correcao encolhe as coordenadas por s
    # em volta de `anchor` (px): no centro da imagem (None) e a perspectiva pura, mas a mao "escorrega" na tela;
    # no ponto de contato da propria mao, so a espessura e corrigida e o contato fica onde aparece.
    def _anchor(self, anchor):
        return (self.cx, self.cy) if anchor is None else anchor

    def to_vase(self, u, v, s=1.0, anchor=None):
        au, av = self._anchor(anchor)
        u = au + (np.asarray(u, dtype=np.float64) - au) / s
        v = av + (np.asarray(v, dtype=np.float64) - av) / s
        return self.k * (u - self.axis_u), self.k * (self.base_v - v)

    def z_to_v(self, z, s=1.0, anchor=None):
        av = self._anchor(anchor)[1]
        return av + s * (self.base_v - np.asarray(z) / self.k - av)

    def x_to_u(self, x, s=1.0, anchor=None):
        au = self._anchor(anchor)[0]
        return au + s * (np.asarray(x) / self.k + self.axis_u - au)

    # ---- arquivo
    def save(self, path):
        self.created = time.strftime("%Y-%m-%dT%H:%M:%S")
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path, w: int, h: int, mirrored: bool) -> tuple["CalibProfile", str | None]:
        d = json.loads(Path(path).read_text())
        known = {f.name for f in fields(CalibProfile)}
        p = CalibProfile(**{k: v for k, v in d.items() if k in known})   # ignora chaves de versoes antigas
        p.base_v = BASE_FRAC * p.height   # a base do vaso e fixa (nao calibrada): BASE_FRAC da imagem
        warn = None
        if (p.width, p.height) != (w, h) or p.mirrored != mirrored:
            warn = f"perfil feito em {p.width}x{p.height} espelho={p.mirrored}; câmera atual {w}x{h} espelho={mirrored} — recalibre"
        return p, warn


class _Dwell:
    """Aceita quando o valor fica estavel (desvio < thr) por dwell_s segundos."""

    def __init__(self, dwell_s: float, thr: float):
        self.dwell, self.thr = dwell_s, thr
        self.buf: list[tuple[float, np.ndarray, object]] = []

    def reset(self):
        self.buf.clear()

    def push(self, t: float, value: np.ndarray, payload=None) -> bool:
        self.buf.append((t, np.asarray(value, dtype=np.float64), payload))
        vals = np.stack([b[1] for b in self.buf])
        if np.abs(vals - vals.mean(0)).max() > 2 * self.thr:   # mexeu: reinicia do quadro atual
            self.buf = [self.buf[-1]]
            return False
        return self.buf[-1][0] - self.buf[0][0] >= self.dwell

    def progress(self) -> float:
        if len(self.buf) < 2:
            return 0.0
        return min(1.0, (self.buf[-1][0] - self.buf[0][0]) / self.dwell)


STEPS = ["HEAD", "HANDS"]
INSTR = {
    "HEAD": "1/2  Olhe para o centro da tela, cabeca relaxada",
    "HANDS": "2/2  Maos de lado, dedos para cima, na LARGURA do vaso (o meio vira o eixo)",
}


class Calibrator:
    """Cada etapa: mostra a instrucao + contagem regressiva ("capturando em 3.. 2.. 1"), depois captura
    quando a pose fica parada por dwell_s."""

    def __init__(self, cfg_calib, cfg_hand, w: int, h: int, mirrored: bool, base: CalibProfile | None = None,
                 r_limits: tuple[float, float] = (0.01, 0.16)):
        self.cc, self.ch = cfg_calib, cfg_hand
        self.r_limits = r_limits
        self.prof = CalibProfile.default(w, h)
        if base is not None:            # mantem o alinhamento antigo visivel enquanto recalibra
            self.prof.axis_u, self.prof.k = base.axis_u, base.k
            self.prof.L_ref, self.prof.kappa = dict(base.L_ref), dict(base.kappa)
        self.prof.mirrored = mirrored
        self.step = 0
        self.phase = "COUNTDOWN"
        self.t_phase: float | None = None
        self.t_now = 0.0
        self.hint = ""
        self.dw = _Dwell(cfg_calib.dwell_s, cfg_calib.still_px)

    @property
    def name(self):
        return STEPS[self.step] if self.step < len(STEPS) else "DONE"

    @property
    def instruction(self):
        return INSTR.get(self.name, "")

    def countdown_left(self) -> float:
        if self.phase != "COUNTDOWN" or self.t_phase is None:
            return self.cc.countdown_s if self.phase == "COUNTDOWN" else 0.0
        return max(0.0, self.cc.countdown_s - (self.t_now - self.t_phase))

    def progress(self):
        return self.dw.progress() if self.phase == "CAPTURE" else 0.0

    def update(self, hands, head, image_bgr, t: float) -> CalibProfile | None:
        self.t_now = t
        if self.t_phase is None:
            self.t_phase = t
        if self.phase == "COUNTDOWN":
            self.hint = ""
            if t - self.t_phase >= self.cc.countdown_s:
                self.phase = "CAPTURE"
                self.dw.reset()
            return None
        name = self.name
        if name == "HEAD":
            if not head.ok:
                self.hint = "rosto não detectado"
                self.dw.reset()
                return None
            self.hint = "segure parado…"
            self.dw.thr = self.cc.still_deg
            if self.dw.push(t, [head.pitch, head.yaw, head.roll]):
                v = np.median(np.stack([b[1] for b in self.dw.buf]), 0)
                self.prof.pitch0, self.prof.yaw0, self.prof.roll0 = map(float, v)
                self._next()
            return None
        if len(hands) != 2:
            self.hint = f"preciso ver as duas mãos ({len(hands)} visível)"
            self.dw.reset()
            return None
        self.hint = "segure parado…"
        hs = sorted(hands, key=lambda h: h.center[0])           # [esquerda da tela, direita da tela]
        vec = np.concatenate([hs[0].center, hs[1].center])
        self.dw.thr = self.cc.still_px
        # borda interna de cada mao (a voltada para o vaso) + comprimento pulso -> no do medio (visivel de lado)
        payload = [(h.handedness, h.d09) for h in hs] + [float(hs[0].lm[:, 0].max()), float(hs[1].lm[:, 0].min())]
        if not self.dw.push(t, vec, payload):
            return None
        # HANDS: escala + eixo + largura do vaso numa captura so
        per: dict[str, list] = {}
        inner_l, inner_r = [], []
        for _, _, pl in self.dw.buf:
            for lab, d09 in pl[:2]:
                per.setdefault(lab, []).append(d09)
            inner_l.append(pl[2])
            inner_r.append(pl[3])
        for lab, arr in per.items():
            self.prof.L_ref[lab] = float(np.median(arr))
            self.prof.kappa.setdefault(lab, self.ch.kappa_default)   # largura da palma nao aparece de lado
        L = float(np.mean(list(self.prof.L_ref.values())))
        self.prof.k = self.ch.L_real_m / L
        rad = self.ch.finger_radius * L                              # landmarks ficam no centro dos dedos
        il, ir = float(np.median(inner_l)) + rad, float(np.median(inner_r)) - rad
        self.prof.axis_u = (il + ir) / 2
        self.prof.base_v = BASE_FRAC * self.prof.height
        lo, hi = self.r_limits
        self.prof.r0 = float(np.clip(self.prof.k * (ir - il) / 2, lo, hi))
        self.prof.calibrated = True
        self._next()
        return self.prof

    def _next(self):
        self.step += 1
        self.phase, self.t_phase = "COUNTDOWN", None
        self.dw = _Dwell(self.cc.dwell_s, self.cc.still_px)
