"""Velocidade do torno pela cabeca: um pouco para cima (> brake_deg) -> para;
para baixo (> accel_deg) -> vai acelerando; entre os dois -> mantem."""
from __future__ import annotations

import math

from .tracking import OneEuro

RPM = 2 * math.pi / 60.0   # rpm -> rad/s


class SpeedController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.locked = False
        self.rpm = cfg.min_rpm
        self.pitch0 = self.yaw0 = 0.0
        self._f = OneEuro(min_cutoff=1.0, beta=0.05)
        self._dwell = 0.0
        self._last_sign = 0
        self.delta = 0.0      # graus; > 0 = olhando para baixo (apos pitch_sign)
        self.valid = False
        self.status = ""      # so avisos (trava, tracking perdido); vazio no uso normal

    def set_neutral(self, pitch0: float, yaw0: float):
        self.pitch0, self.yaw0 = pitch0, yaw0

    def zone(self, d: float) -> int:
        """+1 acelera (baixo), -1 freia (cima), 0 mantem."""
        if d > self.cfg.accel_deg:
            return 1
        if d < -self.cfg.brake_deg:
            return -1
        return 0

    def update(self, head, t: float, dt: float) -> float:
        """Retorna omega (rad/s)."""
        c = self.cfg
        self.valid = bool(head is not None and head.ok and abs(head.yaw - self.yaw0) <= c.yaw_gate_deg)
        if self.locked:
            self.status = "trava"
            return self.rpm * RPM
        if not self.valid:
            self.status = "Tracking perdido" if (head is None or not head.ok) else "rosto virado"
            self._dwell = 0.0
            return self.rpm * RPM
        d = float(self._f(c.pitch_sign * (head.pitch - self.pitch0), t))
        self.delta = d
        sign = self.zone(d)
        self._dwell = self._dwell + dt if (sign != 0 and sign == self._last_sign) else 0.0
        self._last_sign = sign
        self.status = ""
        if sign == 0 or self._dwell < c.dwell_s:          # neutra (ou olhada curta): mantem
            pass
        elif sign > 0:                                     # para baixo: acelera, mais rapido quanto mais inclina
            e = min(1.0, (d - c.accel_deg) / (c.sat_deg - c.accel_deg)) ** c.gamma
            self.rpm = min(c.max_rpm, self.rpm + c.accel_rpm_s * max(e, 0.15) * dt)
        else:                                              # para cima: freia ate o minimo
            self.rpm = max(c.min_rpm, self.rpm - c.brake_rpm_s * dt)
        return self.rpm * RPM
