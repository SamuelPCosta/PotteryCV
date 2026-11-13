"""Gerador de vaso sintético animado (sem câmera) — testa o receptor do Blender e o viewer.

    python tools/synthetic_sender.py                    # 128x128, 30 Hz, Blender :9000 + viewer :9002
    python tools/synthetic_sender.py --drop 0.2 --pattern bulge --height-wave
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pottery.net import F_SYNTH, Sender  # noqa: E402


def make_R(pattern: str, n_t: int, n_z: int, t: float, r0: float) -> np.ndarray:
    th = 2 * np.pi * np.arange(n_t)[:, None] / n_t
    z = np.linspace(0, 1, n_z)[None]
    if pattern == "static":
        R = r0 * (1 + 0.3 * np.sin(np.pi * z) - 0.35 * np.exp(-((z - 0.8) / 0.08) ** 2))
    elif pattern == "bulge":
        R = r0 * (1 + 0.45 * np.sin(np.pi * z) * (0.6 + 0.4 * np.sin(t)) - 0.4 * np.exp(-((z - 0.85) / 0.07) ** 2))
    elif pattern == "twist":
        R = r0 * (1 + 0.15 * np.sin(3 * th + 4 * z + t))
    else:  # wave
        R = r0 * (1 + 0.25 * np.sin(2 * np.pi * z * 1.5 + t) + 0.08 * np.sin(3 * th) * np.sin(t / 2))
    return np.broadcast_to(R, (n_t, n_z)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="128x128", help="N_theta x N_z")
    ap.add_argument("--rate", type=float, default=30)
    ap.add_argument("--rpm", type=float, default=30)
    ap.add_argument("--pattern", default="wave", choices=["wave", "bulge", "twist", "static"])
    ap.add_argument("--r0", type=float, default=0.06)
    ap.add_argument("--height", type=float, default=0.18)
    ap.add_argument("--height-wave", action="store_true", help="altura oscila (imita o vaso subindo ao ser apertado)")
    ap.add_argument("--drop", type=float, default=0.0, help="fração de pacotes descartados")
    ap.add_argument("--max-datagram", type=int, default=65000, help="menor => força mais blocos")
    ap.add_argument("--targets", nargs="+", default=["127.0.0.1:9000", "127.0.0.1:9002"])
    ap.add_argument("--seconds", type=float, default=0, help="0 = infinito")
    a = ap.parse_args()
    nt, nz = map(int, a.res.lower().split("x"))
    snd = Sender(a.targets, a.rate, echo_port=9001, max_datagram=a.max_datagram)
    snd.drop = a.drop
    omega = a.rpm * 2 * math.pi / 60
    t0 = time.perf_counter()
    phi, last, n = 0.0, t0, 0
    lat = []
    print(f"enviando {nt}x{nz} @ {a.rate} Hz para {a.targets} (Ctrl+C para parar)")
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if a.seconds and t > a.seconds:
                break
            phi += omega * (now - last)
            last = now
            H = a.height * (1 + 0.25 * math.sin(t * 0.7)) if a.height_wave else a.height
            R = make_R(a.pattern, nt, nz, t, a.r0)
            snd.maybe_send(R, H, phi, omega, n, time.time_ns(), n, F_SYNTH, force=True)
            n += 1
            for e in snd.poll_echo():
                lat.append((e["t_applied_ns"] - e["t_capture_ns"]) / 1e6)
            if n % int(a.rate * 2) == 0:
                extra = f"  eco: {len(lat)} (mediana capt→aplicado {np.median(lat):.1f} ms)" if lat else ""
                print(f"t={t:6.1f}s  estados {n}  H={H * 100:.1f}cm{extra}")
                lat.clear()
            time.sleep(max(0.0, 1 / a.rate - (time.perf_counter() - now)))
    except KeyboardInterrupt:
        pass
    snd.close()


if __name__ == "__main__":
    main()
