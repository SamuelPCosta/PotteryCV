"""Interface Qt: webcam+overlay, preview 3D do vaso, mapa R desenrolado e painel de depuração."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget)

from .contour import EXTRACTORS

try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception:   # sem OpenGL: fallback 2D
    _HAS_GL = False


def to_pixmap(bgr: np.ndarray, size) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi).scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def vase_mesh(n_theta: int, n_z: int) -> np.ndarray:
    """Faces da parede externa (θ por fora, z por dentro, com costura) + fundo em leque. Mesma ordem do Blender."""
    i = np.arange(n_theta)[:, None]
    j = np.arange(n_z - 1)[None]
    a = i * n_z + j
    b = ((i + 1) % n_theta) * n_z + j
    quads = np.stack([np.stack([a, b, b + 1], -1), np.stack([a, b + 1, a + 1], -1)], 0).reshape(-1, 3)
    c = n_theta * n_z
    k = np.arange(n_theta)
    fan = np.stack([((k + 1) % n_theta) * n_z, k * n_z, np.full(n_theta, c)], 1)
    return np.concatenate([quads, fan]).astype(np.int32)


def vase_vertices(R: np.ndarray, H: float, phi: float, hf: np.ndarray | None = None) -> np.ndarray:
    """hf: altura relativa de cada coluna (0..1 × H); None = topo uniforme."""
    n_theta, n_z = R.shape
    th = 2 * np.pi * np.arange(n_theta) / n_theta + phi
    hc = H * (np.ones(n_theta) if hf is None else np.asarray(hf))
    v = np.empty((n_theta, n_z, 3), np.float32)
    v[..., 0] = R * np.cos(th)[:, None]
    v[..., 1] = R * np.sin(th)[:, None]
    v[..., 2] = hc[:, None] * np.linspace(0, 1, n_z)[None]
    return np.concatenate([v.reshape(-1, 3), np.zeros((1, 3), np.float32)])


def vase_normals(verts: np.ndarray, n_theta: int, n_z: int) -> np.ndarray:
    """Normais por vértice direto da grade (o cálculo do pyqtgraph leva ~200 ms)."""
    P = verts[:-1].reshape(n_theta, n_z, 3)
    dth = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    dz = np.gradient(P, axis=1)
    n = np.cross(dth, dz)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    return np.concatenate([n.reshape(-1, 3), [[0, 0, -1]]]).astype(np.float32)


def vase_meshdata(R, H, phi, faces, hf=None):
    v = vase_vertices(R, H, phi, hf)
    md = gl.MeshData(vertexes=v, faces=faces)
    md._vertexNormals = vase_normals(v, *R.shape)   # evita o cálculo lento interno
    return md


class Preview3D(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.shape = None
        if _HAS_GL:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor((38, 38, 42))
            self.view.setCameraPosition(distance=0.65, elevation=18, azimuth=-90)   # câmera em −Y, como no Blender
            g = gl.GLGridItem()
            g.setSize(0.6, 0.6)
            g.setSpacing(0.05, 0.05)
            self.view.addItem(g)
            self.mesh = None
            lay.addWidget(self.view)
        else:
            self.view = QLabel("OpenGL indisponível — preview 2D")
            self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(self.view)

    def update_vase(self, R: np.ndarray, H: float, phi: float, hf: np.ndarray | None = None):
        if not _HAS_GL:
            self._fallback(R, H, hf)
            return
        if self.shape != R.shape:
            self.shape = R.shape
            self.faces = vase_mesh(*R.shape)
            if self.mesh is not None:
                self.view.removeItem(self.mesh)
            self.mesh = gl.GLMeshItem(meshdata=vase_meshdata(R, H, phi, self.faces, hf), smooth=True,
                                      shader="shaded", color=(0.78, 0.50, 0.34, 1.0), glOptions="opaque")
            self.view.addItem(self.mesh)
        else:
            self.mesh.setMeshData(meshdata=vase_meshdata(R, H, phi, self.faces, hf), smooth=True)
        self.view.opts["center"] = pg_vector(0, 0, H / 2)

    def _fallback(self, R, H, hf=None):
        img = np.full((300, 300, 3), 40, np.uint8)
        sc = 250 / max(H, 2 * R.max(), 1e-3)
        hf = np.ones(R.shape[0]) if hf is None else hf
        for col, sg in ((0, 1), (R.shape[0] // 2, -1)):
            z = H * hf[col] * np.linspace(0, 1, R.shape[1])
            pts = np.stack([150 + sg * R[col] * sc, 280 - z * sc], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, (60, 110, 200), 2)
        self.view.setPixmap(to_pixmap(img, self.view.size()))


def pg_vector(x, y, z):
    from pyqtgraph import Vector
    return Vector(x, y, z)


class WheelIndicator(QLabel):
    """Seta circular (assets/wheel_arrow.png) recortada em círculo, girando com o ângulo do torno."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background:#111")
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "assets" / "wheel_arrow.png"
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.full((200, 200, 3), 255, np.uint8)
        elif img.ndim == 3 and img.shape[2] == 4:          # PNG transparente -> sobre fundo branco
            a = img[:, :, 3:4].astype(np.float32) / 255
            img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        s = min(img.shape[:2])
        y, x = (img.shape[0] - s) // 2, (img.shape[1] - s) // 2
        self.base = cv2.resize(img[y:y + s, x:x + s], (220, 220), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[:220, :220]
        self.circle = ((xx - 109.5) ** 2 + (yy - 109.5) ** 2) <= 108 ** 2

    def update_angle(self, phi: float, rpm: float):
        M = cv2.getRotationMatrix2D((109.5, 109.5), math.degrees(phi), 1.0)   # positivo = anti-horário na tela
        rot = cv2.warpAffine(self.base, M, (220, 220), borderValue=(255, 255, 255))
        pad = 60                                             # margem lateral e superior
        out = np.full((220 + 110, 220 + 2 * pad, 3), 17, np.uint8)   # altura extra embaixo para o texto
        disc = out[15:235, pad:pad + 220]
        disc[self.circle] = rot[self.circle]
        cv2.circle(out, (pad + 110, 125), 108, (60, 110, 200), 2, cv2.LINE_AA)
        txt = f"{rpm:.0f} rpm"
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0][0]
        cv2.putText(out, txt, (pad + 110 - tw // 2, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (230, 230, 230), 2, cv2.LINE_AA)
        self.setPixmap(to_pixmap(out, self.size()))


def rmap_image(R: np.ndarray, r_max: float, cols: dict | None) -> np.ndarray:
    img = (np.clip(R.T[::-1] / r_max, 0, 1) * 255).astype(np.uint8)   # z para cima, θ na horizontal
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    img = cv2.resize(img, (R.shape[0] * 3, R.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
    for e, c in (cols or {}).items():
        x = c * 3 + 1
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 140, 255) if e == "R" else (255, 140, 0), 1)
    return img


class MainWindow(QMainWindow):
    def __init__(self, cfg, pipeline=None, receiver=None):
        super().__init__()
        self.cfg, self.pipe, self.rx = cfg, pipeline, receiver
        self.setWindowTitle("PotteryCV — " + ("viewer" if receiver else "torno virtual"))
        self.resize(1500, 900)
        self.video = QLabel("aguardando câmera…")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video.setStyleSheet("background:#111;color:#aaa")
        self.preview = Preview3D()
        self.rmap = QLabel()
        self.rmap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rmap.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.rmap.setStyleSheet("background:#111")
        self.debug = QLabel()
        self.debug.setFont(QFont("Consolas", 9))
        self.debug.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.debug.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.wheel = WheelIndicator()
        dbg_row = QWidget()
        dl = QHBoxLayout(dbg_row)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(8)
        dl.addWidget(self.rmap, 3)
        dl.addWidget(self.wheel, 1)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.preview)
        right.addWidget(dbg_row)
        right.setSizes([600, 250])
        top = QSplitter(Qt.Orientation.Horizontal)
        if pipeline is not None:
            top.addWidget(self.video)
        top.addWidget(right)
        top.setSizes([950, 550] if pipeline is not None else [1])
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.addWidget(self.debug, 3)
        if pipeline is not None:
            bl.addWidget(self._controls(), 4)
        root = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(top)
        root.addWidget(bottom)
        root.setSizes([700, 200])
        self.setCentralWidget(root)

        if pipeline is not None:
            pipeline.result.connect(self.on_result)
        if receiver is not None:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll_receiver)
            self.timer.start(16)
        QShortcut(QKeySequence("Q"), self, self.close)

    # ---------------------------------------------------------------- controles
    def _controls(self) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        cmd = self.pipe.command

        def button(text, name, checkable=False):
            b = QPushButton(text)
            b.setCheckable(checkable)
            if checkable:
                b.toggled.connect(lambda v: cmd(name, v))
            else:
                b.clicked.connect(lambda: cmd(name))
            return b

        self.b_rec = button("Gravar sessão", "record", True)
        g.addWidget(button("Calibrar [C]", "calibrate"), 0, 0)
        g.addWidget(button("Cancelar calibração", "cancel_calibration"), 0, 1)
        g.addWidget(button("Reset vaso [R]", "reset"), 0, 2)
        g.addWidget(self.b_rec, 0, 3)
        dbg = QCheckBox("Debug [D]")
        dbg.setToolTip("mostra os landmarks numerados e o medidor de inclinação da cabeça")
        dbg.toggled.connect(lambda v: cmd("show_landmarks", v))
        g.addWidget(dbg, 0, 4)
        QShortcut(QKeySequence("D"), self, dbg.toggle)

        g.addWidget(QLabel("Extrator"), 1, 0)
        ex = QComboBox()
        ex.addItems(EXTRACTORS)
        ex.setCurrentText(self.cfg.seg.extractor)
        ex.currentTextChanged.connect(lambda v: cmd("extractor", v))
        g.addWidget(ex, 1, 1)

        g.addWidget(QLabel("Velocidade: cabeça ↓ acelera · ↑ para"), 2, 0, 1, 3)
        lock = QCheckBox("Trava velocidade [L]")
        lock.toggled.connect(lambda v: cmd("lock", v))
        g.addWidget(lock, 2, 3, 1, 2)
        QShortcut(QKeySequence("L"), self, lock.toggle)

        sm = QCheckBox("Suavização")
        sm.setChecked(self.cfg.hand.smoothing)
        sm.toggled.connect(lambda v: cmd("smoothing", v))
        dn = QCheckBox("Normalização de profundidade")
        dn.setChecked(self.cfg.hand.depth_norm)
        dn.toggled.connect(lambda v: cmd("depth_norm", v))
        g.addWidget(sm, 5, 0)
        g.addWidget(dn, 5, 1, 1, 2)
        lbl = lambda v: f"subida ao apertar dos lados: {v:.0%}  (0 = altura fixa, 100% = conserva volume)"
        self.vg_lbl = QLabel(lbl(self.cfg.vase.side_gain))
        vg = QSlider(Qt.Orientation.Horizontal)
        vg.setRange(0, 100)
        vg.setValue(int(self.cfg.vase.side_gain * 100))
        vg.valueChanged.connect(lambda v: (cmd("side_gain", v / 100), self.vg_lbl.setText(lbl(v / 100))))
        g.addWidget(self.vg_lbl, 3, 0, 1, 5)
        g.addWidget(vg, 4, 0, 1, 5)
        lbl2 = lambda v: f"espalhamento da marca: {v:.0%}  (0 = marca seca do dedo, 100% = bem espalhada)"
        self.sp_lbl = QLabel(lbl2(self.cfg.vase.smooth_lambda))
        sp = QSlider(Qt.Orientation.Horizontal)
        sp.setRange(0, 100)
        sp.setValue(int(self.cfg.vase.smooth_lambda * 100))
        sp.valueChanged.connect(lambda v: (cmd("smooth_lambda", v / 100), self.sp_lbl.setText(lbl2(v / 100))))
        g.addWidget(self.sp_lbl, 6, 0, 1, 5)
        g.addWidget(sp, 7, 0, 1, 5)

        QShortcut(QKeySequence("R"), self, lambda: cmd("reset"))
        QShortcut(QKeySequence("C"), self, lambda: cmd("calibrate"))
        return w

    # ---------------------------------------------------------------- atualização
    def on_result(self, r: dict):
        if "fatal" in r:
            self.debug.setText(f"ERRO FATAL: {r['fatal']}")
            return
        if "image" not in r:
            self.debug.setText(r.get("message", ""))
            return
        self.video.setPixmap(to_pixmap(r["image"], self.video.size()))
        self.preview.update_vase(r["R"], r["H"], r["phi"], r.get("Hf"))
        self.rmap.setPixmap(to_pixmap(rmap_image(r["R"], self.cfg.vase.r_max, r["cols"]), self.rmap.size()))
        self.wheel.update_angle(r["phi"], r["rpm"])
        ms = r["ms"]
        pen = {e: (f"{v * 1000:+.1f}mm" if v == v else "  —  ") for e, v in r["pen"].items()}
        self.debug.setText(
            f"estado   {r['state']}   {'● GRAVANDO' if r['recording'] else ''}\n"
            f"calib    {r['calib_step']}\n"
            f"FPS      {r['fps']:5.1f}   mãos {r['n_hands']}\n"
            f"ms       mãos {ms['hands']:5.1f}  rosto {ms['face']:5.1f}  contorno {ms['contour']:5.1f}  sim {ms['sim']:5.1f}  total {ms['total']:5.1f}\n"
            f"torno    {r['rpm']:5.1f} rpm  ({r['speed_status']})   cabeça {r['pitch_delta']:+5.1f}° "
            f"({'↓ baixo' if r['pitch_delta'] > 0 else '↑ cima'})  rosto {'ok' if r['head_ok'] else '—'}\n"
            f"vaso     H {r['H'] * 100:5.1f} cm   V/V0 {r['volume']:.3f}   φ {math.degrees(r['phi']) % 360:5.1f}°\n"
            f"contato  L {r['contacts']['L']:<5} {pen['L']}   R {r['contacts']['R']:<5} {pen['R']}\n"
            f"extrator {r['extractor']}   idade máscara {r['seg_age']}   SAM {r['seg_ms']:.0f} ms\n"
            f"msg      {r['message']}")

    def poll_receiver(self):
        if self.rx.poll() and self.rx.R is not None:
            m = self.rx.meta
            self.preview.update_vase(self.rx.R, m["height"], m["phi"], self.rx.HF)
            self.rmap.setPixmap(to_pixmap(rmap_image(self.rx.R, self.cfg.vase.r_max, None), self.rmap.size()))
            self.wheel.update_angle(m["phi"], m["omega"] * 60 / (2 * math.pi))
            lat = (m["t_recv_ns"] - m["t_capture_ns"]) / 1e6 if m["t_capture_ns"] else float("nan")
            self.debug.setText(f"viewer UDP :{self.cfg.net.viewer_port}   pacotes {self.rx.packets}   seq {m['seq']}\n"
                               f"R {self.rx.R.shape}   H {m['height'] * 100:.1f} cm   φ {math.degrees(m['phi']):.1f}°   "
                               f"ω {m['omega'] * 60 / (2 * math.pi):.1f} rpm\nlatência captura→recepção {lat:.1f} ms")

    def closeEvent(self, ev):
        if self.pipe is not None:
            self.pipe.stop()
        if self.rx is not None:
            self.rx.close()
        super().closeEvent(ev)
