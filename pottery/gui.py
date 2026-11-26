"""Interface Qt: webcam+overlay, preview 3D do vaso, mapa R desenrolado e painel de depuracao."""
from __future__ import annotations

import math
import os
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget)

from .contour import EXTRACTORS
from .net import VIEW_JPEG

ACCENT = "#007788"        # cor de destaque: botoes, caixinhas marcadas e seletor de extrator
CHECK_ICON = (Path(__file__).resolve().parents[1] / "assets" / "check.svg").as_posix()   # tique das caixinhas

try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception:   # sem OpenGL: fallback 2D
    _HAS_GL = False


def to_pixmap(bgr: np.ndarray, size, keep_aspect: bool = True) -> QPixmap:
    """BGR (3 canais) ou BGRA pre-multiplicado (4 canais, com transparencia) -> QPixmap."""
    h, w = bgr.shape[:2]
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        rgba = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA))
        qi = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888_Premultiplied).copy()
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    mode = Qt.AspectRatioMode.KeepAspectRatio if keep_aspect else Qt.AspectRatioMode.IgnoreAspectRatio
    return QPixmap.fromImage(qi).scaled(size, mode, Qt.TransformationMode.SmoothTransformation)


def vase_mesh(n_theta: int, n_z: int) -> np.ndarray:
    """Faces da parede externa (theta por fora, z por dentro, com costura) + fundo em leque. Mesma ordem do Blender."""
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
    """hf: altura relativa de cada coluna (0..1 x H); None = topo uniforme."""
    n_theta, n_z = R.shape
    th = 2 * np.pi * np.arange(n_theta) / n_theta + phi
    hc = H * (np.ones(n_theta) if hf is None else np.asarray(hf))
    v = np.empty((n_theta, n_z, 3), np.float32)
    v[..., 0] = R * np.cos(th)[:, None]
    v[..., 1] = R * np.sin(th)[:, None]
    v[..., 2] = hc[:, None] * np.linspace(0, 1, n_z)[None]
    return np.concatenate([v.reshape(-1, 3), np.zeros((1, 3), np.float32)])


def vase_normals(verts: np.ndarray, n_theta: int, n_z: int) -> np.ndarray:
    """Normais por vertice direto da grade (o calculo do pyqtgraph leva ~200 ms)."""
    P = verts[:-1].reshape(n_theta, n_z, 3)
    dth = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    dz = np.gradient(P, axis=1)
    n = np.cross(dth, dz)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    return np.concatenate([n.reshape(-1, 3), [[0, 0, -1]]]).astype(np.float32)


def vase_meshdata(R, H, phi, faces, hf=None):
    v = vase_vertices(R, H, phi, hf)
    md = gl.MeshData(vertexes=v, faces=faces)
    md._vertexNormals = vase_normals(v, *R.shape)   # evita o calculo lento interno
    return md

def pg_vector(x, y, z):
    from pyqtgraph import Vector
    return Vector(x, y, z)

def rmap_image(R: np.ndarray, r_max: float, cols: dict | None) -> np.ndarray:
    img = (np.clip(R.T[::-1] / r_max, 0, 1) * 255).astype(np.uint8)   # z para cima, theta na horizontal
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    img = cv2.resize(img, (R.shape[0] * 3, R.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
    for e, c in (cols or {}).items():
        x = c * 3 + 1
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 140, 255) if e == "R" else (255, 140, 0), 1)
    return img


def debug_html(rows) -> str:
    """Tabela rotulo | valor: o valor quebra linha e as linhas seguintes ficam alinhadas na coluna do valor."""
    tr = "".join(f"<tr><td width='1%' style='white-space:pre; padding-right:10px'>{escape(k)}</td>"
                 f"<td style='white-space:pre-wrap'>{escape(v)}</td></tr>" for k, v in rows)
    return f"<table width='100%' cellspacing='0' cellpadding='0'>{tr}</table>"


def titled(title: str, widget: QWidget, pad: int = 0) -> QWidget:
    """Painel com cabecalho "Titulo ---------" em cima do widget; title vazio = so o widget.
    pad: margem interna entre o painel e o conteudo, em px."""
    keep = QFont(widget.font())                      # fonte original do conteudo
    keep.setPointSizeF(keep.pointSizeF())            # marca tamanho e peso como definidos: senao o Qt
    keep.setBold(keep.bold())                        # ainda herda do painel (fonte "padrao" nao conta)
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(pad, pad, pad, pad)
    if title:
        head = QHBoxLayout()
        head.setSpacing(20)                          # 20 px entre o texto e a linha
        lbl = QLabel(title)
        f = lbl.font()
        f.setPointSizeF(f.pointSizeF() * 1.6)
        f.setBold(True)
        lbl.setFont(f)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background:#555")
        head.addWidget(lbl)
        head.addWidget(line, 1, Qt.AlignmentFlag.AlignVCenter)   # estica ate a direita, no meio do texto
        lay.addLayout(head)
    lay.addWidget(widget, 1)
    widget.setFont(keep)
    return box


class MainWindow(QMainWindow):
    def __init__(self, cfg, pipeline=None, receiver=None):
        super().__init__()
        self.setStyleSheet("QWidget { background:#111; color:#aaa; }")
        self.cfg, self.pipe, self.rx = cfg, pipeline, receiver
        self.setWindowTitle("PotteryCV — " + ("viewer" if receiver else "Torno virtual"))
        self.resize(1500, 900)
        self.video = QLabel("Aguardando câmera…")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video.setStyleSheet("background:#000;color:#aaa")
        self.rmap = QLabel()
        self.rmap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rmap.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.debug = QLabel()
        self.debug.setFont(QFont("Consolas", 9))
        self.debug.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.debug.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.debug.setWordWrap(True)                  # linhas longas quebram (ver debug_html)
        self.bview = QLabel("Aguardando o Blender…")  # vista da camera do Blender (JPEG que o receptor grava)
        self.bview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.bview.setStyleSheet("background:#000;color:#aaa")
        self._bview_path = str(Path(__file__).resolve().parents[1] / "data" / VIEW_JPEG)
        self._bview_mtime = 0.0
        self._bview_img: np.ndarray | None = None
        self._bview_size = QSize()

        left = QSplitter(Qt.Orientation.Vertical)
        if pipeline is not None:
            left.addWidget(titled("", self.video))
        left.addWidget(titled("", self.rmap))
        left.setSizes([515, 335])

        right = QSplitter(Qt.Orientation.Vertical)    # splitter organiza os filhos sozinho: sem layout
        right.setContentsMargins(0, 12, 0, 0)         # margem no topo, acima de Controles
        if pipeline is not None:
            right.addWidget(titled("Controles", self._controls(), pad=16))
        self.bview_panel = titled("Blender", self.bview, pad=16)
        self.bview_panel.hide()                       # aparece quando "Vista do Blender" e marcada
        right.addWidget(self.bview_panel)
        right.addWidget(titled("Debug", self.debug, pad=16))
        right.setSizes([250, 350, 600])
        right.setFixedWidth(550)                      # painel da direita sempre com 250 px
        
        root = QSplitter(Qt.Orientation.Horizontal)
        root.setHandleWidth(4)
        # linha clara mostra onde arrastar; precisa ser no proprio splitter (no da janela o QWidget{} ganha)
        root.setStyleSheet("QSplitter::handle:horizontal { background: #666; }"
                           "QSplitter::handle:horizontal:hover { background: #aaa; }")
        root.addWidget(left)
        root.addWidget(right)
        root.setStretchFactor(0, 1)                   # ao redimensionar a janela, so a esquerda cresce
        root.setStretchFactor(1, 0)
        root.setSizes([10000, 1])                     # direita comeca no tamanho minimo dela
        self.setCentralWidget(root)

        if pipeline is not None:
            pipeline.result.connect(self.on_result)
        if receiver is not None:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll_receiver)
            self.timer.start(16)
        self.bview_timer = QTimer(self)               # confere o JPEG do Blender (so roda com a vista ligada)
        self.bview_timer.timeout.connect(self.poll_blender_view)
        QShortcut(QKeySequence("Q"), self, self.close)

    # ---------------------------------------------------------------- controles
    def _controls(self) -> QWidget:
        w = QWidget()
        # caixinhas e seletor na cor ACCENT; ao estilizar o indicador o Qt some com o tique, entao vai um svg
        w.setStyleSheet(
            "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666;"
            " border-radius: 3px; background: #222; }"
            f"QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({CHECK_ICON}); }}"
            f"QComboBox {{ background: {ACCENT}; color: #fff; border-radius: 4px; padding: 4px 8px; }}"
            f"QComboBox QAbstractItemView {{ background: #222; color: #fff; selection-background-color: {ACCENT}; }}")
        g = QGridLayout(w)
        cmd = self.pipe.command

        def button(text, name, checkable=False):
            b = QPushButton(text)
            b.setStyleSheet(f"background:{ACCENT};color:#fff; border-radius: 4px; padding: 6px;")
            b.setCheckable(checkable)
            if checkable:
                b.toggled.connect(lambda v: cmd(name, v))
            else:
                b.clicked.connect(lambda: cmd(name))
            return b

        self.b_rec = button("Gravar sessão", "record", True)
        self.setStyleSheet("background:#aaa;color:#111")
        g.addWidget(button("Calibrar [C]", "calibrate"), 0, 0)
        g.addWidget(button("Reset vaso [R]", "reset"), 0, 1)
        g.addWidget(button("Desfazer [Ctrl+Z]", "undo"), 0, 2)
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
        bv = QCheckBox("Vista do Blender")
        bv.setToolTip("mostra a imagem da câmera do Blender (a cada 5 estados); desligada, o Blender não renderiza")
        bv.toggled.connect(lambda v: (cmd("blender_view", v), self.set_blender_view(v)))
        g.addWidget(bv, 1, 2, 1, 3)

        g.addWidget(QLabel("Velocidade: Cabeça ↓ Acelera · ↑ Freia"), 2, 0, 1, 3)
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
        QShortcut(QKeySequence.StandardKey.Undo, self, lambda: cmd("undo"))
        return w

    # ---------------------------------------------------------------- atualizacao
    def on_result(self, r: dict):
        if "fatal" in r:
            self.debug.setText(f"ERRO FATAL: {r['fatal']}")
            return
        if "image" not in r:
            self.debug.setText(r.get("message", ""))
            return
        self.video.setPixmap(to_pixmap(r["image"], self.video.size()))
        vw = self.video.pixmap().width() or self.rmap.width()   # mapa na mesma largura da imagem da camera
        self.rmap.setPixmap(to_pixmap(rmap_image(r["R"], self.cfg.vase.r_max, r["cols"]),
                                      QSize(vw, self.rmap.height()), keep_aspect=False))
        ms = r["ms"]
        pen = {e: (f"{v * 1000:+.1f}mm" if v == v else "  —  ") for e, v in r["pen"].items()}
        self.debug.setText(debug_html([
            ("estado", f"{r['state']}   {'● GRAVANDO' if r['recording'] else ''}"),
            ("calib", f"{r['calib_step']}"),
            ("FPS", f"{r['fps']:5.1f}   mãos {r['n_hands']}"),
            ("latência", f"captura→processado {r['lat_proc']:5.1f} ms   captura→Blender "
                         + (f"{r['lat_blender']:5.1f} ms" if r['lat_blender'] == r['lat_blender'] else "—")),
            ("ms", f"mãos {ms['hands']:5.1f}  rosto {ms['face']:5.1f}  contorno {ms['contour']:5.1f}  "
                   f"sim {ms['sim']:5.1f}  total {ms['total']:5.1f}"),
            ("torno", f"{r['rpm']:5.1f} rpm  ({r['speed_status']})   cabeça {r['pitch_delta']:+5.1f}° "
                      f"({'↓ baixo' if r['pitch_delta'] > 0 else '↑ cima'})  rosto {'ok' if r['head_ok'] else '—'}"),
            ("vaso", f"H {r['H'] * 100:5.1f} cm   V/V0 {r['volume']:.3f}   φ {math.degrees(r['phi']) % 360:5.1f}°"),
            ("contato", f"L {r['contacts']['L']:<5} {pen['L']}   R {r['contacts']['R']:<5} {pen['R']}"),
            ("extrator", f"{r['extractor']}   idade máscara {r['seg_age']}   SAM {r['seg_ms']:.0f} ms"),
            ("msg", f"{r['message']}"),
        ]))

    def set_blender_view(self, on: bool):
        self.bview_panel.setVisible(on)
        if on:
            self._bview_mtime = 0.0
            self.bview_timer.start(50)
        else:
            self.bview_timer.stop()

    def poll_blender_view(self):
        try:
            mt = os.stat(self._bview_path).st_mtime
        except OSError:
            return                                    # Blender ainda nao gravou
        size = self.bview.size()
        if mt == self._bview_mtime and size == self._bview_size:
            return                                    # nada novo: nao gasta CPU redesenhando
        img = cv2.imread(self._bview_path) if mt != self._bview_mtime else self._bview_img
        if img is None:                               # o Blender troca o arquivo de uma vez (os.replace)
            return
        self._bview_mtime, self._bview_img, self._bview_size = mt, img, size
        self.bview.setPixmap(to_pixmap(img, size))

    def poll_receiver(self):
        if self.rx.poll() and self.rx.R is not None:
            m = self.rx.meta
            self.rmap.setPixmap(to_pixmap(rmap_image(self.rx.R, self.cfg.vase.r_max, None), self.rmap.size()))
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
