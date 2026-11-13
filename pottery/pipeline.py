"""Loop por quadro (QThread): captura → mãos/rosto → calibração → contorno → contato → vaso → velocidade → overlay/UDP/CSV."""
from __future__ import annotations

import csv
import math
import queue
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import contour as ct
from .calibration import CalibProfile, Calibrator
from .capture import Camera, Recorder, Replay
from .net import F_CALIB, F_PAUSED, F_RESET, Sender
from .speed import SpeedController
from .tracking import HandTracker, HeadObs, HeadTracker, assign_edges
from .vase import EdgeContact, TopContact, Vase

def _ascii(s: str) -> str:
    """cv2.putText não desenha acentos."""
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


STATE_COLORS ={"FREE": (170, 170, 170), "PUSH": (0, 140, 255), "PRESS": (255, 80, 200)}
CLAY = (60, 110, 200)


class Pipeline(QThread):
    result = Signal(object)

    def __init__(self, cfg, replay: str | None = None, profile: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.replay = replay
        self.profile_path = profile or cfg.calib.profile_path
        self.cmds: queue.Queue = queue.Queue()
        self._running = True
        # a câmera (MSMF) precisa ser aberta na thread principal: dentro da QThread o open trava (COM)
        self.src = Replay(replay) if replay else Camera(cfg.camera)

    # ---------------------------------------------------------------- API para a GUI
    def command(self, name: str, value=None):
        self.cmds.put((name, value))

    def stop(self):
        self._running = False
        self.wait(3000)

    # ---------------------------------------------------------------- setup
    def _setup(self):
        c = self.cfg
        self.hands_t = HandTracker(c.models_dir, c.hand)
        self.head_t = HeadTracker(c.models_dir)
        self.vase = Vase(c.vase)
        self.speed = SpeedController(c.speed)
        self.contacts = {e: EdgeContact(c.contact) for e in ("R", "L")}
        self.tops = {e: TopContact(c.contact) for e in ("R", "L")}
        self.prev_prof: dict[str, ct.ToolProfile | None] = {"R": None, "L": None}
        self.pema = ct.ProfileEMA(c.hand.profile_ema)
        self.scale_s: dict[int, float] = {}
        self.sender = Sender(c.net.targets, c.net.rate_hz, c.net.echo_port, c.net.max_datagram, c.vase.r_scale) if c.net.enabled else None
        self.extractor = c.seg.extractor
        self.seg: ct.SamSegmenter | None = None
        self.message = ""
        self.paused = False
        self.rec: dict | None = None
        self.prof: CalibProfile | None = None
        self.calib: Calibrator | None = None
        self.state = "INIT"
        self.head = HeadObs(False)
        self.fps = 0.0
        self.reset_at: float | None = None   # contagem do reinício após esmagar
        self.reset_left = 0.0
        self.show_lm = False

    def _apply_profile(self):
        """Aplica o perfil ao vaso: altura máxima = 80% da imagem (base no rodapé) e raio inicial
        = largura medida entre as mãos na calibração (se houver). Reinicia o cilindro."""
        p, cv = self.prof, self.cfg.vase
        cv.h_max = 0.8 * p.height * p.k
        if p.calibrated and p.r0 > 0:
            cv.r0 = p.r0
        self.vase.reset()

    def _start_calibration(self, w, h):
        cv = self.cfg.vase
        self.calib = Calibrator(self.cfg.calib, self.cfg.hand, w, h, self.cfg.camera.mirror, base=self.prof,
                                r_limits=(2 * cv.r_min, 0.9 * cv.r_max))
        self.state = "CALIBRATING"

    def _load_profile(self, w, h):
        p = Path(self.profile_path)
        self.state = "MODELING"
        if p.exists():
            self.prof, warn = CalibProfile.load(p, w, h, self.cfg.camera.mirror)
            self.speed.set_neutral(self.prof.pitch0, self.prof.yaw0)
            self.message = warn or f"perfil carregado: {p.name}"
        else:
            self.prof = CalibProfile.default(w, h)
            self.message = "NÃO CALIBRADO — clique em Calibrar"
        self._apply_profile()

    # ---------------------------------------------------------------- comandos
    def _handle(self, name, val, img_shape):
        c = self.cfg
        if name == "calibrate":
            self._start_calibration(img_shape[1], img_shape[0])
        elif name == "cancel_calibration" and self.calib:
            self.calib, self.state = None, "MODELING"
        elif name == "reset":
            self.vase.reset()
            for e in self.contacts.values():
                e._free()
            self.speed.rpm = c.speed.min_rpm
        elif name == "pause":
            self.paused = bool(val)
        elif name == "record":
            self._record(bool(val), img_shape)
        elif name == "extractor":
            self._set_extractor(val)
        elif name == "lock":
            self.speed.locked = bool(val)
        elif name == "smoothing":
            c.hand.smoothing = bool(val)
        elif name == "depth_norm":
            c.hand.depth_norm = bool(val)
        elif name == "side_gain":
            c.vase.side_gain = float(val)
        elif name == "smooth_lambda":
            c.vase.smooth_lambda = float(val)
        elif name == "show_landmarks":
            self.show_lm = bool(val)

    def _set_extractor(self, name):
        if self.seg:
            self.seg.close()
            self.seg = None
        self.extractor = name
        if name in ct._SAM_WEIGHTS:
            try:
                self.seg = ct.SamSegmenter(name, self.cfg.models_dir, self.cfg.seg.device, self.cfg.seg.roi_pad, self.cfg.seg.max_age)
                self.message = f"{name} carregado"
            except Exception as e:
                self.message = f"{name} indisponível ({type(e).__name__}: {e}) → landmarks"
                self.extractor = "landmarks"

    def _record(self, on: bool, shape):
        if on and self.rec is None:
            folder = Path(self.cfg.data_dir) / "sessions" / time.strftime("%Y%m%d-%H%M%S")
            r = Recorder(folder, (shape[1], shape[0]), self.cfg.camera.fps)
            files = {k: open(folder / f"{k}.csv", "w", newline="") for k in ("frames", "hands", "latency")}
            w = {k: csv.writer(f) for k, f in files.items()}
            w["frames"].writerow(["frame_id", "t_capture_ns", "t_send_ns", "ms_hands", "ms_face", "ms_contour", "ms_sim",
                                  "ms_total", "fps", "extractor", "seg_age", "omega", "phi", "height", "volume",
                                  "pitch_delta", "head_valid", "contact_L", "contact_R", "smoothing_on", "depthnorm_on", "state"])
            w["hands"].writerow(["frame_id", "edge", "track_id", "scale_px", "s", "contact_state", "pen_m", "mask_source", "r_near"])
            w["latency"].writerow(["seq", "frame_id", "t_capture_ns", "t_send_ns", "t_recv_ns", "t_applied_ns", "t_echo_rx_ns"])
            self.rec = dict(folder=folder, r=r, files=files, w=w)
            (folder / "config.txt").write_text(repr(self.cfg))
            self.message = f"gravando em {folder}"
        elif not on and self.rec is not None:
            self.rec["r"].close()
            for f in self.rec["files"].values():
                f.close()
            self.message = f"sessão salva: {self.rec['folder']}"
            self.rec = None

    # ---------------------------------------------------------------- loop
    def run(self):
        try:
            self._setup()
        except Exception as e:
            self.result.emit({"fatal": f"{type(e).__name__}: {e}"})
            return
        last_t = None
        t_loop = time.perf_counter()
        try:
            while self._running:
                fr = self.src.read(timeout=0.5)
                if fr is None:
                    if isinstance(self.src, Replay) and self.src.eof:
                        self.message = "fim do replay"
                        self.result.emit({"message": self.message, "eof": True})
                        break
                    if self.src.stalled(1.0):
                        self.result.emit({"message": "SEM CÂMERA", "stalled": True})
                    continue
                img = fr.image
                if self.prof is None:
                    self._load_profile(img.shape[1], img.shape[0])
                while not self.cmds.empty():
                    self._handle(*self.cmds.get_nowait(), img.shape)
                self._process(fr, last_t)
                last_t = fr.ts_ns / 1e9
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(now - t_loop, 1e-6) if self.fps else 1 / max(now - t_loop, 1e-6)
                t_loop = now
        finally:
            self._record(False, None)
            for obj in (self.src, self.hands_t, self.head_t, self.seg, self.sender):
                try:
                    obj and obj.close()
                except Exception:
                    pass

    def _process(self, fr, last_t):
        c = self.cfg
        img = fr.image
        t = fr.ts_ns / 1e9
        dt = 0.0 if last_t is None else min(max(t - last_t, 0.0), 0.2)
        ms = {}
        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        hands = self.hands_t.process(rgb, fr.ts_ns)
        t1 = time.perf_counter()
        ms["hands"] = (t1 - t0) * 1000
        if fr.id % max(1, c.speed.face_every) == 0 or self.state == "CALIBRATING":
            self.head = self.head_t.process(rgb, fr.ts_ns)
        t2 = time.perf_counter()
        ms["face"] = (t2 - t1) * 1000

        if self.state == "CALIBRATING":
            p = self.calib.update(hands, self.head, img, t)
            self.prof = self.calib.prof
            if p is not None:
                p.save(self.profile_path)
                self.speed.set_neutral(p.pitch0, p.yaw0)
                self._apply_profile()
                self.state = "MODELING"
                self.calib = None
                self.message = f"calibrado → {self.profile_path}  (raio {p.r0 * 100:.1f} cm)"
        prof = self.prof
        modeling = self.state == "MODELING" and not self.paused

        # ---------------- contorno → perfil
        sticky = {e: k.track_id for e, k in self.contacts.items() if k.state != "FREE" and k.track_id is not None}
        edges = assign_edges(hands, prof.axis_u, sticky)
        if self.seg is not None:
            if self.seg.error:
                self.message = f"SAM falhou ({self.seg.error}) → landmarks"
                self.seg.close()
                self.seg, self.extractor = None, "landmarks"
            else:
                self.seg.request(fr.id, img, [h for h in edges.values() if h is not None])
        profiles, masks, tops, seg_age = {}, {}, {}, -1
        for e, h in edges.items():
            if h is None:
                profiles[e] = self.pema(e, None, False)
                continue
            kappa = prof.kappa_for(h.handedness, c.hand.kappa_default)
            L = h.scale(kappa)
            s_raw = L / prof.L_ref_for(h.handedness)
            s_prev = self.scale_s.get(h.track_id, s_raw)
            a = c.hand.scale_ema * (0.3 if self.contacts[e].state != "FREE" else 1.0)
            s = s_prev + a * (s_raw - s_prev)
            self.scale_s[h.track_id] = s
            hm = None
            if self.seg is not None:
                hm = self.seg.get(h, fr.id, img.shape)
                if hm is not None:
                    seg_age = max(seg_age, hm.age)
            if hm is None:
                hm = ct.landmarks_mask(h, img.shape, c.hand.finger_radius, kappa)
            cl = ct.clean_mask(hm)
            p = None
            s_eff = s if c.hand.depth_norm else 1.0
            if cl is not None:
                hm, contour = cl
                masks[e] = (hm, contour)
                p = ct.compute_profile(hm, contour, e, s_eff, prof, self.vase.H, self.vase.n_z, c.vase.feather_m)
                tops[e] = ct.hand_bottom_over_vase(hm, s_eff, prof, float(self.vase.R[:, -1].max()))
            profiles[e] = self.pema(e, p, c.hand.smoothing)
        t3 = time.perf_counter()
        ms["contour"] = (t3 - t2) * 1000

        # ---------------- velocidade + vaso
        if modeling and prof.calibrated:
            omega = self.speed.update(self.head, t, dt)
        else:   # sem calibração não há pose neutra da cabeça: torno na velocidade mínima
            self.speed.rpm = c.speed.min_rpm
            omega = self.speed.rpm * 2 * math.pi / 60
            self.speed.status = "calibre primeiro" if not prof.calibrated else "pausado"
        self.vase.omega = omega
        phi0, phi1 = self.vase.advance(dt) if modeling else (self.vase.phi, self.vase.phi)
        # esmagou (altura < 15% da inicial): avisa e reinicia em 3 s — tempo de tirar a mão do meio
        if modeling and self.reset_at is None and self.vase.Hc.mean() < c.vase.crush_reset_frac * c.vase.h0:
            self.reset_at = t + 3.0
        crushing = self.reset_at is not None
        if crushing:
            self.reset_left = self.reset_at - t
            if self.reset_left <= 0:
                self.vase.reset()
                self.reset_at, crushing = None, False
                self.message = "vaso reiniciado"
        for e in ("R", "L"):
            h = edges[e]
            k = self.contacts[e]
            if not modeling or crushing:
                k._free()
                self.prev_prof[e] = None
                continue
            z_low, zc = tops.get(e, (None, None))
            if self.tops[e].update(z_low, zc, self.vase.top_at(e)):   # mão por cima: abaixa só onde ela bate
                self.vase.press_top(e, phi0, phi1, z_low)
                k._free()
                self.prev_prof[e] = None
                continue
            zg = np.linspace(0.0, self.vase.H, self.vase.n_z)          # mesma grade de z do perfil da mão
            act = k.update(profiles.get(e), h.track_id if h else None, self.vase.wall(e, zg), t)
            prev = self.prev_prof[e] if act != "NONE" else None
            self.vase.apply(e, phi0, phi1, prev, profiles.get(e), act)
            self.prev_prof[e] = profiles.get(e) if act != "NONE" else None
        t4 = time.perf_counter()
        ms["sim"] = (t4 - t3) * 1000

        # ---------------- saídas
        overlay = self._draw(img.copy(), edges, profiles, masks)
        flags = (F_CALIB if self.state == "CALIBRATING" else 0) | (F_PAUSED if self.paused else 0) | (F_RESET if self.vase.reset_flag else 0)
        self.vase.reset_flag = False
        t_send = 0
        if self.sender:
            if self.sender.maybe_send(self.vase.R, self.vase.H, self.vase.phi, omega, fr.id, fr.ts_ns, self.vase.version, flags,
                                      hfrac=self.vase.Hc / self.vase.H):
                t_send = time.time_ns()
            echoes = self.sender.poll_echo()
        else:
            echoes = []
        ms["total"] = (time.perf_counter() - t0) * 1000
        vol_ratio = self.vase.volume() / self.vase.V0
        if self.rec:
            self.rec["r"].write(fr)
            w = self.rec["w"]
            w["frames"].writerow([fr.id, fr.ts_ns, t_send, *(f"{ms[k]:.2f}" for k in ("hands", "face", "contour", "sim", "total")),
                                  f"{self.fps:.1f}", self.extractor, seg_age, f"{omega:.4f}", f"{self.vase.phi:.4f}",
                                  f"{self.vase.H:.4f}", f"{vol_ratio:.4f}", f"{self.speed.delta:.2f}", int(self.speed.valid),
                                  self.contacts["L"].state, self.contacts["R"].state, int(c.hand.smoothing), int(c.hand.depth_norm), self.state])
            for e, h in edges.items():
                if h is None:
                    continue
                p = profiles.get(e)
                rn = ";".join("" if np.isnan(x) else f"{x:.5f}" for x in p.r_near) if p is not None else ""
                w["hands"].writerow([fr.id, e, h.track_id, f"{h.scale(prof.kappa_for(h.handedness)):.2f}",
                                     f"{self.scale_s.get(h.track_id, 1):.4f}", self.contacts[e].state,
                                     f"{self.contacts[e].pen:.5f}", masks[e][0].source if e in masks else "", rn])
            for d in echoes:
                w["latency"].writerow([d["seq"], d["frame_id"], d["t_capture_ns"], d["t_send_ns"], d["t_recv_ns"], d["t_applied_ns"], d["t_echo_rx_ns"]])

        self.result.emit(dict(
            image=overlay, R=self.vase.R.copy(), H=self.vase.H, Hf=self.vase.Hc / self.vase.H, phi=self.vase.phi, omega=omega,
            state=self.state + (" (pausado)" if self.paused else ""), fps=self.fps, ms=ms, rpm=self.speed.rpm,
            speed_status=self.speed.status, pitch_delta=self.speed.delta, head_ok=self.head.ok,
            volume=vol_ratio,
            contacts={e: ("PRESS" if self.tops[e].state == "PRESS" else k.state) for e, k in self.contacts.items()},
            pen={e: k.pen for e, k in self.contacts.items()}, extractor=self.extractor, seg_age=seg_age,
            seg_ms=self.seg.ms if self.seg else 0.0, message=self.message, recording=self.rec is not None,
            calib_step=self.calib.name if self.calib else "", n_hands=len(hands),
            cols={e: self.vase.col_index(e) for e in ("R", "L")}))

    # ---------------------------------------------------------------- overlay
    def _draw_head_gauge(self, img):
        """Medidor vertical da inclinação da cabeça (↓ acelera, ↑ para), sempre visível, colado na direita."""
        sc = self.cfg.speed
        H_img, W_img = img.shape[:2]
        F = cv2.FONT_HERSHEY_SIMPLEX
        full = 25.0                                                      # graus no fim da escala
        gw, gh = 44, min(380, H_img - 140)
        gx = W_img - gw - 24                                             # colado na borda direita
        gc = H_img // 2
        top, bot = gc - gh // 2, gc + gh // 2
        px = lambda deg: int(gc + np.clip(deg / full, -1, 1) * gh / 2)   # para baixo = mais abaixo

        def text_left(s, y, scale, color):                               # texto alinhado à esquerda da barra
            tw = cv2.getTextSize(s, F, scale, 2)[0][0]
            cv2.putText(img, s, (gx - 24 - tw, y), F, scale, color, 2, cv2.LINE_AA)

        cv2.rectangle(img, (gx, top), (gx + gw, px(-sc.brake_deg)), (60, 60, 200), -1)          # zona "para"
        cv2.rectangle(img, (gx, px(sc.accel_deg)), (gx + gw, bot), (60, 180, 60), -1)           # zona "acelera"
        cv2.rectangle(img, (gx, px(-sc.brake_deg)), (gx + gw, px(sc.accel_deg)), (120, 120, 120), -1)
        cv2.rectangle(img, (gx, top), (gx + gw, bot), (240, 240, 240), 2)
        cv2.line(img, (gx, gc), (gx + gw, gc), (240, 240, 240), 1)                               # neutro
        text_left("PARA", top + 26, 0.9, (120, 120, 255))
        text_left("ACELERA", bot - 6, 0.9, (120, 255, 120))
        if self.head.ok:
            d = sc.pitch_sign * (self.head.pitch - self.prof.pitch0)    # > 0 = olhando para baixo
            y = px(d)
            cv2.line(img, (gx - 16, y), (gx + gw + 10, y), (0, 255, 255), 6)
            text_left(f"{d:+.1f} deg", y + 10, 1.0, (0, 255, 255))
            if self.show_lm and self.head.bbox:                          # retângulo do rosto só no debug
                x0, y0, x1, y1 = self.head.bbox
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
        else:
            text_left("sem rosto", gc + 10, 0.9, (200, 200, 200))
        status = _ascii(self.speed.status)
        tw = cv2.getTextSize(status, F, 0.8, 2)[0][0]
        cv2.putText(img, status, (W_img - tw - 16, bot + 40), F, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw(self, img, edges, profiles, masks):
        prof, v = self.prof, self.vase
        z = np.linspace(0, v.H, v.n_z)                       # grade global (perfil das mãos)
        au = int(prof.axis_u)
        top = int(prof.z_to_v(v.H))
        bv = int(prof.base_v)
        for y in range(top, bv, 12):
            cv2.line(img, (au, y), (au, min(y + 6, bv)), (200, 200, 200), 1)
        sil = {}
        for e, sgn in (("R", 1), ("L", -1)):                 # cada lado com a altura da SUA coluna
            i = v.col_index(e)
            vv = prof.z_to_v(v.Hc[i] * np.linspace(0, 1, v.n_z))
            sil[e] = np.stack([prof.x_to_u(sgn * v.R[i]), vv], 1).astype(np.int32)
        fill = img.copy()                                    # preenchimento do vaso a 30%
        cv2.fillPoly(fill, [np.concatenate([sil["R"], sil["L"][::-1]])], CLAY)
        cv2.addWeighted(fill, 0.3, img, 0.7, 0, img)
        for pts in sil.values():
            cv2.polylines(img, [pts], False, CLAY, 3, cv2.LINE_AA)
        cv2.line(img, tuple(map(int, sil["L"][-1])), tuple(map(int, sil["R"][-1])), CLAY, 2, cv2.LINE_AA)   # boca
        cv2.line(img, tuple(map(int, sil["L"][0])), tuple(map(int, sil["R"][0])), CLAY, 2)                    # base
        for e, h in edges.items():
            if h is None:
                continue
            st = "PRESS" if self.tops[e].state == "PRESS" else self.contacts[e].state
            col = STATE_COLORS.get(st, (255, 255, 255))
            if e in masks:
                cv2.polylines(img, [masks[e][1].astype(np.int32)], True, col, 2, cv2.LINE_AA)   # contorno real
            p = profiles.get(e)
            if p is not None:
                s = self.scale_s.get(h.track_id, 1.0) if self.cfg.hand.depth_norm else 1.0
                sgn = 1 if e == "R" else -1
                cvz = z[p.cover]
                pu = prof.x_to_u(sgn * p.r_near[p.cover], s)
                pv = prof.z_to_v(cvz, s)
                cv2.polylines(img, [np.stack([pu, pv], 1).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
            if self.state == "CALIBRATING" or self.show_lm:
                for chain in ct.FINGERS:
                    cv2.polylines(img, [h.lm[list(chain)].astype(np.int32)], False, (0, 200, 255), 1, cv2.LINE_AA)
                for i, q in enumerate(h.lm.astype(int)):
                    cv2.circle(img, tuple(q), 4, (0, 255, 255), -1)
                    if self.show_lm:
                        cv2.putText(img, str(i), (q[0] + 5, q[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(img, f"{e}:{st}", (int(h.center[0]) - 30, int(h.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        H, W = img.shape[:2]
        self._draw_head_gauge(img)
        if self.state == "CALIBRATING" and self.calib:
            cb = self.calib
            cv2.rectangle(img, (0, 0), (W, 95), (0, 0, 0), -1)
            cv2.putText(img, _ascii(cb.instruction), (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            if cb.phase == "COUNTDOWN":
                n = math.ceil(cb.countdown_left())
                cv2.putText(img, f"capturando em {n}...", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(img, str(n), (W // 2 - 40, H // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 220, 255), 10, cv2.LINE_AA)
            else:
                cv2.putText(img, _ascii(f"CAPTURANDO - {cb.hint}"), (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2, cv2.LINE_AA)
                cv2.rectangle(img, (520, 60), (520 + int((W - 535) * cb.progress()), 80), (0, 220, 120), -1)
                cv2.rectangle(img, (520, 60), (W - 15, 80), (200, 200, 200), 1)
        elif self.reset_at is not None:
            n = max(1, math.ceil(self.reset_left))
            cv2.rectangle(img, (0, 0), (W, 60), (0, 0, 150), -1)
            cv2.putText(img, f"ESMAGOU! Reiniciando em {n}...", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(n), (W // 2 - 40, H // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
        elif not self.prof.calibrated:
            cv2.rectangle(img, (0, 0), (W, 45), (0, 0, 120), -1)
            cv2.putText(img, "NAO CALIBRADO - clique em Calibrar", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return img
