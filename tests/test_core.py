import math

import numpy as np
import pytest

from pottery.calibration import CalibProfile
from pottery.config import Config
from pottery.contour import HandMask, ToolProfile, compute_profile, feather, hand_bottom_over_vase, mask_row_spans
from pottery.net import decode, encode_state, seq_newer, stamp_send
from pottery.speed import SpeedController
from pottery.tracking import HeadObs
from pottery.vase import EDGE_THETA, EdgeContact, TopContact, Vase, sweep

N = 128
NZ = Config().vase.n_z   # bins em z da config atual


# ---------------------------------------------------------------- varredura (CA-04)
@pytest.mark.parametrize("dphi_deg", [0.5, 2.8, 10, 40, 179])
def test_sweep_covers_every_column_once_per_turn(dphi_deg):
    dphi = math.radians(dphi_deg)
    phi, hits = 0.3, np.zeros(N, int)
    while phi < 0.3 + 2 * math.pi - 1e-9:
        step = min(dphi, 0.3 + 2 * math.pi - phi)
        cols, s, _ = sweep(EDGE_THETA["L"], phi, phi + step, N)
        hits[cols] += 1
        phi += step
    assert (hits >= 1).all(), "listra sem modelagem"
    assert hits.max() <= 2


def test_sweep_seam_and_full_turn():
    cols, _, _ = sweep(0.0, 2 * math.pi - 0.05, 2 * math.pi + 0.05, N)  # cruza a costura
    assert len(cols) >= 2 and len(set(cols.tolist())) == len(cols)
    assert all(0 <= c < N for c in cols)
    cols, s, _ = sweep(0.0, 0.0, 7.0, N)
    assert len(cols) == N
    cols, s, _ = sweep(0.0, 1.0, 1.0, N)  # omega = 0
    assert len(cols) == 1


def _flat_profile(r, n_z=NZ, lo=0.2, hi=0.8):
    cover = np.zeros(n_z, bool)
    cover[int(lo * n_z):int(hi * n_z)] = True
    rn = np.where(cover, r, np.nan)
    return ToolProfile(rn, rn + 0.05, cover, cover.astype(float), r + 0.025, np.zeros((3, 2)))


def _squeeze(v, p, action="PUSH", turns=60):
    for k in range(turns):
        v.apply("R", k * 0.3, (k + 1) * 0.3, p, p, action)
        v.apply("L", k * 0.3, (k + 1) * 0.3, p, p, action)


# ---------------------------------------------------------------- vaso
def test_push_never_increases_and_clamps():
    cfg = Config().vase
    v = Vase(cfg)
    before = v.R.copy()
    p = _flat_profile(0.0)  # empurra ate o eixo
    for k in range(40):
        v.apply("R", k * 0.2, (k + 1) * 0.2, p, p, "PUSH")
    assert (v.R <= before + 1e-6).all()
    assert v.R.min() >= cfg.r_min - 1e-7


def test_dent_never_comes_back():
    """Empurrar numa faixa e depois em outra: o sulco antigo nao e 'preenchido' de volta (barro nao volta)."""
    cfg = Config().vase
    cfg.side_gain, cfg.smooth_lambda = 0.0, 1.0
    v = Vase(cfg)
    for k in range(30):
        v.apply("R", k * 0.3, (k + 1) * 0.3, _flat_profile(0.04, lo=0.2, hi=0.3), _flat_profile(0.04, lo=0.2, hi=0.3), "PUSH")
    dent = v.R.copy()
    for k in range(30, 60):
        v.apply("R", k * 0.3, (k + 1) * 0.3, _flat_profile(0.04, lo=0.6, hi=0.7), _flat_profile(0.04, lo=0.6, hi=0.7), "PUSH")
    assert (v.R <= dent + 1e-7).all()
    assert v.R[:, int(0.25 * NZ)].mean() < cfg.r0 - 0.01      # o sulco continua la


def test_side_gain_controls_height():  # CA-06 + slider
    cfg = Config().vase
    p = _flat_profile(cfg.r0 * 0.75, lo=0.0, hi=1.0)
    heights = {}
    for g in (0.0, 0.4, 1.0):
        cfg.side_gain = g
        v = Vase(cfg)
        v0 = v.volume()
        _squeeze(v, p)
        heights[g] = v.H
        if g == 1.0:
            assert abs(v.volume() - v0) / v0 < 0.01   # conserva volume
    assert heights[0.0] == pytest.approx(cfg.h0)       # 0 = altura fixa
    assert cfg.h0 < heights[0.4] < heights[1.0]



def test_press_top_is_local_and_conserves_volume():
    cfg = Config().vase
    v = Vase(cfg)
    v0, r0, h0 = v.volume(), float(v.R.mean()), v.H
    v.press_top("R", 0.0, 0.3, 0.10)                 # um quadro: so as colunas sob a mao
    hit = v.Hc < h0 - 1e-9
    assert 0 < hit.sum() < 10                         # local, nao o topo todo
    assert (v.R[hit].mean(1) > r0).all() and (v.R[~hit] == np.float32(r0)).all()
    for k in range(1, 400):                           # girando varias voltas: a borda inteira vai baixando
        v.press_top("R", k * 0.3, (k + 1) * 0.3, 0.10)
    assert v.Hc.max() == pytest.approx(0.10, abs=2e-3)
    assert v.R.mean() > r0 * 1.15                     # achatou e engordou
    assert abs(v.volume() - v0) / v0 < 0.01


def test_undo_restores_previous_states():
    v = Vase(Config().vase)
    r0, h0 = v.R.copy(), v.Hc.copy()
    v.push_undo()
    v.press_top("R", 0.0, 3.0, 0.10)
    r1, h1 = v.R.copy(), v.Hc.copy()
    v.push_undo()
    v.reset()
    assert v.undo() and np.array_equal(v.R, r1) and np.array_equal(v.Hc, h1)   # desfaz o reset
    assert v.undo() and np.array_equal(v.R, r0) and np.array_equal(v.Hc, h0)   # desfaz o esmagamento
    assert not v.undo()                                                        # pilha vazia


def test_top_contact_only_from_above():
    c = TopContact(Config().contact)
    H = 0.15
    assert not c.update(0.10, 0.08, H)        # mao abaixo do topo (pelo lado): nao engata
    assert not c.update(0.20, 0.25, H)        # acima, sem encostar
    assert c.update(0.149, 0.22, H)           # encostou vindo de cima
    assert c.update(0.12, 0.19, H)            # empurrando
    for _ in range(3):
        a = c.update(0.20, 0.27, H)           # afastou
    assert not a


# ---------------------------------------------------------------- contato (CA-05)
def test_contact_hysteresis_outside_only():
    c = EdgeContact(Config().contact)
    wall = np.full(NZ, 0.06)
    p_far = _flat_profile(0.10)
    assert c.update(p_far, 1, wall, 0.0) == "NONE"
    assert c.update(_flat_profile(0.059), 1, wall, 0.03) == "PUSH"
    assert c.update(_flat_profile(0.01), 1, wall, 0.06) == "PUSH"   # atravessa: continua empurrando
    assert c.update(_flat_profile(0.065), 1, wall, 0.09) == "PUSH"
    for i in range(3):
        a = c.update(_flat_profile(0.09), 1, wall, 0.12 + i * 0.03)
    assert a == "NONE" and c.state == "FREE"
    assert c.update(_flat_profile(0.0), 2, wall, 0.3) == "NONE"     # mao por dentro nao engata


# ---------------------------------------------------------------- calibracao (2 etapas)
def test_calibration_hands_gives_axis_scale_and_width():
    from types import SimpleNamespace
    from pottery.calibration import Calibrator
    cfg = Config()
    cal = Calibrator(cfg.calib, cfg.hand, 1280, 720, True)

    def hand(label, x_inner, side):          # mao vertical de lado: 21 pontos numa faixa de 30 px de largura
        xs = x_inner + side * np.linspace(0, 30, 21)
        ys = np.linspace(500, 300, 21)
        lm = np.stack([xs, ys], 1)
        return SimpleNamespace(handedness=label, lm=lm, center=lm.mean(0), d09=150.0, d517=40.0)

    hands = [hand("Left", 540, -1), hand("Right", 740, +1)]   # bordas internas a 200 px
    head = HeadObs(True, 2.0, 1.0, 0.0)
    prof, t = None, 0.0
    while prof is None and t < 20:
        prof = cal.update(hands, head, np.zeros((720, 1280, 3), np.uint8), t)
        t += 1 / 30
    assert prof is not None and prof.calibrated
    assert prof.pitch0 == pytest.approx(2.0)
    assert prof.k == pytest.approx(cfg.hand.L_real_m / 150.0)
    assert prof.axis_u == pytest.approx(640.0)
    rad = cfg.hand.finger_radius * 150.0
    assert prof.r0 == pytest.approx(prof.k * (200 - 2 * rad) / 2)
    assert prof.base_v == pytest.approx(0.9 * 720)


# ---------------------------------------------------------------- geometria
def test_depth_normalization_pinhole():
    f, d0, L_real = 900.0, 0.6, 0.095
    w, h = 1280, 720
    prof = CalibProfile.default(w, h)
    prof.axis_u = w / 2
    prof.L_ref = {"Right": f * L_real / d0}
    prof.k = L_real / prof.L_ref["Right"]
    X = 0.08
    for d in np.linspace(0.7 * d0, 1.4 * d0, 8):
        u = w / 2 + f * X / d
        s = (f * L_real / d) / prof.L_ref["Right"]
        x, _ = prof.to_vase(u, h / 2, s)
        assert abs(x - X) / X < 0.02
        x_raw, _ = prof.to_vase(u, h / 2, 1.0)
    assert abs(x_raw - X) / X > 0.2  # sem normalizacao erra muito


def _two_fingers_mask():
    """Mao de lado com dois 'dedos' horizontais apontando para o eixo (a esquerda) e vao entre eles."""
    m = np.zeros((200, 200), np.uint8)
    m[40:80, 0:200] = 255    # dedo de cima (chega ate u=700)
    m[120:160, 60:200] = 255  # dedo de baixo, mais curto
    m[40:160, 150:200] = 255  # palma
    return HandMask((700, 300, 900, 500), m, "test")


def test_profile_follows_fingers_not_convex():
    prof = CalibProfile.default(1280, 720)
    hm = _two_fingers_mask()
    umin, umax = mask_row_spans(hm, np.array([250.0, 360.0, 400.0, 440.0]))
    assert np.isnan(umin[0])
    assert umin[1] == 700 and umin[3] == 760 and umin[2] == 850   # no vao entre os dedos so a palma
    H = (prof.base_v - 300) * prof.k
    p = compute_profile(hm, np.zeros((3, 2)), "R", 1.0, prof, H, N, 0.003)
    near = p.r_near[p.cover]
    assert near.max() - near.min() > 0.5 * 140 * prof.k           # dedo e vao deixam marcas distintas
    w = feather(np.array([0, 1, 1, 1, 1, 0], bool), 0.004, 0.008)
    assert w[0] == 0 and 0 < w[1] < w[2] <= 1


def test_depth_norm_keeps_contact_in_place():
    prof = CalibProfile.default(1280, 720)
    anchor = (700.0, 200.0)                                  # contato acima do centro da imagem
    z1 = prof.to_vase(*anchor, 1.0)[1]
    for s in (0.8, 1.3):
        x, z = prof.to_vase(*anchor, s, anchor)
        assert z == pytest.approx(z1) and x == pytest.approx(prof.to_vase(*anchor, 1.0)[0])   # contato nao escorrega
        assert prof.z_to_v(z, s, anchor) == pytest.approx(anchor[1])                          # ida e volta
    _, z_up = prof.to_vase(anchor[0], anchor[1] - 50, 1.3, anchor)
    assert (z_up - z1) == pytest.approx(prof.k * 50 / 1.3)                                    # espessura corrigida


def test_hand_bottom_over_vase():
    prof = CalibProfile.default(1280, 720)
    m = np.zeros((100, 300), np.uint8)
    m[0:60, :] = 255
    hm = HandMask((int(prof.axis_u) - 150, 200, int(prof.axis_u) + 150, 300), m, "test")
    z_low, zc = hand_bottom_over_vase(hm, 1.0, prof, 0.06)
    _, z_exp = prof.to_vase(prof.axis_u, 259)
    assert z_low == pytest.approx(float(z_exp), abs=1e-6) and zc > z_low
    far = HandMask((0, 200, 100, 300), m[:, :100], "test")
    assert hand_bottom_over_vase(far, 1.0, prof, 0.06)[0] is None


# ---------------------------------------------------------------- protocolo (CA-08)
def test_protocol_roundtrip_single_datagram():
    R = np.random.default_rng(0).uniform(0.01, 0.2, (128, 128)).astype(np.float32)
    hf = np.linspace(0.5, 1.0, 128)
    bufs = encode_state(R, 0.22, 7.0, 3.0, 5, 9, 123, 4, hfrac=hf, wetness=0.7)
    assert len(bufs) == 1 and len(bufs[0]) == 64 + 128 * 128 * 2 + 128 * 2 <= 65507
    stamp_send(bufs[0], 456)
    d = decode(bytes(bufs[0]))
    assert d["t_capture_ns"] == 123 and d["t_send_ns"] == 456 and d["height"] == pytest.approx(0.22)
    assert np.abs(d["R"] - R).max() <= 0.30 / 65535
    assert np.abs(d["hfrac"] - hf).max() <= 1 / 65535
    assert d["wetness"] == pytest.approx(0.7, abs=1 / 65535)
    assert decode(b"XXXX" + bytes(bufs[0][4:])) is None
    assert decode(bytes(bufs[0][:-2])) is None


def test_protocol_chunks():
    R = np.random.default_rng(1).uniform(0.01, 0.2, (256, 256)).astype(np.float32)
    bufs = encode_state(R, 0.2, 0, 0, 1, 1, 1, 1)
    assert len(bufs) == 3 and all(len(b) <= 65000 for b in bufs)
    out = np.zeros_like(R)
    for b in bufs:
        d = decode(bytes(b))
        out[d["col_start"]:d["col_start"] + d["col_count"]] = d["R"]
    assert np.abs(out - R).max() <= 0.30 / 65535
    assert seq_newer(0, 0xFFFFFFFF) and not seq_newer(5, 6)


# ---------------------------------------------------------------- velocidade (CA-07)
def _run(sc, pitch, n, t, ok=True):
    dt = 1 / 30
    for _ in range(n):
        t += dt
        sc.update(HeadObs(ok, pitch, 0, 0), t, dt)
    return t


def test_speed_down_accelerates_up_stops():
    scfg = Config().speed
    scfg.accel_deg, scfg.brake_deg = 8.0, 4.0  # limites fixos no teste (independem do padrao do config)
    sc = SpeedController(scfg)             # pitch > 0 = olhando para baixo (pitch_sign = 1)
    t = _run(sc, 3.0, 150, 0.0)            # zona morta
    assert sc.rpm == sc.cfg.min_rpm
    t = _run(sc, 15.0, 5, t)               # olhada curta para baixo (< dwell)
    t = _run(sc, 0.0, 30, t)
    assert sc.rpm == sc.cfg.min_rpm
    t = _run(sc, 15.0, 60, t)              # para baixo sustentado: acelera
    r = sc.rpm
    assert r > 10
    t = _run(sc, 0.0, 60, t)               # neutra: mantem (+- atraso do filtro)
    assert sc.rpm == pytest.approx(r, rel=0.03)
    r = sc.rpm
    t = _run(sc, 0.0, 30, t, ok=False)     # rosto perdido: mantem
    assert sc.rpm == pytest.approx(r)
    t = _run(sc, 6.0, 60, t)               # 6 graus para baixo (< 8 graus): ainda mantem
    assert sc.rpm == pytest.approx(r, rel=0.05)
    t = _run(sc, -3.0, 60, t)              # 3 graus para cima (< 4 graus): mantem
    assert sc.rpm == pytest.approx(r, rel=0.05)
    t = _run(sc, -6.0, 40, t)              # um pouco para cima (> 4 graus): para
    assert sc.rpm == sc.cfg.min_rpm
