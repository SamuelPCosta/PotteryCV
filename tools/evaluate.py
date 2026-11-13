"""Avaliação (specs/evaluation.md) a partir de sessões gravadas. Resultados em data/results/.

    python tools/evaluate.py fps      data/sessions/<pasta>         # CA-01: FPS e ms por etapa
    python tools/evaluate.py contour  data/sessions/<pasta> [--extractors landmarks mobilesam]   # E1
    python tools/evaluate.py jitter   data/sessions/<pasta> ...     # E2 (agrupa por suavização ligada/desligada)
    python tools/evaluate.py depth    data/sessions/<pasta> ...     # E3 (agrupa por normalização ligada/desligada)
    python tools/evaluate.py latency  data/sessions/<pasta> ...     # E4
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pottery  # noqa: E402,F401  (antes de cv2/Qt — ver pottery/__init__.py)
import cv2  # noqa: E402


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stats(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return "n=0"
    return f"n={len(x)} média={x.mean():.2f} mediana={np.median(x):.2f} p95={np.percentile(x, 95):.2f} dp={x.std():.2f}"


def out_csv(name, header, data):
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    print(f"→ {p}")


# ---------------------------------------------------------------- CA-01
def cmd_fps(sessions):
    for s in sessions:
        r = [x for x in rows(Path(s) / "frames.csv") if x["state"] == "MODELING"] or rows(Path(s) / "frames.csv")
        ts = np.array([int(x["t_capture_ns"]) for x in r]) / 1e9
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0
        print(f"\n{s}\n  FPS médio (capturas processadas) = {fps:.1f}  {'OK ≥ 20' if fps >= 20 else 'ABAIXO DE 20'}")
        for k in ("ms_hands", "ms_face", "ms_contour", "ms_sim", "ms_total"):
            print(f"  {k:10s} {stats([float(x[k]) for x in r])}")


# ---------------------------------------------------------------- E1
def cmd_contour(sessions, extractors):
    from pottery.calibration import CalibProfile
    from pottery.config import load_config
    from pottery import contour as ct
    from pottery.tracking import HandTracker
    cfg = load_config(ROOT / "config.toml")
    out = []
    for s in sessions:
        s = Path(s)
        ann = {int(p.stem): p for p in (s / "annot").glob("*.png")}
        if not ann:
            print(f"{s}: sem anotações (rode tools/annotate.py)")
            continue
        ids = [int(r["frame_id"]) for r in rows(s / "frames_ts.csv")]
        ts = [int(r["ts_ns"]) for r in rows(s / "frames_ts.csv")]
        prof_p = ROOT / cfg.calib.profile_path
        cap = cv2.VideoCapture(str(s / "video.mp4"))
        frames = {}
        for i, fid in enumerate(ids):
            ok, img = cap.read()
            if not ok:
                break
            if fid in ann:
                frames[fid] = (img, ts[i])
        prof = CalibProfile.load(prof_p, 1280, 720, True)[0] if prof_p.exists() else CalibProfile.default(1280, 720)
        for ex in extractors:
            ht = HandTracker(cfg.models_dir, cfg.hand)   # novo por extrator (modo vídeo exige ts crescente)
            seg = ct.SamSegmenter(ex, cfg.models_dir, cfg.seg.device, cfg.seg.roi_pad, 999) if ex in ct._SAM_WEIGHTS else None
            for fid, (img, t) in sorted(frames.items()):
                lab = cv2.imread(str(ann[fid]), cv2.IMREAD_GRAYSCALE)
                hands = ht.process(np.ascontiguousarray(img[:, :, ::-1]), t)
                t0 = time.perf_counter()
                masks = []
                if seg is not None:
                    seg.request(fid, img, hands)
                    t_lim = time.time() + 5
                    while time.time() < t_lim and any(seg.get(h, fid, img.shape) is None or seg.get(h, fid, img.shape).age for h in hands):
                        time.sleep(0.005)
                for h in hands:
                    k = prof.kappa_for(h.handedness)
                    if seg is not None:
                        hm = seg.get(h, fid, img.shape)
                    else:
                        hm = ct.landmarks_mask(h, img.shape, cfg.hand.finger_radius, k)
                    masks.append(hm)
                ms = (time.perf_counter() - t0) * 1000
                for hm in masks:
                    if hm is None:
                        continue
                    cl = ct.clean_mask(hm)
                    pred = np.zeros(lab.shape, np.uint8)
                    if cl is not None:
                        cv2.fillPoly(pred, [cl[1].astype(np.int32)], 1)   # contorno real (com os dedos)
                    ids_gt = [v for v in np.unique(lab[pred > 0]) if v > 0]
                    if not ids_gt:
                        out.append([s.name, fid, ex, 0.0, np.nan, ms])
                        continue
                    g = max(ids_gt, key=lambda v: ((lab == v) & (pred > 0)).sum())
                    ghm = (lab == g).astype(np.uint8)
                    iou = ((pred & ghm).sum()) / max(((pred | ghm).sum()), 1)
                    # Chamfer simétrico entre contornos (px)
                    c1 = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                    c2 = cv2.findContours(ghm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                    ch = np.nan
                    if c1 and c2:
                        a, b = c1[0].reshape(-1, 2)[::3].astype(float), c2[0].reshape(-1, 2)[::3].astype(float)
                        d = np.linalg.norm(a[:, None] - b[None], axis=2)
                        ch = (d.min(1).mean() + d.min(0).mean()) / 2
                    out.append([s.name, fid, ex, iou, ch, ms])
            ht.close()
            if seg:
                seg.close()
    print("\nE1 — contorno da mão vs anotação manual")
    for ex in extractors:
        r = [o for o in out if o[2] == ex]
        print(f"  {ex:10s} IoU {stats([o[3] for o in r])}\n  {'':10s} Chamfer px {stats([o[4] for o in r])}\n  {'':10s} ms {stats([o[5] for o in r])}")
    out_csv("E1_contour", ["session", "frame_id", "extractor", "iou", "chamfer_px", "ms"], out)


# ---------------------------------------------------------------- E2 / E3
def _hand_rows(s):
    fr = {int(r["frame_id"]): r for r in rows(Path(s) / "frames.csv")}
    for r in rows(Path(s) / "hands.csv"):
        if not r["r_near"]:
            continue
        v = np.array([float(x) if x else np.nan for x in r["r_near"].split(";")])
        yield r, fr.get(int(r["frame_id"]), {}), v


def cmd_jitter(sessions):
    groups = {}
    for s in sessions:
        per_edge = {}
        for r, f, v in _hand_rows(s):
            per_edge.setdefault((f.get("smoothing_on", "?"), r["edge"]), []).append(v)
        for (sm, e), vs in per_edge.items():
            A = np.stack(vs)
            ok = np.mean(np.isfinite(A), 0) > 0.8          # bins cobertos quase sempre
            if ok.sum() == 0:
                continue
            jit = np.nanstd(A[:, ok], 0).mean() * 1000
            groups.setdefault(sm, []).append(jit)
            print(f"  {Path(s).name} borda {e} suavização={sm}: jitter = {jit:.2f} mm ({len(vs)} quadros)")
    print("\nE2 — jitter médio do perfil (dp temporal de r_near, mm)")
    for sm, j in groups.items():
        print(f"  suavização={sm}: {stats(j)}")
    out_csv("E2_jitter", ["smoothing_on", "jitter_mm"], [[k, x] for k, v in groups.items() for x in v])


def cmd_depth(sessions):
    data = []
    for s in sessions:
        for r, f, v in _hand_rows(s):
            if np.isfinite(v).sum() < 5:
                continue
            data.append((f.get("depthnorm_on", "?"), float(r["s"]), float(np.nanmedian(v)) * 1000))
    print("\nE3 — raio medido (mediana em z) vs escala s; mão deslizando a X fixo")
    res = []
    for dn in sorted({d[0] for d in data}):
        sv = np.array([d[1] for d in data if d[0] == dn])
        rv = np.array([d[2] for d in data if d[0] == dn])
        slope = np.polyfit(sv, rv, 1)[0] if len(sv) > 2 else np.nan
        print(f"  normalização={dn}: raio {stats(rv)} mm | variação (p95-p5) {np.percentile(rv, 95) - np.percentile(rv, 5):.1f} mm"
              f" | inclinação dr/ds {slope:.1f} mm por unidade de escala")
        res += [[dn, a, b] for a, b in zip(sv, rv)]
    out_csv("E3_depth", ["depthnorm_on", "s", "r_mm"], res)


# ---------------------------------------------------------------- E4
def cmd_latency(sessions):
    send, net, app, tot = [], [], [], []
    for s in sessions:
        for r in rows(Path(s) / "latency.csv"):
            tc, ts, tr, ta = (int(r[k]) for k in ("t_capture_ns", "t_send_ns", "t_recv_ns", "t_applied_ns"))
            send.append((ts - tc) / 1e6)
            net.append((tr - ts) / 1e6)
            app.append((ta - tr) / 1e6)
            tot.append((ta - tc) / 1e6)
    print("\nE4 — latência (ms)")
    for n, x in (("captura→envio", send), ("envio→recepção", net), ("recepção→aplicado", app), ("captura→aplicado", tot)):
        print(f"  {n:18s} {stats(x)}")
    out_csv("E4_latency", ["capture_send", "send_recv", "recv_applied", "capture_applied"], list(zip(send, net, app, tot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fps", "contour", "jitter", "depth", "latency"])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--extractors", nargs="+", default=["landmarks", "mobilesam"])
    a = ap.parse_args()
    if a.cmd == "contour":
        cmd_contour(a.sessions, a.extractors)
    else:
        {"fps": cmd_fps, "jitter": cmd_jitter, "depth": cmd_depth, "latency": cmd_latency}[a.cmd](a.sessions)


if __name__ == "__main__":
    main()
