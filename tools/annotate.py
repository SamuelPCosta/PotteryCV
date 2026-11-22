"""Anotacao manual de mascaras de mao (poligono) sobre quadros de uma sessao gravada - referencia do E1.

    python tools/annotate.py data/sessions/<pasta> --n 40

Clique esquerdo: adiciona ponto * clique direito ou Enter: fecha o poligono da mao atual
U: desfaz ponto * N: proximo quadro (salva) * S: pula quadro * Q: sai
Saida: <sessao>/annot/<frame_id>.png  (0 = fundo, 1 = 1a mao, 2 = 2a mao)
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--n", type=int, default=40, help="quadros amostrados uniformemente")
    a = ap.parse_args()
    ses = Path(a.session)
    out = ses / "annot"
    out.mkdir(exist_ok=True)
    with open(ses / "frames_ts.csv", newline="") as f:
        ids = [int(r["frame_id"]) for r in csv.DictReader(f)]
    picks = set(np.linspace(0, len(ids) - 1, a.n).astype(int).tolist())
    cap = cv2.VideoCapture(str(ses / "video.mp4"))
    st = {"pts": [], "polys": []}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["pts"].append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and len(st["pts"]) >= 3:
            st["polys"].append(st["pts"])
            st["pts"] = []

    cv2.namedWindow("anotar", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("anotar", on_mouse)
    i = -1
    while True:
        ok, img = cap.read()
        i += 1
        if not ok:
            break
        if i not in picks or (out / f"{ids[i]}.png").exists():
            continue
        st["pts"], st["polys"] = [], []
        while True:
            vis = img.copy()
            for k, poly in enumerate(st["polys"]):
                cv2.polylines(vis, [np.array(poly)], True, (0, 255, 0) if k == 0 else (255, 120, 0), 2)
            if st["pts"]:
                cv2.polylines(vis, [np.array(st["pts"])], False, (0, 255, 255), 1)
                for p in st["pts"]:
                    cv2.circle(vis, p, 3, (0, 255, 255), -1)
            cv2.putText(vis, f"quadro {ids[i]}  maos {len(st['polys'])}  [esq] ponto [dir/Enter] fecha  U N S Q",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("anotar", vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and len(st["pts"]) >= 3:
                st["polys"].append(st["pts"])
                st["pts"] = []
            elif k in (ord("u"), ord("U")) and st["pts"]:
                st["pts"].pop()
            elif k in (ord("n"), ord("N")):
                lab = np.zeros(img.shape[:2], np.uint8)
                for j, poly in enumerate(st["polys"]):
                    cv2.fillPoly(lab, [np.array(poly)], j + 1)
                cv2.imwrite(str(out / f"{ids[i]}.png"), lab)
                break
            elif k in (ord("s"), ord("S")):
                break
            elif k in (ord("q"), ord("Q")):
                return
    print(f"anotações em {out}")


if __name__ == "__main__":
    main()
