import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from pottery.net import Receiver, encode_state

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pottery_receiver", ROOT / "blender" / "pottery_receiver.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def test_blender_decode_matches_python_encoder():
    R = np.random.default_rng(2).uniform(0.01, 0.2, (128, 128)).astype(np.float32)
    hf = np.linspace(0.3, 1.0, 128)
    d = br.decode_state(bytes(encode_state(R, 0.2, 1.0, 2.0, 7, 3, 99, 5, hfrac=hf)[0]))
    assert d["seq"] == 7 and d["height"] == np.float32(0.2)
    assert np.abs(d["R"].reshape(128, 128) - R).max() <= 0.3 / 65535
    assert np.abs(d["hfrac"] - hf).max() <= 1 / 65535
    zn = br.zn_array(d["hfrac"], 128)
    assert len(zn) == 2 * 128 * 128 + 2 and zn[127] == pytest.approx(hf[0], abs=1e-4)


def test_blender_decode_old_format_without_heights():
    R = np.full((8, 4), 0.05, np.float32)
    buf = bytes(encode_state(R, 0.2, 0, 0, 1, 1, 1, 1)[0])
    old = buf[:64 + 8 * 4 * 2]                            # pacote antigo: so os raios
    assert (br.decode_state(old)["hfrac"] == 1).all()


def test_blender_topology():
    faces, theta, zn, wall = br.build_topology(8, 5)
    nv = 2 * 8 * 5 + 2
    assert len(theta) == len(zn) == len(wall) == nv
    assert all(0 <= v < nv for f in faces for v in f)
    # malha fechada (variedade sem borda): cada aresta em exatamente 2 faces
    edges = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[frozenset((a, b))] = edges.get(frozenset((a, b)), 0) + 1
    assert set(edges.values()) == {2}
    assert len(br.radius_array(np.ones((8, 5)))) == nv


def test_synthetic_sender_to_receiver_with_drop():  # CA-09
    rx = Receiver(9123)
    p = subprocess.Popen([sys.executable, str(ROOT / "tools" / "synthetic_sender.py"), "--seconds", "2",
                          "--drop", "0.2", "--res", "256x256", "--targets", "127.0.0.1:9123"],
                         stdout=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 4 and p.poll() is None:
        rx.poll()
        time.sleep(0.01)
    rx.poll()
    p.wait(5)
    rx.close()
    assert rx.R is not None and rx.R.shape == (256, 256)
    assert rx.packets > 20
    assert 0.02 < rx.R.mean() < 0.2
