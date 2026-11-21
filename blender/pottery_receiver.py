"""
Rodar o script de novo LIGA/DESLIGA o receptor.
- Cria/valida o objeto `PotteryVase`: malha de DUAS paredes (externa + interna), borda e fundos fechados.
- Atributos de ponto: `radius` (atualizado), `theta`, `zn` (0..1), `wall` (0 externa, 1 interna).
- Propriedade do objeto `pottery_height` (H em m) — ligue o input Height do Geometry Nodes a ela por driver.
- Escuta UDP :9000 (protocolo specs/protocol.md), aplica a rotação Z = φ e devolve ECHO para :9001.
"""
import math
import socket
import struct
import time

import numpy as np

try:
    import bpy  # noqa
except ImportError:
    bpy = None

# ---------------------------------------------------------------- configuracao
PORT = 9000
ECHO_TO = ("127.0.0.1", 9001)
OBJ_NAME = "PotteryVase"
N_THETA, N_Z = 96, 64            # tem que bater com o Python
TIMER_S = 1 / 60
R0_INIT = 0.06

MAGIC = b"PCV1"
HDR = struct.Struct("<4sBBHIIQQHHHHBBHffffI")   # 64 B
ECHO = struct.Struct("<4sBBHIIQQQQ")             # 48 B


def decode_state(data: bytes):
    if len(data) < HDR.size or data[:4] != MAGIC or data[4] != 1 or data[5] != 1:
        return None
    (_, _, _, flags, seq, fid, tc, ts, nt, nz, c0, cc, ci, cn, _, H, rs, phi, om, ver) = HDR.unpack_from(data)
    nr = cc * nz * 2
    if c0 + cc > nt or len(data) not in (HDR.size + nr, HDR.size + nr + cc * 2):
        return None
    q = np.frombuffer(data, "<u2", count=cc * nz, offset=HDR.size)
    if len(data) == HDR.size + nr + cc * 2:      # altura relativa de cada coluna (0..1 × H)
        hf = np.frombuffer(data, "<u2", count=cc, offset=HDR.size + nr).astype(np.float32) / 65535.0
    else:
        hf = np.ones(cc, np.float32)
    return dict(seq=seq, frame_id=fid, t_capture_ns=tc, t_send_ns=ts, n_theta=nt, n_z=nz, col_start=c0,
                col_count=cc, chunk_index=ci, height=H, phi=phi, omega=om, version=ver,
                R=q.astype(np.float32) * (rs / 65535.0), hfrac=hf)


def seq_newer(a, b):
    return ((a - b) & 0xFFFFFFFF) < 0x80000000


def build_topology(nt: int, nz: int):
    """Vértices: externa k=i*nz+j (0..N-1), interna N+k, centro do fundo externo 2N, interno 2N+1."""
    N = nt * nz
    faces = []
    for i in range(nt):
        i1 = (i + 1) % nt
        for j in range(nz - 1):
            a, b = i * nz + j, i1 * nz + j
            faces.append((a, b, b + 1, a + 1))                 # externa (normal para fora)
            faces.append((N + a + 1, N + b + 1, N + b, N + a))  # interna (normal para dentro do vaso)
        t, t1 = i * nz + nz - 1, i1 * nz + nz - 1
        faces.append((t, t1, N + t1, N + t))                    # borda do topo
        faces.append((2 * N, i1 * nz, i * nz))                  # fundo externo
        faces.append((2 * N + 1, N + i * nz, N + i1 * nz))      # fundo interno
    theta = np.tile(np.repeat(2 * np.pi * np.arange(nt) / nt, nz), 2)
    zn = np.tile(np.tile(np.linspace(0, 1, nz), nt), 2)
    wall = np.concatenate([np.zeros(N), np.ones(N)])
    theta = np.concatenate([theta, [0, 0]]).astype(np.float32)
    zn = np.concatenate([zn, [0, 0]]).astype(np.float32)
    wall = np.concatenate([wall, [0, 1]]).astype(np.float32)
    return faces, theta, zn, wall


def radius_array(R: np.ndarray) -> np.ndarray:
    flat = R.reshape(-1)
    return np.concatenate([flat, flat, [0.0, 0.0]]).astype(np.float32)


def zn_array(hf: np.ndarray, nz: int) -> np.ndarray:
    """zn por vértice = altura relativa da coluna × j/(nz−1). O GN faz z = zn · Height (topo varia por coluna)."""
    flat = (hf[:, None] * np.linspace(0, 1, nz)[None]).reshape(-1)
    return np.concatenate([flat, flat, [0.0, 0.0]]).astype(np.float32)


# ---------------------------------------------------------------- Blender
def _ensure_object():
    nt, nz = N_THETA, N_Z
    nverts = 2 * nt * nz + 2
    obj = bpy.data.objects.get(OBJ_NAME)
    if obj is not None and obj.type == "MESH" and len(obj.data.vertices) == nverts and "radius" in obj.data.attributes:
        return obj
    faces, theta, zn, wall = build_topology(nt, nz)
    r = np.full(nverts, R0_INIT, np.float32)
    r[-2:] = 0
    co = np.stack([r * np.cos(theta), r * np.sin(theta), zn * 0.15], 1)
    mesh = bpy.data.meshes.new(OBJ_NAME + "Mesh")
    mesh.from_pydata(co.tolist(), [], faces)                                       # VERIFICAR
    for name, vals in (("radius", r), ("theta", theta), ("zn", zn), ("wall", wall)):
        att = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")         # VERIFICAR
        att.data.foreach_set("value", vals)                                        # VERIFICAR
    mesh.update()
    if obj is None:
        obj = bpy.data.objects.new(OBJ_NAME, mesh)
        bpy.context.scene.collection.objects.link(obj)                             # VERIFICAR
    else:
        obj.data = mesh
    obj["pottery_height"] = 0.15
    print(f"[pottery] malha criada: {nverts} vértices")
    return obj


class _Rx:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", PORT))
        self.sock.setblocking(False)
        self.obj = _ensure_object()
        self.R = np.full((N_THETA, N_Z), R0_INIT, np.float32)
        self.HF = np.ones(N_THETA, np.float32)
        self.chunk_seq = {}
        self.version = -1
        self.packets = 0
        self.lat = []
        self.t_stat = time.monotonic()
        self.warned = False
        print(f"[pottery] escutando UDP {PORT}")

    def tick(self):
        latest = {}
        while True:
            try:
                data, _ = self.sock.recvfrom(65507)
            except (BlockingIOError, ConnectionResetError, OSError):
                break
            d = decode_state(data)
            if d is None:
                continue
            d["t_recv_ns"] = time.time_ns()
            ci = d["chunk_index"]
            if ci not in latest or seq_newer(d["seq"], latest[ci]["seq"]):
                latest[ci] = d
        if not latest:
            return
        any_d = None
        for ci, d in latest.items():
            if (d["n_theta"], d["n_z"]) != (N_THETA, N_Z):
                if not self.warned:
                    print(f"[pottery] malha incompatível: recebido {d['n_theta']}x{d['n_z']}, esperado {N_THETA}x{N_Z}")
                    self.warned = True
                return
            if ci in self.chunk_seq and not seq_newer(d["seq"], self.chunk_seq[ci]):
                continue
            self.chunk_seq[ci] = d["seq"]
            self.R[d["col_start"]:d["col_start"] + d["col_count"]] = d["R"].reshape(d["col_count"], d["n_z"])
            self.HF[d["col_start"]:d["col_start"] + d["col_count"]] = d["hfrac"]
            any_d = d
        if any_d is None:
            return
        obj = self.obj
        if any_d["version"] != self.version:
            self.version = any_d["version"]
            obj.data.attributes["radius"].data.foreach_set("value", radius_array(self.R))  # VERIFICAR
            obj.data.attributes["zn"].data.foreach_set("value", zn_array(self.HF, N_Z))     # topo por coluna
            obj.data.update()                                                             # VERIFICAR: reavalia o GN?
        if abs(obj.get("pottery_height", 0) - any_d["height"]) > 1e-6:
            obj["pottery_height"] = any_d["height"]
            obj.update_tag()                                                              # VERIFICAR: driver atualiza?
        obj.rotation_euler[2] = any_d["phi"]
        t_applied = time.time_ns()
        self.packets += 1
        self.lat.append((t_applied - any_d["t_capture_ns"]) / 1e6)
        if ECHO_TO:
            try:
                self.sock.sendto(ECHO.pack(MAGIC, 1, 3, 0, any_d["seq"], any_d["frame_id"], any_d["t_capture_ns"],
                                           any_d["t_send_ns"], any_d["t_recv_ns"], t_applied), ECHO_TO)
            except OSError:
                pass
        now = time.monotonic()
        if now - self.t_stat > 2:
            print(f"[pottery] {self.packets / (now - self.t_stat):.1f} estados/s, captura→malha mediana "
                  f"{np.median(self.lat):.1f} ms, H={any_d['height'] * 100:.1f} cm")
            self.packets, self.lat, self.t_stat = 0, [], now

    def close(self):
        self.sock.close()


def _timer():
    rx = bpy.app.driver_namespace.get("pottery_rx")
    if rx is None:
        return None
    try:
        rx.tick()
    except Exception as e:
        print("[pottery] erro:", e)
    return TIMER_S

def toggle():
    ns = bpy.app.driver_namespace
    rx = ns.get("pottery_rx")
    if rx is not None:
        rx.close()
        del ns["pottery_rx"]
        if bpy.app.timers.is_registered(_timer):
            bpy.app.timers.unregister(_timer)
        print("[pottery] receptor PARADO")
        return
    for k in [k for k in ns.keys() if k == "pottery_rx"]:
        del ns[k]
    ns["pottery_rx"] = _Rx()
    bpy.app.timers.register(_timer, first_interval=0.0, persistent=True)
    print("[pottery] receptor LIGADO (rode o script de novo para parar)")


if __name__ == "__main__" and bpy is not None:
    toggle()
