"""
Rodar o script (uma vez) registra o painel "Pottery" na barra lateral do viewport (N) e liga o receptor;
rodar de novo recarrega o codigo e reinicia. O painel tem Ligar/Desligar, o intervalo da vista e o status.
- Cria/valida o objeto `PotteryVase`: malha de DUAS paredes (externa + interna), borda e fundos fechados.
- Atributos de ponto: `radius` (atualizado), `theta`, `zn` (0..1), `wall` (0 externa, 1 interna).
- Propriedade do objeto `pottery_height` (H em m) - ligue o input Height do Geometry Nodes a ela por driver.
- Escuta UDP :9000 (protocolo specs/protocol.md), aplica a rotacao Z = phi e devolve ECHO para :9001.
- Com a flag F_VIEW (Python marcou "Vista do Blender"): a cada N estados (painel) grava a vista da camera em
  <projeto>/data/blender_view.jpg; a interface Python mostra.
"""
import math
import os
import socket
import struct
import tempfile
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
F_VIEW = 16                      # flag: a interface pediu a vista da camera
VIEW_EVERY = 5                   # padrao do painel: grava a vista a cada N estados aplicados
VIEW_MAX = 480                   # lado maior do JPEG (px)
VIEW_NAME = "blender_view.jpg"   # em <projeto>/data/ (o .blend fica em <projeto>/blender/); igual ao pottery/net.py


def _log(msg):
    """print + <projeto>/data/blender_receiver.log (para ver o que aconteceu sem o console do Blender)."""
    print(msg)
    try:
        with open(os.path.join(os.path.dirname(_view_path()), "blender_receiver.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def _view_path():
    blend = bpy.data.filepath
    root = os.path.dirname(os.path.dirname(blend)) if blend else tempfile.gettempdir()
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    return os.path.join(root, "data", VIEW_NAME)

HDR = struct.Struct("<4sBBHIIQQHHHHBBHffffI")   # 64 B
ECHO = struct.Struct("<4sBBHIIQQQQ")             # 48 B


def decode_state(data: bytes):
    if len(data) < HDR.size or data[:4] != MAGIC or data[4] != 1 or data[5] != 1:
        return None
    (_, _, _, flags, seq, fid, tc, ts, nt, nz, c0, cc, ci, cn, qw, H, rs, phi, om, ver) = HDR.unpack_from(data)
    nr = cc * nz * 2
    if c0 + cc > nt or len(data) not in (HDR.size + nr, HDR.size + nr + cc * 2):
        return None
    q = np.frombuffer(data, "<u2", count=cc * nz, offset=HDR.size)
    if len(data) == HDR.size + nr + cc * 2:      # altura relativa de cada coluna (0..1 x H)
        hf = np.frombuffer(data, "<u2", count=cc, offset=HDR.size + nr).astype(np.float32) / 65535.0
    else:
        hf = np.ones(cc, np.float32)
    return dict(flags=flags, seq=seq, frame_id=fid, t_capture_ns=tc, t_send_ns=ts, n_theta=nt, n_z=nz, col_start=c0,
                col_count=cc, chunk_index=ci, height=H, phi=phi, omega=om, version=ver, wetness=qw / 65535.0,
                R=q.astype(np.float32) * (rs / 65535.0), hfrac=hf)


def seq_newer(a, b):
    return ((a - b) & 0xFFFFFFFF) < 0x80000000


def build_topology(nt: int, nz: int):
    """Vertices: externa k=i*nz+j (0..N-1), interna N+k, centro do fundo externo 2N, interno 2N+1."""
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
    """zn por vertice = altura relativa da coluna x j/(nz-1). O GN faz z = zn * Height (topo varia por coluna)."""
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
        self.n_view = 0                  # estados aplicados desde a ultima vista gravada
        self.want_view = False           # o proximo redesenho do viewport grava a vista
        self.view_ms = float("nan")
        self.offscreen = None
        self.view_path = _view_path()
        self.view_said = False           # avisa no console a primeira vista gravada
        self.last_err = ""
        self.cam_warned = False
        self.view_mode = ""
        self.vs = dict(pedidas=0, feitas=0, falhas=0, draw=0.0, read=0.0, set=0.0, save=0.0)
        self.rate, self.lat_med = 0.0, float("nan")   # para o painel
        _log(f"[pottery] escutando UDP {PORT}; vista da camera em {self.view_path}")

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
        try:
            self.obj.name                    # Ctrl+Z / recarregar o arquivo recria o objeto: a referencia morre
        except ReferenceError:
            self.obj, self.version = _ensure_object(), -1   # busca de novo pelo nome e reenvia tudo
        obj = self.obj
        if any_d["flags"] & F_VIEW:              # antes da malha: um erro na malha nao segura a vista
            if not self.view_said:
                _log("[pottery] interface pediu a vista da camera")
                self.view_said = True
            self.n_view += 1
            if self.n_view >= getattr(bpy.context.scene, "pottery_view_every", VIEW_EVERY):
                self.n_view, self.want_view = 0, True
                self.vs["pedidas"] += 1
                _redraw_views()                   # o desenho acontece no draw handler (precisa do contexto do viewport)
        if any_d["version"] != self.version:
            ra, za = obj.data.attributes.get("radius"), obj.data.attributes.get("zn")
            need = 2 * N_THETA * N_Z + 2
            if ra is None or za is None or len(ra.data) != need or len(za.data) != need:
                # Edit Mode (atributos vazios) ou malha trocada: pula a malha, tenta de novo no proximo estado
                msg = (f"[pottery] malha nao atualizada: modo {obj.mode}, {len(obj.data.vertices)} vertices, "
                       f"radius {None if ra is None else (ra.domain, len(ra.data))}, esperado {need}")
                if msg != self.last_err:
                    _log(msg)
                    self.last_err = msg
            else:
                self.version = any_d["version"]
                ra.data.foreach_set("value", radius_array(self.R))
                za.data.foreach_set("value", zn_array(self.HF, N_Z))                      # topo por coluna
                obj.data.update()                                                         # VERIFICAR: reavalia o GN?
        changed = False
        if abs(obj.get("pottery_height", 0) - any_d["height"]) > 1e-6:
            obj["pottery_height"] = any_d["height"]
            changed = True
        if abs(obj.get("wetness", -1.0) - any_d["wetness"]) > 1e-3:   # material le com Attribute (Object) "wetness"
            obj["wetness"] = any_d["wetness"]
            changed = True
        if changed:
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
                  f"{np.median(self.lat):.1f} ms, H={any_d['height'] * 100:.1f} cm"
                  + (f", vista {self.view_ms:.0f} ms" if self.view_ms == self.view_ms else ""))
            v = self.vs
            if v["pedidas"]:                 # diagnostico da vista: quantas pedidas/gravadas e o tempo de cada etapa
                n = max(1, v["feitas"])
                _log(f"[pottery] vista em {now - self.t_stat:.1f} s: {self.packets} estados, {v['pedidas']} pedidas, "
                     f"{v['feitas']} desenhadas, {v['falhas']} replace falhou | ms medios: recorte {v['draw'] / n:.0f} "
                     f"leitura {v['read'] / n:.0f} pixels {v['set'] / n:.0f} jpeg {v['save'] / n:.0f}")
            self.vs = dict.fromkeys(self.vs, 0)
            self.rate, self.lat_med = self.packets / (now - self.t_stat), float(np.median(self.lat))
            self.packets, self.lat, self.t_stat = 0, [], now
            _redraw_views()                  # atualiza o status no painel

    def save_view(self, context):
        """Desenha a cena pela camera ativa num offscreen e grava o JPEG (troca atomica do arquivo)."""
        import gpu
        t0 = time.perf_counter()
        scene = context.scene
        cam = scene.camera
        if cam is None:
            _log("[pottery] vista: a cena nao tem camera ativa")
            return
        # copia o que o viewport JA desenhou (redesenhar a cena num offscreen custava ~250 ms no modo Rendered).
        # Com o viewport na vista da camera (Numpad 0) recorta so a moldura da camera; senao, o viewport inteiro.
        region, rv3d = context.region, context.region_data
        x0, y0, w, h = 0, 0, region.width, region.height
        if rv3d is not None and rv3d.view_perspective == "CAMERA":
            from bpy_extras.view3d_utils import location_3d_to_region_2d
            pts = [location_3d_to_region_2d(region, rv3d, cam.matrix_world @ c) for c in cam.data.view_frame(scene=scene)]
            if all(p is not None for p in pts):
                x0, y0 = max(0, int(min(p.x for p in pts))), max(0, int(min(p.y for p in pts)))
                w = min(region.width, int(max(p.x for p in pts))) - x0
                h = min(region.height, int(max(p.y for p in pts))) - y0
        elif not self.cam_warned:
            _log("[pottery] vista: o viewport nao esta na vista da camera (Numpad 0) - capturando o viewport inteiro")
            self.cam_warned = True
        if w < 16 or h < 16:
            return
        t1 = time.perf_counter()
        fb = gpu.state.active_framebuffer_get()     # no draw handler: o framebuffer do proprio viewport
        probe = fb.read_color(x0 + w // 2 - 2, y0 + h // 2 - 2, 4, 4, 4, 0, "FLOAT").to_list()
        if max(max(p) for row in probe for p in row) > 1e-4:
            mode = "copia do viewport"
            buf = fb.read_color(x0, y0, w, h, 4, 0, "FLOAT")
        else:
            # Rendered/Material Preview (EEVEE): o framebuffer do draw handler vem vazio (a imagem do render e
            # composta depois) -> redesenha pela camera num offscreen (~250 ms; use VIEW_EVERY maior ou Solid)
            mode = "redesenho (viewport em Rendered)"
            rw, rh = scene.render.resolution_x, scene.render.resolution_y
            f = VIEW_MAX / max(rw, rh)
            w, h = max(16, int(rw * f)), max(16, int(rh * f))
            if self.offscreen is None or (self.offscreen.width, self.offscreen.height) != (w, h):
                self.offscreen = gpu.types.GPUOffScreen(w, h)
            proj = cam.calc_matrix_camera(context.evaluated_depsgraph_get(), x=w, y=h,
                                          scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)
            self.offscreen.draw_view3d(scene, context.view_layer, context.space_data, region,
                                       cam.matrix_world.inverted(), proj, do_color_management=True)
            with self.offscreen.bind():
                buf = gpu.state.active_framebuffer_get().read_color(0, 0, w, h, 4, 0, "FLOAT")
        if mode != self.view_mode:
            _log(f"[pottery] vista: modo {mode}")
            self.view_mode = mode
        buf.dimensions = w * h * 4
        t2 = time.perf_counter()
        if self.view_ms != self.view_ms:     # primeira vista: registra o brilho para diagnostico
            px = buf.to_list()
            _log(f"[pottery] vista: pixels min {min(px):.3f} max {max(px):.3f} media {sum(px) / len(px):.3f}")
        img = bpy.data.images.get("PotteryView")
        if img is None:
            img = bpy.data.images.new("PotteryView", w, h, alpha=False)
        elif tuple(img.size) != (w, h):
            img.scale(w, h)
        img.pixels.foreach_set(buf)         # framebuffer e Image comecam embaixo: nao precisa inverter
        f = VIEW_MAX / max(w, h)
        if f < 1:
            img.scale(max(16, int(w * f)), max(16, int(h * f)))   # JPEG menor: grava e le mais rapido
        img.update()
        t3 = time.perf_counter()
        tmp = self.view_path + ".tmp.jpg"
        img.filepath_raw = tmp
        img.file_format = "JPEG"
        img.save()                                                                # VERIFICAR
        try:
            os.replace(tmp, self.view_path)  # a interface nunca le o arquivo pela metade
        except PermissionError:
            self.vs["falhas"] += 1          # arquivo aberto por outro processo: fica para a proxima
        t4 = time.perf_counter()
        v = self.vs
        v["feitas"] += 1
        v["draw"] += (t1 - t0) * 1000
        v["read"] += (t2 - t1) * 1000
        v["set"] += (t3 - t2) * 1000
        v["save"] += (t4 - t3) * 1000
        if self.view_ms != self.view_ms:
            _log(f"[pottery] primeira vista gravada ({w}x{h})")
        self.view_ms = (time.perf_counter() - t0) * 1000

    def close(self):
        self.sock.close()
        if self.offscreen is not None:
            self.offscreen.free()


def _timer():
    rx = bpy.app.driver_namespace.get("pottery_rx")
    if rx is None:
        return None
    try:
        rx.tick()
    except Exception as e:
        msg = f"[pottery] erro: {type(e).__name__}: {e}"
        if msg != rx.last_err:               # o mesmo erro a cada tick: registra so uma vez
            _log(msg)
            rx.last_err = msg
    return TIMER_S

def _redraw_views():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _draw_view():
    """Draw handler do viewport 3D: so aqui existe o contexto de desenho que o offscreen precisa."""
    rx = bpy.app.driver_namespace.get("pottery_rx")
    if rx is None or not rx.want_view:
        return
    rx.want_view = False
    if rx.view_ms != rx.view_ms:
        _log("[pottery] draw handler chamado: desenhando a vista")
    try:
        rx.save_view(bpy.context)
    except Exception as e:
        _log(f"[pottery] erro na vista: {type(e).__name__}: {e}")


def stop():
    ns = bpy.app.driver_namespace
    h = ns.pop("pottery_view_handle", None)
    if h is not None:
        bpy.types.SpaceView3D.draw_handler_remove(h, "WINDOW")
    t = ns.pop("pottery_timer", None)         # o timer de uma execucao anterior do script (outra funcao)
    if t is not None and bpy.app.timers.is_registered(t):
        bpy.app.timers.unregister(t)
    rx = ns.pop("pottery_rx", None)
    if rx is not None:
        rx.close()
        _log("[pottery] receptor PARADO")


def start():
    stop()
    ns = bpy.app.driver_namespace
    ns["pottery_rx"] = _Rx()
    ns["pottery_view_handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), "WINDOW", "POST_PIXEL")
    ns["pottery_timer"] = _timer
    bpy.app.timers.register(_timer, first_interval=0.0, persistent=True)
    _log("[pottery] receptor LIGADO")


def running() -> bool:
    return bpy.app.driver_namespace.get("pottery_rx") is not None


# ---------------------------------------------------------------- painel (barra lateral N > Pottery)
if bpy is not None:
    class POTTERY_OT_toggle(bpy.types.Operator):
        bl_idname = "pottery.toggle"
        bl_label = "Ligar/desligar o receptor"

        def execute(self, context):
            stop() if running() else start()
            _redraw_views()
            return {"FINISHED"}

    class POTTERY_PT_panel(bpy.types.Panel):
        bl_label = "PotteryCV"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Pottery"

        def draw(self, context):
            lay = self.layout
            rx = bpy.app.driver_namespace.get("pottery_rx")
            on = rx is not None
            lay.operator("pottery.toggle", text="Desligar receptor" if on else "Ligar receptor",
                         icon="PAUSE" if on else "PLAY", depress=on)
            col = lay.column(align=True)
            if on:
                col.label(text=f"{rx.rate:.0f} estados/s   latência {rx.lat_med:.0f} ms"
                          if rx.lat_med == rx.lat_med else "aguardando o Python…")
            else:
                col.label(text="desligado")
            lay.separator()
            lay.label(text="Vista da câmera (para a interface)")
            lay.prop(context.scene, "pottery_view_every")
            if on and rx.view_ms == rx.view_ms:
                lay.label(text=f"{rx.view_ms:.0f} ms por imagem ({rx.view_mode})")
                if rx.view_ms > 100:
                    lay.label(text="lento: viewport em Solid ou N maior", icon="INFO")

    _CLASSES = (POTTERY_OT_toggle, POTTERY_PT_panel)


def register():
    bpy.types.Scene.pottery_view_every = bpy.props.IntProperty(
        name="A cada N estados", default=VIEW_EVERY, min=1, max=60,
        description="grava a vista da câmera a cada N estados recebidos (o Python manda ~30 por segundo)")
    for c in _CLASSES:
        old = getattr(bpy.types, c.__name__, None)   # versao de uma execucao anterior do script
        if old is not None:
            bpy.utils.unregister_class(old)
        bpy.utils.register_class(c)


if __name__ == "__main__" and bpy is not None:
    register()
    start()                                   # rodar de novo recarrega o codigo e reinicia o receptor
