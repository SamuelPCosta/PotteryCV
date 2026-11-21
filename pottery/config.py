"""Configuração: dataclasses com valores padrão, sobrescritos por config.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class CameraCfg:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "msmf"  # msmf | dshow | any
    mirror: bool = True


@dataclass
class VaseCfg:
    n_theta: int = 96
    n_z: int = 64
    r0: float = 0.06          # raio inicial (m)
    h0: float = 0.15          # altura inicial (m)
    r_min: float = 0.008
    r_max: float = 0.18
    h_min: float = 0.01
    h_max: float = 0.40
    r_scale: float = 0.30     # faixa de quantização do protocolo
    alpha: float = 0.6        # plasticidade por passagem
    smooth_lambda: float = 0.5        # espalhamento da marca para os vizinhos (0 = marca seca, 1 = bem espalhada)
    smooth_sigma_theta: float = 1.5
    smooth_sigma_z: float = 1.5
    crush_reset_frac: float = 0.20    # esmagou abaixo de 20% de h0 -> reinicia (com contagem de 3 s)
    feather_m: float = 0.003   # rampa nas pontas de cada dedo (pequena para não apagar a marca do dedo)
    side_gain: float = 0.4    # apertar dos lados -> sobe: 0 = altura fixa, 1 = conserva volume exato
    top_rate: float = 0.5     # rapidez do achatamento ao empurrar o topo (conserva volume: engorda)


@dataclass
class ContactCfg:             # a mão só empurra por fora (único modo)
    engage_m: float = 0.002
    release_m: float = 0.015
    release_frames: int = 3
    lost_ms: float = 200.0


@dataclass
class HandCfg:
    L_real_m: float = 0.095   # pulso -> nó do dedo médio
    kappa_default: float = 1.3
    finger_radius: float = 0.12  # meia largura do dedo (x L): landmarks ficam no centro do dedo
    profile_ema: float = 0.5
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 0.02
    scale_ema: float = 0.2
    smoothing: bool = True
    depth_norm: bool = True
    min_score: float = 0.5


@dataclass
class SpeedCfg:
    # cabeça um pouco para cima -> para; para baixo -> vai acelerando; neutra -> mantém
    accel_deg: float = 4.0    # para baixo além disso: acelera
    brake_deg: float = 4.0    # para cima além disso: freia até parar
    sat_deg: float = 22.0
    gamma: float = 1.5
    dwell_s: float = 0.3
    accel_rpm_s: float = 50.0
    brake_rpm_s: float = 200.0
    max_rpm: float = 90.0
    min_rpm: float = 2.0      # o torno nunca para: mínimo sempre
    yaw_gate_deg: float = 25.0
    face_every: int = 2
    pitch_sign: float = 1.0   # inverter (-1) se olhar para baixo estiver freando


@dataclass
class CalibCfg:
    countdown_s: float = 3.0  # "capturando em 3.. 2.. 1" antes de cada etapa
    dwell_s: float = 1.0
    still_px: float = 6.0
    still_deg: float = 1.5
    profile_path: str = "calibration.json"


@dataclass
class NetCfg:
    enabled: bool = True
    targets: list = field(default_factory=lambda: ["127.0.0.1:9000", "127.0.0.1:9002"])
    rate_hz: float = 30.0
    echo_port: int = 9001
    viewer_port: int = 9002
    max_datagram: int = 65000


@dataclass
class SegCfg:
    extractor: str = "landmarks"  # landmarks | mobilesam
    device: str = "cuda"
    max_age: int = 6
    roi_pad: float = 0.15
    async_threshold_ms: float = 25.0


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    vase: VaseCfg = field(default_factory=VaseCfg)
    contact: ContactCfg = field(default_factory=ContactCfg)
    hand: HandCfg = field(default_factory=HandCfg)
    speed: SpeedCfg = field(default_factory=SpeedCfg)
    calib: CalibCfg = field(default_factory=CalibCfg)
    net: NetCfg = field(default_factory=NetCfg)
    seg: SegCfg = field(default_factory=SegCfg)
    models_dir: str = "models"
    data_dir: str = "data"


def _apply(obj, data: dict):
    names = {f.name for f in fields(obj)}
    for k, v in data.items():
        if k not in names:
            print(f"[config] chave desconhecida ignorada: {k}")
            continue
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _apply(cur, v)
        else:
            setattr(obj, k, type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v)


def load_config(path: str | Path | None = "config.toml") -> Config:
    cfg = Config()
    if path and Path(path).exists():
        with open(path, "rb") as f:
            _apply(cfg, tomllib.load(f))
    return cfg
