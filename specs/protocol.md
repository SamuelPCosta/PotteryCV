# PotteryCV — Protocolo Python → Blender (v1)

## Transporte
- UDP em `127.0.0.1` (sempre a mesma máquina). Little-endian.
- O sender envia para uma lista de destinos: Blender `:9000` e visualizador `:9002`. ECHO opcional de volta para `:9001`.
- Taxa: até 30 estados/s, com keep-alive a 2 Hz quando nada muda.
- **Sempre o estado completo** (idempotente): uma perda é corrigida pelo pacote seguinte, sem retransmissão.

## Por que uint16
O mapa 128 × 128 em float32 tem 65.536 B, acima do limite de 65.507 B de um datagrama. Em uint16 quantizado em [0, r_scale] fica com **32.768 B + 256 B de alturas + 64 B de cabeçalho = 1 datagrama**. O passo é r_scale/65535 (≈ 5 µm com 0,3 m). Para resoluções maiores, o mesmo formato é dividido em blocos de colunas autossuficientes.

## STATE (msg_type 1): cabeçalho de 64 B + payload
| Off | Tipo | Campo |
|---|---|---|
| 0 | 4s | magic `PCV1` |
| 4 | u8 | version = 1 |
| 5 | u8 | msg_type = 1 |
| 6 | u16 | flags (bit0 reset, bit1 calibrando, bit2 pausado, bit3 sintético, bit4 pede a vista da câmera do Blender em JPEG: `data/blender_view.jpg`, a cada 5 estados) |
| 8 | u32 | seq |
| 12 | u32 | frame_id |
| 16 | u64 | t_capture_ns (`time.time_ns()` na captura) |
| 24 | u64 | t_send_ns |
| 32 | u16 | n_theta |
| 34 | u16 | n_z |
| 36 | u16 | col_start |
| 38 | u16 | col_count |
| 40 | u8 | chunk_index |
| 41 | u8 | chunk_count |
| 42 | u16 | wetness (umidade do barro, 0..65535 = 0..1; o Blender grava em `obj["wetness"]`) |
| 44 | f32 | height_m (**H, pode mudar a qualquer momento**) |
| 48 | f32 | r_scale_m |
| 52 | f32 | phi_rad ∈ [0, 2π) |
| 56 | f32 | omega_rad_s |
| 60 | u32 | state_version (muda quando R muda) |
| 64 | u16[] | `col_count × n_z` valores: θ por fora, z por dentro; R = q·r_scale/65535 |
| 64 + 2·col_count·n_z | u16[] | `col_count` alturas relativas de cada coluna: h_i = q·height_m/65535. O topo varia por coluna (pressão local no topo). Se o pacote vier sem essa parte (formato antigo), o topo é uniforme. |

`height_m` é a altura **máxima** do vaso. A coluna i vai de z = 0 até h_i, e o bin j fica em z = h_i·j/(n_z−1).

## ECHO (msg_type 3, receptor → Python, opcional)
Cabeçalho comum (8 B) + seq u32, frame_id u32, t_capture_ns u64, t_send_ns u64, t_recv_ns u64, t_applied_ns u64. O Python grava esses campos em `latency.csv`.

## Convenções
- Metros e radianos. Eixo do vaso = **Z local** do objeto; base na origem.
- θ_i = 2π·i/N_θ, a partir de +X, anti-horário visto de +Z. z_j = H·j/(N_z−1).
- O mapa R fica no referencial do vaso. O receptor gira o objeto em Z por φ.
- Ordem dos vértices na malha: k = i·N_z + j (igual ao payload).

## Validação no receptor
Descartar pacotes com magic ou versão diferente, tamanho inconsistente ou seq mais antigo (com *wrap*). Drenar o socket a cada tick e aplicar só o pacote mais novo. Se n_theta ou n_z não baterem com a malha, avisar e ignorar.

## Latência
t_send − t_capture no Python; t_recv − t_send e t_applied − t_recv no receptor. É o mesmo relógio, porque roda na mesma máquina.
