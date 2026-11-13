# PotteryCV — Avaliação (protótipo)

Participante: só o autor (estudo de caso, N = 1); se outra pessoa usar, recalibrar. Tudo roda sobre **sessões gravadas**, para que todos os métodos vejam os mesmos quadros.

## Logs
- `frames.csv`: frame_id, t_capture_ns, t_send_ns, ms_hands, ms_face, ms_contour, ms_sim, ms_total, fps, extractor, seg_age, omega, phi, height, pitch_raw, pitch_filt, contact_L, contact_R, smoothing_on, depthnorm_on
- `hands.csv`: frame_id, edge, scale_px, contact_state, r_near (N_z valores separados por `;`)
- `latency.csv`: seq, t_capture_ns, t_send_ns, t_recv_ns, t_applied_ns (do ECHO)

## E1: Contorno da mão (landmarks × MobileSAM)
- **Dados**: ~40 quadros de 2 sessões gravadas (poses e distâncias variadas), anotados à mão por polígono, contornando os dedos (`tools/annotate.py`).
- **Métricas**: IoU do contorno real vs. anotação; Chamfer (px); ms por quadro (média e p95); FPS do pipeline.
- **Saída**: tabela + gráfico erro × ms.

## E2: Estabilidade
- **Procedimento**: mão parada 10 s × 3 posições. A mesma gravação é processada com e sem suavização (One Euro + EMA).
- **Métricas**: dp temporal de r_near(z), média sobre z, em mm.

## E3: Efeito da distância
- **Procedimento**: a mão desliza encostada numa guia (livro em pé) a X fixo, em 5 profundidades (−10 a +10 cm do plano de calibração).
- **Métricas**: erro do raio medido vs. X real e variação ao longo da profundidade, **com e sem** normalização.

## E4: Latência
- **Medidas**: t_send − t_capture (Python) e t_recv, t_applied via ECHO (receptor do Blender e visualizador).
- **Métricas**: mediana e p95 por etapa.

## E5 (opcional): Robustez da cabeça a olhadas
- **Tarefa**: a ~60 rpm, olhar 1 s para 4 cantos da tela; medir |Δω| máximo, com e sem T_dwell.
