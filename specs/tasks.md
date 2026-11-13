# PotteryCV — Tarefas (protótipo)

Cada marco pode ser demonstrado sozinho. Arquivos conforme o design §2. "Pronto quando" = critério de pronto.

| ID | Marco / tarefa | Dep. | Pronto quando | Req. |
|---|---|---|---|---|
| **T0** | **Ambiente**: venv 3.15, `requirements.txt` (numpy, opencv-python, mediapipe, PySide6, pyqtgraph, PyOpenGL, pytest), `config.toml`, `config.py`; checar torch com CUDA e ultralytics no 3.15 | — | imports OK; nota do que falhou | RNF-03 |
| **M1** | **Webcam + MediaPipe + interface Qt** | | | |
| T1.1 | `capture.py` (thread, 720p a 30 FPS, espelhada, timestamp, gravar e reproduzir) | T0 | 30 FPS de captura | RF-01 |
| T1.2 | `tracking.py`: mãos + rosto (pitch, yaw), One Euro; checar a API do MediaPipe 1.x e o sinal do pitch | T1.1 | landmarks e pitch disponíveis | RF-02 |
| T1.3 | `pipeline.py` (QThread, overlay OpenCV, ms por etapa) + `gui.py` (janela com webcam e painel de depuração) + `main.py` na raiz | T1.2 | **CA-01**; janela abre com a webcam e o painel | RF-19, RF-24, RNF-01 |
| **M2** | **Calibração** | | | |
| T2.1 | `calibration.py`: pixel → vaso + normalização; etapas cabeça → escala → eixo e base; JSON; botão Calibrar | T1.3 | **CA-03**; cilindro alinhado às mãos | RF-04, RF-05 |
| T2.2 | Atribuição de bordas por posição | T2.1 | **CA-02** | RF-03 |
| **M3** | **Landmarks + fecho + mapa R girando + preview (nível 1)** | | | |
| T3.1 | `contour.py`: extrator landmarks + fecho → perfil r_near(z) | T2.1 | perfil no overlay | RF-07 |
| T3.2 | `vase.py`: mapa R, varredura de arco, PUSH com α = 1, clamp, reset | T0 | **CA-04** (pytest) | RF-11, RF-12 |
| T3.3 | Ligar ao pipeline com o slider de ω; silhueta no overlay | T3.1, T3.2 | mãos cortam o vaso girando, sem listras | RF-19 |
| T3.4 | `gui.py`: preview 3D (pyqtgraph) + mapa R desenrolado, com fallback NumPy | T3.3 | preview acompanha a modelagem; **CA-11** | RF-19 |
| **M4** | **Nível 2 + altura** | | | |
| T4.1 | Contato com histerese (só por fora), α, atenuação, suavização Gaussiana, EMA do perfil | T3.3 | **CA-05**; modelagem suave | RF-13, RF-15 |
| T4.3 | Altura por conservação de volume (V₀ no reset, H_alvo, η, limites, `volume_gain`) | T4.1 | **CA-06** (pytest: volume ±1%); demo: empurrar sobe o vaso | RF-06 |
| **M5** | **Protocolo + sintético + receptor** (você monta o Blender em paralelo) | | | |
| T5.1 | `net.py`: protocolo (STATE, ECHO) + sender com múltiplos destinos + receiver | T3.2 | **CA-08** (pytest) | RF-21 |
| T5.2 | `main.py --viewer` (preview alimentado por UDP) | T5.1, T3.4 | em sincronia com o app | RF-20 |
| T5.3 | `tools/synthetic_sender.py` (padrões, `--drop`, `--height-wave`) | T5.1 | **CA-09** | RF-22 |
| T5.4 | Escrever `blender/pottery_receiver.py` (não integrado; pontos marcados com `# VERIFICAR`) | T5.1 | arquivo entregue; decodificação testada fora do Blender | RF-21 |
| T5.5 | **Você**: montar o Blender seguindo o [blender-guide.md](blender-guide.md) | T5.3, T5.4 | roteiro §5 OK; **CA-06** no Blender | — |
| **M6** | **Cabeça → velocidade** | | | |
| T6.1 | `speed.py`: zona morta, incremental com T_dwell, portão de validade, trava; seletor cabeça ou slider na GUI | T1.2 | **CA-07** | RF-16..18 |
| **M7** | **Pele + SAM + comparação** | | | |
| T7.1 | Extrator de pele (limiares da calibração) | T3.1 | máscara visível | RF-08 |
| T7.2 | SAM via ultralytics (MobileSAM, SAM 2.1-t, FastSAM) + EfficientSAM se instalar; YOLO-pose como região de interesse (opcional) | T0, T3.1 | ms medido por variante | RF-08 |
| T7.3 | Processo de GPU + propagação por landmarks + fallback; seletor de extrator na GUI | T7.2 | **CA-10** | RF-10 |
| **M8** | **Avaliação** | | | |
| T8.1 | CSV no pipeline + `tools/annotate.py` + `tools/evaluate.py` (E1–E4) | M7 | tabelas e figuras geradas por um comando | RF-23 |
