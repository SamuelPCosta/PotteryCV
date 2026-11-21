# PotteryCV — Design (protótipo)

## 1. Arquitetura
```
Thread Qt (GUI)                         Thread do pipeline (QThread)
 MainWindow  ◄── sinal: FrameResult ───  captura → hands → face (1/2 q.) → bordas → [calibração]
  webcam+overlay | preview 3D            → extrator (landmarks | pele | SAM*) → perfil p(z)
  mapa R | painel debug                  → contato → vaso (varredura, nível 2, altura por volume)
  controles ── sinal: comandos ────────► → velocidade → overlay (OpenCV) → CSV → UDP ──► Blender :9000
* SAM roda num **processo** separado (fila de tamanho 1). Medido: CUDA do torch 2.14 em thread secundária
  no Python 3.15 dá access violation; MobileSAM ≈ 38–46 ms, SAM 2.1-t ≈ 56 ms, FastSAM ≈ 20 ms → sempre assíncrono.
tools/synthetic_sender.py ──► mesmo protocolo ──► Blender  e  main.py --viewer (:9002)
```
- A GUI nunca bloqueia o pipeline. Ela recebe o resultado mais recente do quadro (imagem com overlay, cópia de R, H, φ, métricas) por sinal Qt. Se a GUI atrasar, os quadros intermediários são descartados.
- Comandos da GUI (botões, checkboxes, slider) entram numa fila lida pelo pipeline no início de cada quadro.

## 2. Estrutura de arquivos
```
main.py                 # entrada: `python main.py [--profile p.json] [--replay sessao/] [--viewer]`
pottery/
  config.py             # dataclasses + TOML
  capture.py            # câmera (thread), gravação e replay
  tracking.py           # MediaPipe mãos e rosto, One Euro, atribuição de bordas
  calibration.py        # pixel → vaso, etapas, JSON
  contour.py            # extratores (landmarks, pele, SAM*), fecho → perfil, modo degradado
  vase.py               # mapa R, varredura, contato, nível 2, altura por volume
  speed.py              # cabeça → ω, manual
  net.py                # protocolo (codificar e decodificar), sender, receiver
  pipeline.py           # loop por quadro (QThread), overlay OpenCV, logs CSV
  gui.py                # MainWindow PySide6, preview 3D, mapa R, painel
tools/  synthetic_sender.py  annotate.py  evaluate.py
blender/pottery_receiver.py   # escrito, não integrado
config.toml   requirements.txt   tests/
```

## 3. Coordenadas
- Imagem espelhada. Mundo e vaso: X à direita da tela, Y para dentro da cena, Z para cima, em metros. Base em z = 0.
- O mapa R está no referencial do vaso. Um ponto local θ fica no ângulo de mundo θ + φ. Bordas de contato: direita θ_w = 0, esquerda θ_w = π.
- **Pixel → vaso**: a calibração dá o eixo u_a, a base v_b e L_ref. k = L_real/L_ref (L_real = 0,095 m na config). A escala da mão é s = L/L_ref, com L = max(|P0−P9|, κ·|P5−P17|) suavizado.
  `x = k·[(u−c_x)/s − (u_a−c_x)]`, `z = k·[(v_b−c_y) − (v−c_y)/s]` (com s = 1 se a normalização estiver desligada).
- **H variável**: z_j = H·j/(N_z−1). Quando H muda, as linhas do mapa esticam ou comprimem, sem reamostrar R. H vai em cada mensagem. A regra que define H está no §6.

## 4. Mão → perfil
1. O extrator devolve a máscara na região de interesse:
   - **landmarks**: fecho dos 21 pontos com discos de raio 0,09·L nas articulações;
   - **pele**: YCrCb com limiares amostrados na calibração, ∩ região de interesse;
   - **SAM**: MobileSAM, SAM 2.1-t, FastSAM e EfficientSAM (se instalar), com prompt de caixa + 6 pontos positivos dos landmarks.
2. Pós-processamento comum: maior componente da máscara → **contorno real (não convexo)** → para cada altura z_j, a linha da máscara dá [u_min, u_max] → coordenadas do vaso. Na borda direita r_near = x_min; na esquerda r_near = −x_max. Entre dedos separados a linha fica vazia, e o dedo deixa a sua marca.
3. Cobertura, peso (rampa cosseno de 3 mm nas pontas de cada dedo) e EMA temporal (β = 0,5).
4. **Topo (local)**: cada coluna θ tem a sua altura Hc[θ]. Para cada mão, calcula-se o ponto mais baixo da máscara na faixa |x| ≤ raio da boca. Uma mão que vem de cima e encosta no topo da coluna sob ela entra em PRESS. **Só as colunas varridas sob a mão** (mesma varredura de arco dos lados) abaixam: Hc ← Hc − α·top_rate·(Hc − z_low), com R da coluna ← R·√(Hc_antes/Hc_novo), o que conserva o volume local (achata e engorda ali). Com o torno girando, a borda inteira vai baixando aos poucos.
4. **Modo degradado**: o SAM roda assíncrono, recebendo um recorte com margem de 50% em volta das caixas (com o recorte justo, ele devolve o recorte inteiro) e, como prompt, a caixa mais o centro da palma. A máscara velha é levada ao quadro atual por `estimateAffinePartial2D` entre os landmarks antigos e os atuais. Se tiver idade > 6 quadros ou se o SAM der erro, usa landmarks.

## 5. Contato (só por fora)
Parede atual na borda e: W_j = R[i_e(φ), j]. Penetração: pen = max_j (W_j − r_near_j) sobre os bins cobertos.
- FREE → CONTACT quando pen ≥ −2 mm.
- CONTACT → FREE quando pen < −15 mm por 3 quadros, ou a mão some por > 200 ms.
- Em CONTACT, a mão só empurra para dentro, mesmo se cruzar a parede.

Há um modo só: a mão empurra por fora (ou pelo topo, §4). Não existe "agarrar e puxar".

## 6. Mapa R
**Varredura de arco** por borda, do quadro k−1 para k:
```
a = (θ_e − φ_{k−1})·N/2π ;  b = (θ_e − φ_k)·N/2π
|b−a| ≥ N  → todas as colunas
a == b     → coluna round(b)
senão      → inteiros n em (a, b] (semiaberto), i = n mod N, s = (n−a)/(b−a)
             perfil = (1−s)·p_{k−1} + s·p_k
```
O intervalo semiaberto garante que cada coluna é tocada uma vez por passagem, sem depender de ω nem do FPS.

**Aplicação (nível 2)**, por bin coberto j com peso w_j:
- PUSH: se R > r_near, então R ← R − α·w·(R − r_near)

Em seguida, suavização Gaussiana local (σ_θ = 1 coluna, σ_z = 1 bin, λ = 0,3) só nas colunas tocadas, e clamp em [R_min, R_max]. Com α = 1 e sem suavização tem-se o nível 1.

**Altura por conservação de volume (global)**. O barro é tratado como um sólido de revolução, V = π·H·média(R²).
A cada aplicação lateral: H ← H·(m_antes/m_depois)^side_gain, com m = média(R²).
- side_gain = 1 conserva o volume exatamente;
- side_gain = 0 deixa a altura fixa;
- o padrão é 0,4, ajustável no slider.

H ∈ [H_min, H_max]. Pressão no topo: ver §4.
- Empurrar para dentro reduz a média de R², então H sobe.
- Ao mudar H, as mãos passam a corresponder a outros bins z (z_j = H·j/(N_z−1)), o que é coerente com o barro subindo.
- Com o vaso subindo, as mãos naturalmente acompanham, como no "coning" real.
- O parâmetro `volume_gain` ∈ [0, 1] ajusta a força do efeito: 0 deixa H fixo, 1 é conservação exata.
- Não há escoamento local: a parede não "engorda" ao lado da mão. Fica como trabalho futuro.

## 7. Velocidade
- Pitch vem da matriz de transformação do Face Landmarker; o sinal é verificado na T1. Δ = pitch − pitch₀, filtrado com One Euro.
- Portão de validade: rosto ausente ou |yaw| > 25° mantém ω.
- Se a cabeça ficar **para baixo além de 8°** por ≥ T_dwell (0,3 s): ω += a·e·dt, com e = clip((Δ−8°)/(22°−8°))^1,5 (vai acelerando).
- Se ficar **para cima além de 4°**: freia rápido até parar.
- Neutra: mantém. ω ∈ [0, 90 rpm], sem inversão. Trava: tecla L. Sem calibração, ω = 0.

## 8. Estados e interface Qt
```
INIT → (perfil?) → MODELING ⇄ PAUSED
          └─não─► CALIBRATING → MODELING
```
**Layout da janela** (PySide6):
```
┌──────────────────────────────┬────────────────────────┐
│ webcam + overlay (QLabel)    │ preview 3D do vaso     │
│ silhueta, fecho, perfil,     │ (pyqtgraph GLMeshItem, │
│ instruções da calibração     │  orbitável)            │
│                              ├────────────────────────┤
│                              │ mapa R desenrolado     │
├──────────────────────────────┴────────────────────────┤
│ Debug: estado · FPS · ms por etapa · ω rpm · H · V/V₀ │
│        contato L/R · extrator · idade da máscara SAM  │
│ [Calibrar] [Cancelar] [Reset] [Gravar] ☐Debug extrator▼│
│ velocidade: (•)cabeça ( )slider ═══○═══  [Trava]      │
│ ☑ suavização  ☑ normalização de profundidade          │
└───────────────────────────────────────────────────────┘
```
- **Preview 3D**: a malha cilíndrica é montada a partir de R, H e φ (mesma topologia e convenções do Blender), atualizada a ≤ 30 Hz, com cor de barro e sombreamento simples.
- **Fallback**: se o OpenGL ou o pyqtgraph falharem, um render NumPy simples num QLabel.
- **Linhas de topologia** (overlay da câmera): 12 linhas verticais presas a θ fixos do barro, só na metade da frente, desenhadas pelo perfil de cada coluna e mais fortes no centro. Elas deixam o giro visível.
- **Mapa R**: QLabel com colormap (θ × z) e linhas das colunas de contato.
- **Atalhos**: Space pausa, R reset, C calibra, Q sai.
- `--viewer`: a mesma janela, só com o preview e o mapa, alimentados por UDP (`net.receiver`).

## 9. Desempenho
- Captura em thread (MJPG, buffer 1). MediaPipe na CPU; rosto a cada 2 quadros. Se faltar folga, as mãos são processadas a 640×360.
- SAM: recorte da região de interesse, fp16, as duas mãos no mesmo lote.
- O painel de depuração mostra ms por etapa. A GUI é atualizada a ≤ 30 Hz e não entra na conta do FPS do pipeline.

## 10. Decisões
| Decisão | Por quê / descartado |
|---|---|
| Mapa R(θ,z) cilíndrico | voxels e SDF são exagero para um protótipo |
| UDP, estado completo em uint16 (32 KB, 1 datagrama) | float32 não cabe (65.536 B); só colunas alteradas é frágil a perda; TCP traz atraso |
| Atributo `radius` na malha → Geometry Nodes | textura tem interpolação e problemas de espaço de cor |
| Espessura **dentro do Geometry Nodes** (malha com parede externa e interna, atributo `wall`) | o usuário não quer modificador Solidify |
| Lado decidido pela posição | o rótulo do MediaPipe inverte com o espelho |
| Velocidade incremental | o modo absoluto reage a toda olhada |
| GUI em PySide6 + pyqtgraph (os dois têm wheels para 3.15) | janelas OpenCV não têm controles nem 3D orbitável; Open3D e pyvista são incertos no 3.15 |
| Pipeline numa QThread separada da GUI | GUI no mesmo loop derrubaria o FPS |
| ~10 módulos planos em `pottery/` | subpastas são exagero para um protótipo |

## 11. Testes (pytest)
Varredura (CA-04, costura, ω = 0, |Δφ| ≥ 2π); contato (CA-05); volume constante e H sobe ao empurrar (CA-06); push nunca aumenta R; conversão pixel → vaso com projeção sintética; protocolo (CA-08, blocos, rejeição de pacotes inválidos); velocidade (CA-07). Integração: replay de uma sessão gravada.
