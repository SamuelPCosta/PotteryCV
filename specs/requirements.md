# PotteryCV — Requisitos (protótipo)

Torno de cerâmica virtual: a webcam vê as duas mãos; o contorno de cada mão (fecho convexo) empurra a parede de um vaso que gira. O pitch da cabeça controla a velocidade. O Python simula o mapa de raios R(θ, z) e envia por UDP local ao Blender 5.2, que só exibe (Geometry Nodes, montado pelo usuário).

**Responsabilidades**: o Claude faz todo o Python e **escreve** o script receptor do Blender (`blender/pottery_receiver.py`), sem integrá-lo. O usuário monta a cena e o Geometry Nodes no Blender.

## Requisitos funcionais

**Captura**
- RF-01: Capturar a webcam a 720p e 30 FPS, espelhada.
- RF-02: Detectar até 2 mãos (MediaPipe Hands) e a pose da cabeça (MediaPipe Face Landmarker → pitch).
- RF-03: Atribuir a borda (esquerda ou direita) pela posição da mão em relação ao eixo, não pelo rótulo de lateralidade.

**Calibração** (só alinhamento)
- RF-04: A calibração só começa pelo **botão Calibrar**. Cada etapa mostra a instrução e uma contagem "capturando em 3… 2… 1…" e depois captura com ~1 s de pose parada. Duas etapas: (1) cabeça neutra; (2) mãos de lado, dedos para cima, na largura do vaso. A etapa 2 dá ao mesmo tempo a escala (pulso → nó do médio), o eixo (meio das bordas internas) e o **raio inicial** (metade da distância entre as bordas internas). **A base do vaso é sempre o rodapé da imagem** (não calibrada). Sem calibração, a velocidade fica em 0.
- RF-05: Salvar e carregar o perfil em JSON; recalibrar com uma tecla.
- RF-06: Raio inicial R₀ e altura inicial H₀ vêm da config, em metros. **Durante a modelagem, H varia com as mãos, como no torno real**:
  - apertar dos lados sobe o vaso, numa intensidade ajustada por um **slider** (0 = altura fixa, 100% = conserva volume);
  - **empurrar o topo para baixo achata o vaso, que engorda** conservando o volume;
  - a altura máxima é **80% da altura da imagem**;
  - se a altura ficar **abaixo de 20% da inicial**, aparece "reiniciando em 3… 2… 1…" e o vaso volta ao cilindro (dá tempo de tirar a mão);
  - o barro nunca "volta": a suavização só espalha a deformação nova (slider "espalhamento da marca").

**Ferramenta (mão)**
- RF-07: A ferramenta é o **contorno real da mão, com os dedos** (não convexo): cada dedo deixa sua marca. Para cada z, usa-se o pixel da mão mais próximo do eixo.
- RF-08: Dois extratores, trocáveis na interface: **landmarks** (padrão: palma + dedos como cápsulas ao longo dos ossos) e **MobileSAM** (prompt de caixa + centro da palma vindos dos landmarks).
- RF-09: Normalização por escala da mão (L_ref/L) e suavização temporal, ambas desligáveis para a avaliação.
- RF-10: Se a segmentação não aguentar o tempo real, entra o modo degradado: segmentar em taxa menor e propagar a máscara pelos landmarks.

**Vaso**
- RF-11: Mapa R de 128 × 128, iniciado como cilindro. φ ← φ + ω·Δt.
- RF-12: Aplicar o perfil em **todas** as colunas varridas entre o ângulo anterior e o atual (sem listras).
- RF-13: **Só por fora (modo B)**: mão em contato empurra a parede para dentro. O contato tem histerese (começa ao tocar e termina ao afastar além de uma margem).
- RF-15: Nível 2: plasticidade α, atenuação nas pontas da mão e suavização espacial; R ∈ [R_min, R_max]. Tecla de reset.

**Velocidade**
- RF-16: Δpitch em relação ao neutro → zona morta → suavização. **Cabeça mais de 4° para cima: o torno para. Mais de 8° para baixo: vai acelerando. Entre os dois: mantém.** Sem inversão de sentido.
- RF-17: Só reage a inclinações sustentadas por ≥ T_dwell, para ignorar olhadas rápidas. Rosto perdido mantém ω. Tecla de trava.

**Saídas**
- RF-19b: Caixinha **Debug**: mostra os landmarks numerados e um medidor visual da inclinação da cabeça. Ao lado do mapa R, uma seta circular gira na velocidade do torno. O vaso aparece preenchido a 30% sobre a imagem da câmera. Sem botão de pausa.
- RF-19: **Interface Qt básica** (PySide6) para ver e depurar, numa janela só:
  - webcam com overlay (silhueta do vaso, contornos e perfis das mãos);
  - **preview 3D do vaso** (orbitável com o mouse);
  - mapa R desenrolado;
  - painel de depuração (estado, FPS e ms por etapa, ω, H, volume, estado de contato de cada mão);
  - controles: calibrar, cancelar calibração, reset, gravar sessão, debug, extrator, trava da velocidade, sliders (subida ao apertar e espalhamento da marca) e liga/desliga da suavização e da normalização.
- RF-20: `python main.py --viewer` abre só o preview, recebendo pelo protocolo UDP. Serve de receptor de referência e para testar o gerador sintético.
- RF-24: O ponto de entrada é `main.py`, na raiz do repositório.
- RF-21: UDP local para o Blender, conforme [protocol.md](protocol.md), com timestamps para latência.
- RF-22: Gerador sintético (vaso animado falso) que fala o mesmo protocolo.
- RF-23: Logs CSV por quadro e scripts de avaliação ([evaluation.md](evaluation.md)).

## Não funcionais
- RNF-01: **≥ 20 FPS** médios com duas mãos e o extrator landmarks (medido em 60 s).
- RNF-02: Nenhum treino ou fine-tuning. Só modelos prontos e OpenCV.
- RNF-03: Windows 10, Python 3.15 + venv + pip, RTX 3060. Código em inglês, docs em português.
- RNF-04: O núcleo (mapa R, varredura, contato, protocolo) é testável com pytest, sem câmera.
- RNF-05: Modular, mas pequeno: um pacote `pottery/` com cerca de 9 módulos, sem subpastas.

## Critérios de aceitação
| ID | Dado / quando / então |
|---|---|
| CA-01 | Dado duas mãos e o extrator landmarks, quando roda 60 s, então FPS médio ≥ 20 (log). |
| CA-02 | Dado as mãos cruzadas, quando visíveis, então cada uma trabalha a borda do lado da tela onde está. |
| CA-03 | Dado a calibração, quando as mãos ficam paradas 1 s, então eixo e base são registrados; se se moverem antes, o tempo reinicia. |
| CA-04 | Dado Δφ = 40° por quadro, quando a mão fica em contato uma volta inteira, então as 128 colunas são modificadas (teste unitário, incluindo a costura 2π → 0). |
| CA-05 | Dado o contato iniciado por fora, quando a mão atravessa a parede sem se afastar, então R só diminui. |
| CA-06 | Dado o cilindro, quando as mãos apertam dos lados, então H sobe proporcionalmente ao slider (0 = fixo; 100% = volume ±1%). Quando uma mão empurra o topo para baixo, então H desce e R cresce, com volume ±1%. |
| CA-07 | Dado Δpitch na zona morta, ou uma olhada de < T_dwell, então ω não muda. Para baixo sustentado, ω cresce; um pouco para cima, ω vai a 0. |
| CA-08 | Dado um mapa de 128 × 128, quando codificado, então cabe em 1 datagrama ≤ 65.507 B, e a decodificação tem erro ≤ passo de quantização. |
| CA-09 | Dado o gerador sintético, quando `main.py --viewer` escuta, então mostra o vaso girando (com 20% de perda simulada continua coerente). |
| CA-11 | Dado `python main.py`, quando a câmera abre, então a janela Qt mostra webcam + overlay, preview 3D atualizando e painel de depuração, sem derrubar o FPS abaixo de 20. |
| CA-10 | Dado um extrator SAM, quando ativo, então o pipeline mantém ≥ 20 FPS (modo degradado se for preciso), e uma falha do SAM faz voltar para landmarks. |

## Fora de escopo
Conservação de volume local, com o barro escoando para os vizinhos (só existe a global, aplicada pela altura); empurrar por dentro; exportar malha; foto e vaso-alvo; múltiplos usuários; câmera de profundidade; rede entre máquinas; qualquer automação no Blender.

## Riscos principais
| Risco | Mitigação |
|---|---|
| SAM lento | recorte da região de interesse, fp16, variantes leves, modo degradado |
| Olhar para a tela muda o pitch | modo incremental com T_dwell, zona morta, trava, slider |
| Distância da mão à câmera muda o raio | normalização por escala da mão (medida em E3) |
| Jitter do perfil | One Euro, EMA do perfil, α < 1 |
| Lateralidade invertida pelo espelho | atribuir a borda por posição |
| Wheels do 3.15 (torch com CUDA, EfficientSAM) | verificar na T0; plano B: venv 3.11 |
| API do Blender 5.2 | pontos marcados como "verificar na documentação" |
