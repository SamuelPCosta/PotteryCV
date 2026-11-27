# PotteryCV — torno de cerâmica virtual

A webcam lê as mãos e a cabeça (MediaPipe), as mãos modelam um vaso que gira e a inclinação da cabeça controla a velocidade do torno. O vaso vai por UDP para o Blender, que monta a malha e renderiza. Specs em [specs/](specs/).

<!-- RELATORIO: substituir as secoes abaixo pelo texto final -->

## Introdução

_(texto do relatório)_

## Desenvolvimento

_(técnicas usadas: estimativa de pose, segmentação, profundidade)_

## Resultados

_(métricas e análise crítica)_

## Conclusão

_(texto do relatório)_

## Referências

_(referências)_

<!-- FIM DO RELATORIO -->

---

## Como rodar

### Instalação (Windows)

**Jeito rápido (portátil):** dois cliques em `rodar.bat`. Na primeira vez (com internet) ele baixa um Python portátil para `python\` (via [uv](https://docs.astral.sh/uv/), sem admin), instala as bibliotecas nele e baixa os modelos do MediaPipe. Depois a pasta inteira (~1,2 GB) roda em qualquer Windows 64 bits **sem instalar nada e sem internet**: copie a pasta toda (pendrive, Drive) e dê dois cliques no `rodar.bat`. Se o `requirements.txt` mudar, ele reinstala sozinho (aí precisa de internet).

Ou na mão:

Python 3.12 ou mais novo. No terminal (`cmd`), dentro da pasta do projeto:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- `python -m venv .venv` cria um Python só do projeto (uma vez só).
- `.venv\Scripts\activate` liga esse Python: aparece `(.venv)` no começo da linha. **Rode toda vez que abrir um terminal novo**, senão cai no Python do sistema, sem as bibliotecas.
- No PowerShell, se o `activate` reclamar de "execução de scripts desabilitada": use o `cmd` ou rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Modelos do MediaPipe** (a pasta `models/` não vai para o repositório): baixe para `models/`
- [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

**Opcional, extratores SAM (só GPU NVIDIA):** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` e `pip install ultralytics`. Pesos `mobile_sam.pt` / `sam3.pt` em `models/`. Sem NVIDIA, use só o extrator `landmarks` (o padrão), que roda no processador.

### Rodar

Com o `.venv` ativado:

```
python main.py                       # webcam + interface
python main.py --viewer              # só o preview, via UDP
python tools\synthetic_sender.py     # vaso falso -> Blender (:9000) e viewer (:9002)
python -m pytest -q                  # testes
```

Se abrir e fechar na hora, rode pelo terminal para ver o erro. Câmera não abre: troque `backend` (`msmf`, `dshow`, `any`) ou `index` no `config.toml`.

### Uso

1. **Calibrar** [C]. Cada etapa mostra a instrução e espera **Continuar** [Espaço]; depois uma máscara aparece em volta do rosto/mãos, onde estiverem, e enche enquanto você fica parado:
   - 1/2: cabeça relaxada olhando para a tela;
   - 2/2: mãos de lado, dedos para cima, na **largura do vaso**. Define a escala, o eixo (meio das mãos) e o raio inicial. A base fica a 90% da altura da imagem.

   O perfil fica salvo em `calibration.json`.
2. **Modelar**:
   - Encoste a mão na silhueta laranja, por fora: os dedos deixam marca, o vaso afina e sobe (slider "subida ao apertar dos lados").
   - Empurre o topo de cima para baixo: o vaso achata e engorda.
   - Esmagou abaixo de 20% da altura inicial: reinicia após 3 s.
   - **Desfazer** [Ctrl+Z]: volta até 10 toques (o reset também).
   - **Debug** [D]: landmarks numerados e medidor da inclinação da cabeça.
3. **Velocidade pela cabeça**: um pouco para cima, freia; para baixo, acelera; neutra, mantém. Invertido: `pitch_sign = -1` no `config.toml`. Máximo em `[speed] max_rpm`.
4. **Gravar sessão**: salva em `data/sessions/<data>/` o vídeo e os CSVs de quadros, mãos e latência. Revise com `main.py --replay <pasta>`.

O mapa desenrolado (embaixo da câmera) mostra o raio do vaso em cada ângulo (horizontal) e altura (vertical); as linhas Left Hand / Right Hand marcam onde cada mão está tocando.

### Blender

Abra `blender/Pottery.blend`, no Text Editor abra `blender/pottery_receiver.py` e rode uma vez. Isso liga o receptor e cria o painel **Pottery** na barra lateral do 3D Viewport (tecla N), com o botão Ligar/Desligar e a vista da câmera a cada N estados. Detalhes em [specs/blender-guide.md](specs/blender-guide.md).

- **Vista do Blender** (caixinha na interface): mostra a imagem da câmera do Blender no painel da direita. Mais rápido com o viewport em Solid ou Material Preview.
- **Umidade**: o objeto recebe `wetness` (0..1), que sobe ao tocar e seca com o tempo (`wet_rise_s`, `wet_dry_s` em `[vase]`). No material, use o nó Attribute (Type = Object, Name = `wetness`).

### Avaliação

`tools\annotate.py <sessão>` para anotar máscaras; depois `tools\evaluate.py {fps|contour|jitter|depth|latency} <sessões>`.
