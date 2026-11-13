# PotteryCV — torno de cerâmica virtual

Specs em [specs/](specs/). Ambiente: Windows, Python 3.15, venv em `.venv`.

```powershell
.\.venv\Scripts\python.exe main.py                      # webcam + interface
.\.venv\Scripts\python.exe main.py --viewer             # só o preview, via UDP
.\.venv\Scripts\python.exe tools\synthetic_sender.py    # vaso falso → Blender (:9000) e viewer (:9002)
.\.venv\Scripts\python.exe -m pytest -q                 # testes
```

**Uso**
1. **Calibrar** (botão). Cada etapa mostra "capturando em 3… 2… 1…":
   - 1/2: cabeça relaxada olhando para a tela;
   - 2/2: mãos de lado, dedos para cima, na **largura do vaso**. Isso define a escala, o eixo (meio das mãos) e o raio inicial. A base é sempre o rodapé da imagem.

   O perfil fica salvo em `calibration.json`.
2. **Modelar**:
   - Encoste a mão na silhueta laranja, por fora: os dedos deixam marca, o vaso afina e sobe (intensidade no slider "subida ao apertar dos lados").
   - Empurre o topo de cima para baixo: o vaso achata e engorda.
   - Se esmagar abaixo de 20% da altura inicial, o vaso reinicia após uma contagem de 3 s.
   - Há um único modo: a mão só empurra, por fora ou pelo topo.
   - **Debug** [D]: landmarks numerados e medidor da inclinação da cabeça.
3. **Velocidade pela cabeça**: um pouco para cima, para; para baixo, vai acelerando; neutra, mantém. Olhadas curtas são ignoradas. Se estiver invertido, use `pitch_sign = -1` no `config.toml`.
4. **Gravar sessão**: salva em `data/sessions/<data>/` o vídeo e os CSVs de quadros, mãos e latência. A sessão pode ser revista com `main.py --replay <pasta>`.

**Avaliação**: `tools\annotate.py <sessão>` para anotar máscaras; depois `tools\evaluate.py {fps|contour|jitter|depth|latency} <sessões>`.

**Blender**: `blender/pottery_receiver.py` (rode no Text Editor do Blender) + [specs/blender-guide.md](specs/blender-guide.md).
