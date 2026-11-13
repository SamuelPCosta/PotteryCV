# PotteryCV — Guia do Blender 5.2 (texto, sem código)

Você monta tudo à mão. **[VERIFICAR]** = conferir na documentação oficial do 5.2 (Manual e Python API). **Sem modificadores**: a forma e a espessura saem do Geometry Nodes.

## 1. Receptor
- Arquivo: `blender/pottery_receiver.py` (escrito pelo Claude, não integrado). Abra no Text Editor e use *Run Script*. Rodar de novo liga ou desliga.
- O que ele faz:
  - Cria ou valida o objeto `PotteryVase` com uma malha de **duas paredes**:
    - grade externa N_θ × N_z, mais uma interna igual (ordem k = i·N_z + j em cada uma);
    - faces que fecham a costura θ;
    - faces da **borda** ligando o anel superior externo ao interno;
    - um vértice central em cada fundo (externo e interno), com triângulos em leque.
  - Atributos de ponto: `radius` e `zn` (atualizados a cada pacote, iguais nas duas paredes; zn = altura relativa da coluna × j/(N_z−1), então o topo pode variar por coluna), `theta` (estático) e `wall` (0 = externa, 1 = interna).
  - Guarda H numa propriedade do objeto, `pottery_height`, atualizada a cada pacote.
  - Abre o UDP não bloqueante na porta 9000 e registra um timer de ~60 Hz **[VERIFICAR bpy.app.timers]**, que drena o socket, escreve `radius` em lote **[VERIFICAR foreach_set + atualização da malha]**, aplica rotação Z = φ e envia ECHO.
- Requisitos: NumPy embutido no Blender **[VERIFICAR]**; liberar o Blender no firewall, se o Windows pedir.

## 2. Geometry Nodes (modificador único em `PotteryVase`)
Nomes de nós e sockets: **[VERIFICAR no 5.2]**.
1. **Named Attribute** `radius`, `theta`, `zn` e `wall` (Float).
2. **Altura**: entrada do grupo `Height`, ligada à propriedade `pottery_height` por um *driver* (botão direito no campo → Add Driver, ou copiar como driver) **[VERIFICAR]**. z = zn·Height.
3. **Espessura**: entradas `Thickness` (≈ 6 mm) e `BottomThickness` (≈ 8 mm).
   - r_eff = max(radius − wall·Thickness, 0)
   - z_eff = z + wall·BottomThickness·(1 − zn), para elevar o fundo interno. Simplificação aceitável: aplicar só no centro e no anel inferior.
4. **Posição**: Combine XYZ (r_eff·cos θ, r_eff·sin θ, z_eff) → **Set Position** (Position, não Offset).
5. **Set Shade Smooth** → **Group Output**.
6. Opcional: **Subdivide Mesh** ou **Subdivision Surface node** (nível 1) *dentro* do GN, se o FPS permitir.

Não use primitivas do GN para a malha: a ordem dos vértices tem que bater com o protocolo.

## 3. Material de barro (Principled BSDF, [VERIFICAR nomes dos sockets])
- **Úmido**: Base Color cinza-marrom (≈ 0,48 / 0,42 / 0,36), Roughness 0,35, Coat 0,3 (rugosidade 0,2).
- **Terracota**: (≈ 0,70 / 0,38 / 0,24), Roughness 0,75.
- **Bump** com coordenadas *Object*: Noise de escala alta (≈ 300) com intensidade 0,1, mais uma Wave em bandas ao longo de Z (anéis de torneamento), fraca. Os anéis tornam a rotação visível.

## 4. Cena
- **Cabeça do torno**: disco cinza logo abaixo de z = 0, filho do vaso.
- **Luz**:
  - *key*: Area, 45° à frente e à direita;
  - *fill*: fraca, do lado oposto;
  - *rim*: atrás, para destacar a silhueta.
- **Fundo** cinza escuro. **Câmera** frontal em −Y, olhando +Y, de 50 a 85 mm.
- **Viewport** em EEVEE, modo Rendered **[VERIFICAR o nome no 5.2]**.

## 5. Teste sem câmera
1. Rode `python tools/synthetic_sender.py`.
2. Rode o receptor: o console mostra pacotes por segundo.
3. O *Spreadsheet* mostra `radius` mudando.
4. Com o GN: a forma igual à do preview do `python main.py --viewer`, girando.
5. A altura varia no gerador (`--height-wave`, que imita o vaso subindo ao ser apertado): o vaso estica e encolhe no Blender.
6. Rodar o receptor de novo não pode dar "porta em uso".

## 6. Problemas comuns
| Sintoma | Causa provável |
|---|---|
| Nada atualiza | timer não registrado ou malha não marcada como atualizada |
| Costura aberta ou torcida | ordem de vértices errada |
| Altura não muda | driver de `Height` não ligado |
| "Porta em uso" | receptor anterior não foi parado |
