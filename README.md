# Projeto4Drums

Aplicativo de bateria virtual com câmera usando MediaPipe + OpenCV + Pygame.

## Modos

- `Playground`: modo livre para tocar qualquer sequência sem score.
- `Memory Game`: modo de memória em que o app mostra a sequência e o jogador precisa repetir para pontuar.

## Controles

- `M`: alterna entre Playground e Memory Game em tempo real.
- `Q`: encerra a aplicação.
- `R`: reinicia a partida no Memory Game quando estiver em `GAME OVER`.

## Controles por gesto

- `2 dedos para cima` (vitória): muda para `Playground` (fora dos círculos).
- `1 dedo para cima` (apontando): muda para `Memory Game` (fora dos círculos).
- `2 punhos fechados` (duas mãos): reinicia o `Memory Game` quando estiver em `GAME OVER`.
- Importante: os gestos de comando precisam ser segurados por `3 segundos` para serem confirmados.

## Regras rápidas

- Ao entrar no `Memory Game`, o jogo sempre começa do zero (score e rodada resetados).
- No `Playground`, os sons podem ser acionados por strike (velocidade) e por gestos (pinch/fist) dentro das áreas.
- No `Memory Game`, entradas duplicadas no mesmo círculo em até `250ms` ativam uma janela de segurança de `1s`, onde entradas erradas não causam falha.
- Em cada rodada do `Memory Game`, há contagem regressiva de `3 segundos` antes de mostrar a sequência.
- O `Memory Game` toca som de erro ao falhar e som de sucesso ao completar uma sequência.
- O score aparece no topo central e o high score é mantido durante a sessão atual.
