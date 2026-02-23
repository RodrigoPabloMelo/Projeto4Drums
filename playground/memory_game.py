import random
import math
from typing import Dict, List, Optional, Tuple


class MemoryGameController:
    """Controla a sequencia tipo Simon e valida entradas."""

    # Estados do jogo.
    ROUND_READY = "ROUND_READY"
    SHOW_SEQUENCE = "SHOW_SEQUENCE"
    WAIT_INPUT = "WAIT_INPUT"
    GAME_OVER = "GAME_OVER"

    def __init__(self, drum_count: int, game_config: Dict):
        # Configuracoes basicas.
        self.drum_count = drum_count
        self.game_config = game_config
        # Converte tempos de ms para segundos.
        self.highlight_s = game_config["highlight_ms"] / 1000.0
        self.between_s = game_config["between_steps_ms"] / 1000.0
        self.round_ready_s = game_config["round_ready_countdown_ms"] / 1000.0
        self.input_timeout_s = game_config["input_timeout_ms"] / 1000.0
        self.feedback_s = game_config["feedback_ms"] / 1000.0
        # Pontuacao e combo.
        self.base_points = game_config["base_points"]
        self.combo_step = game_config["combo_step"]
        self.max_combo_multiplier = game_config["max_combo_multiplier"]
        self.round_bonus = game_config["round_bonus"]

        # RNG dedicado para sorteios da sequencia.
        self.rng = random.Random()

        # Estado inicial do jogo.
        self.phase = self.SHOW_SEQUENCE
        self.sequence: List[int] = []
        self.round = 1
        self.score = 0
        self.combo_count = 0
        self.expected_index = 0
        self.phase_started_at = 0.0
        self.input_deadline = 0.0
        # Feedback visual temporario (indice, status, expira_em).
        self.feedback: Optional[Tuple[int, str, float]] = None
        self.failed_drum_index: Optional[int] = None

    def start_new_run(self, now: float) -> None:
        # Inicia nova partida com uma sequencia inicial.
        self.sequence = [self.rng.randrange(self.drum_count)]
        self.phase = self.ROUND_READY
        self.round = 1
        self.score = 0
        self.combo_count = 0
        self.expected_index = 0
        self.phase_started_at = now
        self.input_deadline = 0.0
        self.feedback = None
        self.failed_drum_index = None

    def restart(self, now: float) -> None:
        # Reinicia a partida reutilizando a logica inicial.
        self.start_new_run(now)

    def is_game_over(self) -> bool:
        # Indica se o jogo terminou.
        return self.phase == self.GAME_OVER

    def get_combo_multiplier(self) -> float:
        # Calcula multiplicador de combo limitado ao maximo.
        return min(1.0 + self.combo_count * self.combo_step, self.max_combo_multiplier)

    def _advance_to_show_sequence(self, now: float) -> None:
        # Passa para o estado de mostrar a sequencia.
        self.phase = self.SHOW_SEQUENCE
        self.phase_started_at = now
        self.expected_index = 0
        self.input_deadline = 0.0

    def _advance_to_round_ready(self, now: float) -> None:
        # Prepara a contagem antes de mostrar a nova sequencia.
        self.phase = self.ROUND_READY
        self.phase_started_at = now
        self.expected_index = 0
        self.input_deadline = 0.0

    def _advance_to_wait_input(self, now: float) -> None:
        # Passa para o estado de espera da entrada do jogador.
        self.phase = self.WAIT_INPUT
        self.expected_index = 0
        self.input_deadline = now + self.input_timeout_s

    def _set_feedback(self, drum_index: int, status: str, now: float) -> None:
        # Define feedback visual temporario para um tambor.
        self.feedback = (drum_index, status, now + self.feedback_s)

    def _set_game_over(self, now: float, failed_drum_index: Optional[int]) -> None:
        # Finaliza o jogo e registra o tambor que falhou.
        self.phase = self.GAME_OVER
        self.phase_started_at = now
        self.failed_drum_index = failed_drum_index
        if failed_drum_index is not None:
            self._set_feedback(failed_drum_index, "wrong", now)

    def _update_phase(self, now: float) -> bool:
        # Atualiza a maquina de estados baseada no tempo.
        failed_this_frame = False
        if self.phase == self.ROUND_READY:
            if now - self.phase_started_at >= self.round_ready_s:
                self._advance_to_show_sequence(now)
        elif self.phase == self.SHOW_SEQUENCE:
            cycle = self.highlight_s + self.between_s
            if now - self.phase_started_at >= len(self.sequence) * cycle:
                self._advance_to_wait_input(now)
        elif self.phase == self.WAIT_INPUT and now > self.input_deadline:
            self._set_game_over(now, None)
            failed_this_frame = True

        # Limpa feedback vencido.
        if self.feedback and now > self.feedback[2]:
            self.feedback = None
        return failed_this_frame

    def _process_inputs(self, events: List[Tuple[int, str]], now: float) -> Tuple[List[int], bool, bool]:
        # Processa entradas do jogador e atualiza pontuacao.
        play_indices: List[int] = []
        sequence_completed_this_frame = False
        failed_this_frame = False
        if self.phase != self.WAIT_INPUT:
            return play_indices, sequence_completed_this_frame, failed_this_frame

        for drum_index, _source in events:
            # Comparacao com o item esperado da sequencia.
            expected = self.sequence[self.expected_index]
            if drum_index == expected:
                # Acerto: atualiza combo e pontuacao.
                self.combo_count += 1
                points = round(self.base_points * self.get_combo_multiplier())
                self.score += points
                self._set_feedback(drum_index, "correct", now)
                play_indices.append(drum_index)
                self.expected_index += 1
                self.input_deadline = now + self.input_timeout_s

                if self.expected_index >= len(self.sequence):
                    # Sequencia completada: inicia nova rodada.
                    self.score += self.round_bonus
                    self.round += 1
                    self.sequence.append(self.rng.randrange(self.drum_count))
                    self._advance_to_round_ready(now + self.between_s)
                    sequence_completed_this_frame = True
                    break
            else:
                # Erro: finaliza o jogo imediatamente.
                self._set_game_over(now, drum_index)
                failed_this_frame = True
                break

        return play_indices, sequence_completed_this_frame, failed_this_frame

    def _build_indicator_states(self, now: float) -> Dict[int, str]:
        # Inicia todos os tambores em estado idle.
        states = {idx: "idle" for idx in range(self.drum_count)}

        if self.phase == self.SHOW_SEQUENCE and self.sequence:
            # Destaca o passo atual da sequencia.
            cycle = self.highlight_s + self.between_s
            elapsed = now - self.phase_started_at
            if elapsed >= 0:
                step_index = int(elapsed // cycle)
                step_offset = elapsed % cycle
                if step_index < len(self.sequence) and step_offset <= self.highlight_s:
                    states[self.sequence[step_index]] = "active"

        if self.phase == self.GAME_OVER and self.failed_drum_index is not None:
            # Mostra o tambor que causou falha.
            states[self.failed_drum_index] = "wrong"

        if self.feedback:
            # Aplica feedback temporario (correto/errado).
            drum_index, status, _feedback_until = self.feedback
            states[drum_index] = status

        return states

    def _countdown_value(self, now: float) -> Optional[int]:
        # Retorna o valor da contagem regressiva antes da rodada.
        if self.phase != self.ROUND_READY:
            return None
        remaining = self.round_ready_s - (now - self.phase_started_at)
        if remaining <= 0:
            return None
        return max(1, int(math.ceil(remaining)))

    def update(self, now: float, events: List[Tuple[int, str]]) -> Dict:
        """Atualiza o estado do jogo e retorna instrucoes de render/audio."""
        # Avanca fase por tempo.
        timed_out_fail = self._update_phase(now)
        # Processa entradas do jogador.
        play_indices, sequence_completed, input_fail = self._process_inputs(events, now)
        failed_this_frame = timed_out_fail or input_fail

        # Retorna dados para HUD, sons e indicadores.
        return {
            "indicator_states": self._build_indicator_states(now),
            "play_indices": play_indices,
            "countdown_value": self._countdown_value(now),
            "failure_visual_active": self.phase == self.GAME_OVER,
            "sequence_completed_this_frame": sequence_completed,
            "failed_this_frame": failed_this_frame,
            "hud": {
                "score": self.score,
                "combo_multiplier": self.get_combo_multiplier(),
                "round": self.round,
                "state": self.phase,
                "message": "GAME OVER - press R to restart" if self.phase == self.GAME_OVER else "",
            },
        }
