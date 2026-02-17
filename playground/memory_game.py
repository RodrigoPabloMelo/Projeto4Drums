import random
import math
from typing import Dict, List, Optional, Tuple


class MemoryGameController:
    """Controls Simon-style sequence playback and input validation."""

    ROUND_READY = "ROUND_READY"
    SHOW_SEQUENCE = "SHOW_SEQUENCE"
    WAIT_INPUT = "WAIT_INPUT"
    GAME_OVER = "GAME_OVER"

    def __init__(self, drum_count: int, game_config: Dict):
        self.drum_count = drum_count
        self.game_config = game_config
        self.highlight_s = game_config["highlight_ms"] / 1000.0
        self.between_s = game_config["between_steps_ms"] / 1000.0
        self.round_ready_s = game_config["round_ready_countdown_ms"] / 1000.0
        self.input_timeout_s = game_config["input_timeout_ms"] / 1000.0
        self.feedback_s = game_config["feedback_ms"] / 1000.0
        self.base_points = game_config["base_points"]
        self.combo_step = game_config["combo_step"]
        self.max_combo_multiplier = game_config["max_combo_multiplier"]
        self.round_bonus = game_config["round_bonus"]

        self.rng = random.Random()

        self.phase = self.SHOW_SEQUENCE
        self.sequence: List[int] = []
        self.round = 1
        self.score = 0
        self.combo_count = 0
        self.expected_index = 0
        self.phase_started_at = 0.0
        self.input_deadline = 0.0
        self.feedback: Optional[Tuple[int, str, float]] = None
        self.failed_drum_index: Optional[int] = None

    def start_new_run(self, now: float) -> None:
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
        self.start_new_run(now)

    def is_game_over(self) -> bool:
        return self.phase == self.GAME_OVER

    def get_combo_multiplier(self) -> float:
        return min(1.0 + self.combo_count * self.combo_step, self.max_combo_multiplier)

    def _advance_to_show_sequence(self, now: float) -> None:
        self.phase = self.SHOW_SEQUENCE
        self.phase_started_at = now
        self.expected_index = 0
        self.input_deadline = 0.0

    def _advance_to_round_ready(self, now: float) -> None:
        self.phase = self.ROUND_READY
        self.phase_started_at = now
        self.expected_index = 0
        self.input_deadline = 0.0

    def _advance_to_wait_input(self, now: float) -> None:
        self.phase = self.WAIT_INPUT
        self.expected_index = 0
        self.input_deadline = now + self.input_timeout_s

    def _set_feedback(self, drum_index: int, status: str, now: float) -> None:
        self.feedback = (drum_index, status, now + self.feedback_s)

    def _set_game_over(self, now: float, failed_drum_index: Optional[int]) -> None:
        self.phase = self.GAME_OVER
        self.phase_started_at = now
        self.failed_drum_index = failed_drum_index
        if failed_drum_index is not None:
            self._set_feedback(failed_drum_index, "wrong", now)

    def _update_phase(self, now: float) -> bool:
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

        if self.feedback and now > self.feedback[2]:
            self.feedback = None
        return failed_this_frame

    def _process_inputs(self, events: List[Tuple[int, str]], now: float) -> Tuple[List[int], bool, bool]:
        play_indices: List[int] = []
        sequence_completed_this_frame = False
        failed_this_frame = False
        if self.phase != self.WAIT_INPUT:
            return play_indices, sequence_completed_this_frame, failed_this_frame

        for drum_index, _source in events:
            expected = self.sequence[self.expected_index]
            if drum_index == expected:
                self.combo_count += 1
                points = round(self.base_points * self.get_combo_multiplier())
                self.score += points
                self._set_feedback(drum_index, "correct", now)
                play_indices.append(drum_index)
                self.expected_index += 1
                self.input_deadline = now + self.input_timeout_s

                if self.expected_index >= len(self.sequence):
                    self.score += self.round_bonus
                    self.round += 1
                    self.sequence.append(self.rng.randrange(self.drum_count))
                    self._advance_to_round_ready(now + self.between_s)
                    sequence_completed_this_frame = True
                    break
            else:
                self._set_game_over(now, drum_index)
                failed_this_frame = True
                break

        return play_indices, sequence_completed_this_frame, failed_this_frame

    def _build_indicator_states(self, now: float) -> Dict[int, str]:
        states = {idx: "idle" for idx in range(self.drum_count)}

        if self.phase == self.SHOW_SEQUENCE and self.sequence:
            cycle = self.highlight_s + self.between_s
            elapsed = now - self.phase_started_at
            if elapsed >= 0:
                step_index = int(elapsed // cycle)
                step_offset = elapsed % cycle
                if step_index < len(self.sequence) and step_offset <= self.highlight_s:
                    states[self.sequence[step_index]] = "active"

        if self.phase == self.GAME_OVER and self.failed_drum_index is not None:
            states[self.failed_drum_index] = "wrong"

        if self.feedback:
            drum_index, status, _feedback_until = self.feedback
            states[drum_index] = status

        return states

    def _countdown_value(self, now: float) -> Optional[int]:
        if self.phase != self.ROUND_READY:
            return None
        remaining = self.round_ready_s - (now - self.phase_started_at)
        if remaining <= 0:
            return None
        return max(1, int(math.ceil(remaining)))

    def update(self, now: float, events: List[Tuple[int, str]]) -> Dict:
        """Update state machine and return render/audio instructions."""
        timed_out_fail = self._update_phase(now)
        play_indices, sequence_completed, input_fail = self._process_inputs(events, now)
        failed_this_frame = timed_out_fail or input_fail

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
