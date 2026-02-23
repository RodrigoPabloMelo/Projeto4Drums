import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from pygame import mixer

# Logger do modulo.
logger = logging.getLogger(__name__)


class Drum:
    """Representa um tambor com posicao, som e deteccao de impacto."""

    def __init__(self, name: str, pos: Tuple[int, int], radius: int, sound_path: str, cooldown: float):
        # Metadados basicos do tambor.
        self.name = name
        self.pos = pos
        self.radius = radius
        self.cooldown = cooldown
        # Momento do ultimo toque para controlar cooldown.
        self.last_hit = 0.0
        try:
            # Carrega o som do tambor.
            self.sound = mixer.Sound(sound_path)
        except Exception as e:
            logger.error(f"Failed to load sound {sound_path}: {e}")
            raise

    def contains_point(self, hand_pos: Tuple[int, int]) -> bool:
        """Retorna True se o ponto estiver dentro do raio do tambor."""
        # Calcula distancia entre o ponto e o centro do tambor.
        dist = np.linalg.norm(np.array(hand_pos) - np.array(self.pos))
        return dist < self.radius

    def can_trigger(self, current_time: float) -> bool:
        """Retorna True se o cooldown do tambor terminou."""
        return current_time - self.last_hit > self.cooldown

    def try_play(self, current_time: float) -> bool:
        """Toca o som se o cooldown permitir."""
        if not self.can_trigger(current_time):
            return False
        try:
            # Toca o som e registra o horario do hit.
            self.sound.play()
            self.last_hit = current_time
            return True
        except Exception as e:
            logger.error(f"Error playing sound for {self.name}: {e}")
            return False

    def draw(
        self,
        frame: np.ndarray,
        current_time: float,
        indicator_state: Optional[str] = None,
        indicator_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> None:
        """Desenha o tambor com cores opcionais do jogo."""
        # Usa cores do modo memory quando fornecidas.
        if indicator_state and indicator_colors:
            color = indicator_colors.get(indicator_state, indicator_colors.get("idle", (0, 0, 255)))
        else:
            # Alterna cor baseada no cooldown do toque.
            color = (0, 255, 0) if current_time - self.last_hit < self.cooldown else (0, 0, 255)
        # Desenha o circulo do tambor.
        cv2.circle(frame, self.pos, self.radius, color, 4)
        # Escreve o nome do tambor acima do circulo.
        cv2.putText(
            frame,
            self.name,
            (self.pos[0] - self.radius, self.pos[1] - self.radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def check_hit(self, hand_pos: Tuple[int, int], hand_vel: float, current_time: float, velocity_threshold: float) -> bool:
        """Verifica se o tambor foi atingido pela posicao e velocidade."""
        # Confere se a mao esta sobre o tambor e se a velocidade passou o limiar.
        if self.contains_point(hand_pos) and hand_vel > velocity_threshold:
            return self.try_play(current_time)
        return False

