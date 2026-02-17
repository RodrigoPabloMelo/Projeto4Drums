import logging
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from pygame import mixer

logger = logging.getLogger(__name__)

class Drum:
    """Represents a single drum with position, sound, and hit detection."""
    def __init__(self, name: str, pos: Tuple[int, int], radius: int, sound_path: str, cooldown: float):
        self.name = name
        self.pos = pos
        self.radius = radius
        self.cooldown = cooldown
        self.last_hit = 0.0
        try:
            self.sound = mixer.Sound(sound_path)
        except Exception as e:
            logger.error(f"Failed to load sound {sound_path}: {e}")
            raise

    def contains_point(self, hand_pos: Tuple[int, int]) -> bool:
        """Return True if a point falls inside the drum area."""
        dist = np.linalg.norm(np.array(hand_pos) - np.array(self.pos))
        return dist < self.radius

    def can_trigger(self, current_time: float) -> bool:
        """Return True if drum cooldown has elapsed."""
        return current_time - self.last_hit > self.cooldown

    def try_play(self, current_time: float) -> bool:
        """Play sound if cooldown allows."""
        if not self.can_trigger(current_time):
            return False
        try:
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
        indicator_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> None:
        """Draw the drum with optional game indicator colors."""
        if indicator_state and indicator_colors:
            color = indicator_colors.get(indicator_state, indicator_colors.get("idle", (0, 0, 255)))
        else:
            color = (0, 255, 0) if current_time - self.last_hit < self.cooldown else (0, 0, 255)
        cv2.circle(frame, self.pos, self.radius, color, 4)
        cv2.putText(
            frame, self.name, (self.pos[0] - self.radius, self.pos[1] - self.radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    def check_hit(self, hand_pos: Tuple[int, int], hand_vel: float, current_time: float, velocity_threshold: float) -> bool:
        """Check if the drum is hit based on hand position and velocity."""
        if self.contains_point(hand_pos) and hand_vel > velocity_threshold:
            return self.try_play(current_time)
        return False
