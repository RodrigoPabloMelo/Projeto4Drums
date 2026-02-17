from typing import Dict, List, Optional, Tuple
import numpy as np
import asyncio

from playground.drum import Drum
from config.config import CONFIG

class DrumKit:
    """Manages a collection of drums."""
    def __init__(self, frame_dim: Tuple[int, int]):
        self.drums: List[Drum] = []
        self.indicator_colors = CONFIG.get("memory_game", {}).get("indicator_colors")
        w, h = frame_dim
        for drum_config in CONFIG['drums']:
            pos = (int(w * drum_config['pos'][0]), int(h * drum_config['pos'][1]))
            self.drums.append(Drum(
                drum_config['name'], pos, drum_config['radius'], 
                drum_config['sound'], CONFIG['drum_cooldown']
            ))

    def draw(self, frame: np.ndarray, indicator_states: Optional[Dict[int, str]] = None) -> None:
        """Draw all drums on the frame."""
        current_time = asyncio.get_event_loop().time()
        for idx, drum in enumerate(self.drums):
            state = indicator_states.get(idx) if indicator_states else None
            drum.draw(frame, current_time, state, self.indicator_colors)

    def interact(self, hand_pos: Tuple[int, int], hand_vel: float) -> None:
        """Check for interactions with all drums."""
        current_time = asyncio.get_event_loop().time()
        for drum in self.drums:
            drum.check_hit(hand_pos, hand_vel, current_time, CONFIG['hit_velocity_threshold'])

    def get_drum_count(self) -> int:
        return len(self.drums)

    def get_drum_index_at_position(self, hand_pos: Tuple[int, int]) -> Optional[int]:
        """Return drum index containing the point, if any."""
        for idx, drum in enumerate(self.drums):
            if drum.contains_point(hand_pos):
                return idx
        return None

    def get_hit_drum_index_by_strike(
        self,
        hand_pos: Tuple[int, int],
        hand_vel: float,
        current_time: float
    ) -> Optional[int]:
        """Return hit drum index for strike gesture without playing audio."""
        if hand_vel <= CONFIG["hit_velocity_threshold"]:
            return None

        for idx, drum in enumerate(self.drums):
            if drum.contains_point(hand_pos) and drum.can_trigger(current_time):
                return idx
        return None

    def play_drum_by_index(self, index: int, current_time: float) -> bool:
        """Play drum by index with cooldown enforcement."""
        if index < 0 or index >= len(self.drums):
            return False
        return self.drums[index].try_play(current_time)
