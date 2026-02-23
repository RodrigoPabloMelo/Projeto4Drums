from typing import Dict, List, Optional, Tuple
import numpy as np
import asyncio

from playground.drum import Drum
from config.config import CONFIG

class DrumKit:
    """Gerencia uma colecao de tambores."""
    def __init__(self, frame_dim: Tuple[int, int]):
        # Lista de objetos Drum em uso.
        self.drums: List[Drum] = []
        # Cores de indicador usadas no modo memory.
        self.indicator_colors = CONFIG.get("memory_game", {}).get("indicator_colors")
        # Converte posicoes relativas para pixels.
        w, h = frame_dim
        for drum_config in CONFIG['drums']:
            pos = (int(w * drum_config['pos'][0]), int(h * drum_config['pos'][1]))
            # Cria cada tambor com sua configuracao e cooldown.
            self.drums.append(Drum(
                drum_config['name'], pos, drum_config['radius'], 
                drum_config['sound'], CONFIG['drum_cooldown']
            ))

    def draw(self, frame: np.ndarray, indicator_states: Optional[Dict[int, str]] = None) -> None:
        """Desenha todos os tambores no frame."""
        # Usa tempo atual para indicar cooldown visual.
        current_time = asyncio.get_event_loop().time()
        for idx, drum in enumerate(self.drums):
            # Estado do indicador por tambor, se houver.
            state = indicator_states.get(idx) if indicator_states else None
            drum.draw(frame, current_time, state, self.indicator_colors)

    def interact(self, hand_pos: Tuple[int, int], hand_vel: float) -> None:
        """Verifica interacoes com todos os tambores."""
        # Tempo atual para respeitar cooldown.
        current_time = asyncio.get_event_loop().time()
        for drum in self.drums:
            # Executa logica de hit no tambor.
            drum.check_hit(hand_pos, hand_vel, current_time, CONFIG['hit_velocity_threshold'])

    def get_drum_count(self) -> int:
        # Quantidade total de tambores.
        return len(self.drums)

    def get_drum_index_at_position(self, hand_pos: Tuple[int, int]) -> Optional[int]:
        """Retorna o indice do tambor que contem o ponto, se existir."""
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
        """Retorna indice do tambor para strike sem tocar audio."""
        # Exige velocidade minima para considerar strike.
        if hand_vel <= CONFIG["hit_velocity_threshold"]:
            return None

        for idx, drum in enumerate(self.drums):
            # Garante que o ponto esta no tambor e respeita cooldown.
            if drum.contains_point(hand_pos) and drum.can_trigger(current_time):
                return idx
        return None

    def play_drum_by_index(self, index: int, current_time: float) -> bool:
        """Toca tambor pelo indice, respeitando cooldown."""
        # Protege contra indices invalidos.
        if index < 0 or index >= len(self.drums):
            return False
        # Tenta tocar o tambor e retorna sucesso.
        return self.drums[index].try_play(current_time)
