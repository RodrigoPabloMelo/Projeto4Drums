import asyncio
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import CONFIG
from playground.drum import Drum
from playground.svg_renderer import SVGRenderer


class DrumKit:
    """Gerencia a colecao de tambores e seu render top-view."""

    def __init__(self, frame_dim: Tuple[int, int]):
        self.frame_dim = frame_dim
        self.renderer = SVGRenderer()
        self.drums: List[Drum] = []
        self.indicator_colors = CONFIG.get("memory_game", {}).get("indicator_colors")
        self._build_drums()

    def _build_drums(self) -> None:
        # Cria tambores com base no schema responsivo da configuracao.
        sound_bindings = CONFIG.get("sound_bindings", {})
        for drum_config in CONFIG["drums"]:
            sound_binding = drum_config["sound_binding"]
            sound_path = sound_bindings.get(sound_binding)
            if not sound_path:
                raise ValueError(f"Missing sound path for binding '{sound_binding}' in config['sound_bindings']")

            asset_path = drum_config["asset"]
            native_size = self.renderer.get_asset_size(asset_path)
            drum = Drum(
                drum_id=drum_config["id"],
                asset_path=asset_path,
                position_pct=drum_config["position_pct"],
                size_pct=drum_config["size_pct"],
                radius_pct=drum_config["radius_pct"],
                sound_binding=sound_binding,
                sound_path=sound_path,
                cooldown=CONFIG["drum_cooldown"],
                asset_native_size=native_size,
            )
            drum.update_geometry(self.frame_dim)
            self.drums.append(drum)

    def update_frame_dim(self, frame_dim: Tuple[int, int]) -> None:
        # Atualiza geometria de todos os tambores para o tamanho atual do frame.
        if self.frame_dim == frame_dim:
            return
        self.frame_dim = frame_dim
        for drum in self.drums:
            drum.update_geometry(frame_dim)

    def draw(self, frame: np.ndarray, indicator_states: Optional[Dict[int, str]] = None) -> None:
        """Desenha todos os tambores no frame."""
        self.update_frame_dim((frame.shape[1], frame.shape[0]))
        current_time = asyncio.get_event_loop().time()
        for idx, drum in enumerate(self.drums):
            state = indicator_states.get(idx) if indicator_states else None
            drum.draw(frame, self.renderer, current_time, state, self.indicator_colors)

    def interact(self, hand_pos: Tuple[int, int], hand_vel: float) -> None:
        """Verifica interacoes com todos os tambores."""
        current_time = asyncio.get_event_loop().time()
        for drum in self.drums:
            drum.check_hit(hand_pos, hand_vel, current_time, CONFIG["hit_velocity_threshold"])

    def get_drum_count(self) -> int:
        return len(self.drums)

    def get_drum_index_at_position(self, hand_pos: Tuple[int, int]) -> Optional[int]:
        """Retorna o indice do tambor que contem o ponto, se existir."""
        for idx, drum in enumerate(self.drums):
            if drum.collides_with(hand_pos):
                return idx
        return None

    def get_hit_drum_index_by_strike(
        self,
        hand_pos: Tuple[int, int],
        hand_vel: float,
        current_time: float,
    ) -> Optional[int]:
        """Retorna indice do tambor para strike sem tocar audio."""
        if hand_vel <= CONFIG["hit_velocity_threshold"]:
            return None

        for idx, drum in enumerate(self.drums):
            if drum.collides_with(hand_pos) and drum.can_trigger(current_time):
                return idx
        return None

    def play_drum_by_index(self, index: int, current_time: float) -> bool:
        """Toca tambor pelo indice, respeitando cooldown."""
        if index < 0 or index >= len(self.drums):
            return False
        return self.drums[index].try_play(current_time)

    def get_layout_data(self) -> List[Dict[str, object]]:
        """Exporta metadados de layout dos tambores."""
        return [drum.to_layout_data() for drum in self.drums]
