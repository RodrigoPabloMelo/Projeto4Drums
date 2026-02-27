import logging
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from pygame import mixer

from config.config import CONFIG
from playground.svg_renderer import SVGRenderer

# Logger do modulo.
logger = logging.getLogger(__name__)


class Drum:
    """Representa um elemento de bateria em layout responsivo."""

    def __init__(
        self,
        drum_id: str,
        asset_path: str,
        position_pct: Tuple[float, float],
        size_pct: float,
        radius_pct: float,
        sound_binding: str,
        sound_path: str,
        cooldown: float,
        asset_native_size: Tuple[int, int],
    ) -> None:
        # Identidade e metadados do tambor.
        self.id = drum_id
        self.asset_path = asset_path
        self.position_pct = position_pct
        self.size_pct = size_pct
        self.radius_pct = radius_pct
        self.sound_binding = sound_binding
        self.cooldown = cooldown

        # Geometria resolvida em pixels (recalculada ao mudar tamanho do frame).
        self.center: Tuple[int, int] = (0, 0)
        self.radius_px = 1
        self.render_size: Tuple[int, int] = (1, 1)
        self.last_frame_dim: Optional[Tuple[int, int]] = None

        # Ratio do asset para manter proporcao visual.
        native_w = max(1, int(asset_native_size[0]))
        native_h = max(1, int(asset_native_size[1]))
        self.asset_ratio = native_w / native_h

        # Estado de hit e animacao.
        self.last_hit = 0.0
        self.hit_feedback_s = 0.16

        try:
            self.sound = mixer.Sound(sound_path)
        except Exception as exc:
            logger.error(f"Failed to load sound {sound_path} for drum '{drum_id}': {exc}")
            raise

    def update_geometry(self, frame_dim: Tuple[int, int]) -> None:
        """Atualiza centro, raio e tamanho em pixels usando percentual."""
        if self.last_frame_dim == frame_dim:
            return

        frame_w, frame_h = frame_dim
        min_dim = max(1, min(frame_w, frame_h))
        cx = int(round(frame_w * self.position_pct[0]))
        cy = int(round(frame_h * self.position_pct[1]))
        radius_px = max(10, int(round(min_dim * self.radius_pct)))
        base_size = max(24, int(round(min_dim * self.size_pct)))

        if self.asset_ratio >= 1.0:
            render_w = base_size
            render_h = max(1, int(round(base_size / self.asset_ratio)))
        else:
            render_h = base_size
            render_w = max(1, int(round(base_size * self.asset_ratio)))

        self.center = (cx, cy)
        self.radius_px = radius_px
        self.render_size = (render_w, render_h)
        self.last_frame_dim = frame_dim

    def collides_with(self, hand_pos: Tuple[int, int]) -> bool:
        """Retorna True se o ponto estiver dentro do raio do tambor."""
        dx = float(hand_pos[0] - self.center[0])
        dy = float(hand_pos[1] - self.center[1])
        return math.hypot(dx, dy) < float(self.radius_px)

    def contains_point(self, hand_pos: Tuple[int, int]) -> bool:
        """Alias de compatibilidade para deteccao de colisao."""
        return self.collides_with(hand_pos)

    def can_trigger(self, current_time: float) -> bool:
        """Retorna True se o cooldown do tambor terminou."""
        return current_time - self.last_hit > self.cooldown

    def try_play(self, current_time: float) -> bool:
        """Toca o som se o cooldown permitir."""
        if not self.can_trigger(current_time):
            return False
        try:
            self.sound.play()
            self.last_hit = current_time
            return True
        except Exception as exc:
            logger.error(f"Error playing sound for drum '{self.id}': {exc}")
            return False

    def _draw_feedback_glow(
        self,
        frame: np.ndarray,
        glow_color: Tuple[int, int, int],
        glow_alpha: float,
        scale: float,
    ) -> None:
        # Desenha brilho circular suave ao redor do tambor.
        overlay = frame.copy()
        glow_radius = max(6, int(self.radius_px * 1.22 * scale))
        cv2.circle(overlay, self.center, glow_radius, glow_color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, glow_alpha, frame, 1.0 - glow_alpha, 0, frame)

    def draw(
        self,
        frame: np.ndarray,
        renderer: SVGRenderer,
        current_time: float,
        indicator_state: Optional[str] = None,
        indicator_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> None:
        """Desenha o tambor com feedback visual por estado/hit."""
        frame_dim = (frame.shape[1], frame.shape[0])
        self.update_geometry(frame_dim)

        hit_elapsed = current_time - self.last_hit
        hit_progress = 0.0
        if 0.0 <= hit_elapsed < self.hit_feedback_s:
            hit_progress = 1.0 - (hit_elapsed / self.hit_feedback_s)

        scale = 1.0 + (0.08 * hit_progress)

        glow_color = (100, 100, 100)
        glow_alpha = 0.08
        if indicator_state and indicator_colors:
            glow_color = indicator_colors.get(indicator_state, glow_color)
            glow_alpha = 0.18 if indicator_state != "idle" else 0.10
        elif hit_progress > 0.0:
            glow_color = (230, 230, 230)
            glow_alpha = 0.18

        # Sombra para destacar profundidade no layout top-view.
        shadow_offset = max(4, int(self.radius_px * 0.08))
        shadow_center = (self.center[0] + shadow_offset, self.center[1] + shadow_offset)
        cv2.circle(
            frame,
            shadow_center,
            max(8, int(self.radius_px * 1.06 * scale)),
            (30, 30, 30),
            -1,
            lineType=cv2.LINE_AA,
        )

        self._draw_feedback_glow(frame, glow_color, glow_alpha, scale)

        draw_w = max(1, int(round(self.render_size[0] * scale)))
        draw_h = max(1, int(round(self.render_size[1] * scale)))
        brand_color = tuple(CONFIG.get("ui", {}).get("brand_color_bgr", (0, 53, 168)))
        renderer.draw_asset(
            frame,
            self.asset_path,
            self.center,
            (draw_w, draw_h),
            tint_bgr=brand_color,
        )

    def check_hit(self, hand_pos: Tuple[int, int], hand_vel: float, current_time: float, velocity_threshold: float) -> bool:
        """Verifica se o tambor foi atingido pela posicao e velocidade."""
        if self.collides_with(hand_pos) and hand_vel > velocity_threshold:
            return self.try_play(current_time)
        return False

    def to_layout_data(self) -> Dict[str, object]:
        """Exporta metadados uteis para depuracao e colisao."""
        return {
            "id": self.id,
            "position": {"x": self.center[0], "y": self.center[1]},
            "radius": self.radius_px,
            "sound_binding": self.sound_binding,
        }
