import logging
from typing import Dict, Tuple

import cv2
import numpy as np
import pygame

from config.config import CONFIG

logger = logging.getLogger(__name__)


class TextRenderer:
    """Renderiza texto com Instrument Sans via pygame.font."""

    def __init__(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

        fonts_cfg = CONFIG.get("fonts", {})
        self.font_paths = {
            "regular": fonts_cfg.get("instrument_sans_regular", ""),
            "bold": fonts_cfg.get("instrument_sans_bold", fonts_cfg.get("instrument_sans_regular", "")),
        }
        self._cache: Dict[Tuple[str, int], pygame.font.Font] = {}
        self._font_available = True

    def _get_font(self, size: int, weight: str) -> pygame.font.Font:
        size = max(8, int(size))
        key = (weight, size)
        if key in self._cache:
            return self._cache[key]

        path = self.font_paths["bold"] if weight == "bold" else self.font_paths["regular"]
        try:
            font = pygame.font.Font(path, size)
        except Exception as exc:
            if self._font_available:
                logger.warning(f"Could not load Instrument Sans font '{path}', fallback to default: {exc}")
            self._font_available = False
            font = pygame.font.SysFont(None, size)
        self._cache[key] = font
        return font

    def measure(self, text: str, size: int, weight: str = "regular") -> Tuple[int, int]:
        font = self._get_font(size, weight)
        return font.size(text)

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        size: int,
        color_bgr: Tuple[int, int, int],
        align: str = "left",
        weight: str = "regular",
        alpha: float = 1.0,
        fallback_scale: float = 0.8,
    ) -> None:
        # Tenta desenhar com pygame.font e cai para cv2.putText em caso de erro.
        alpha = float(np.clip(alpha, 0.0, 1.0))
        try:
            font = self._get_font(size, weight)
            rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
            text_surface = font.render(text, True, rgb)
            text_rgba = pygame.image.tostring(text_surface, "RGBA")
            w, h = text_surface.get_size()
            src = np.frombuffer(text_rgba, dtype=np.uint8).reshape((h, w, 4))
            x, y = int(pos[0]), int(pos[1])
            if align == "center":
                x -= w // 2
            elif align == "right":
                x -= w

            x1 = max(0, x)
            y1 = max(0, y - h)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y)
            if x1 >= x2 or y1 >= y2:
                return

            src_x1 = x1 - x
            src_y1 = y1 - (y - h)
            src_x2 = src_x1 + (x2 - x1)
            src_y2 = src_y1 + (y2 - y1)

            clipped = src[src_y1:src_y2, src_x1:src_x2]
            src_bgr = clipped[..., :3][:, :, ::-1].astype(np.float32)
            dst = frame[y1:y2, x1:x2].astype(np.float32)
            a = (clipped[..., 3:4].astype(np.float32) / 255.0) * alpha
            blended = src_bgr * a + dst * (1.0 - a)
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        except Exception as exc:
            if self._font_available:
                logger.warning(f"TextRenderer failure, fallback to cv2.putText: {exc}")
            self._font_available = False
            cv2.putText(
                frame,
                text,
                (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                fallback_scale,
                color_bgr,
                2,
                lineType=cv2.LINE_AA,
            )
