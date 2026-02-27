import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pygame

# Logger do modulo.
logger = logging.getLogger(__name__)


class SVGRenderer:
    """Renderizador simples de SVG com cache para composicao em frames OpenCV."""

    def __init__(self) -> None:
        self._source_cache: Dict[str, Optional[pygame.Surface]] = {}
        self._scaled_rgba_cache: Dict[Tuple[str, int, int], Optional[np.ndarray]] = {}

    def _normalize_path(self, asset_path: str) -> str:
        # Padroniza caminho para uso no cache.
        return str(Path(asset_path))

    def _load_source_surface(self, asset_path: str) -> Optional[pygame.Surface]:
        # Carrega o SVG bruto uma unica vez por caminho.
        key = self._normalize_path(asset_path)
        if key in self._source_cache:
            return self._source_cache[key]

        try:
            surface = pygame.image.load(key)
            self._source_cache[key] = surface
            return surface
        except Exception as exc:
            logger.warning(f"Failed to load SVG asset '{asset_path}': {exc}")
            self._source_cache[key] = None
            return None

    def get_asset_size(self, asset_path: str) -> Tuple[int, int]:
        """Retorna tamanho nativo do asset."""
        surface = self._load_source_surface(asset_path)
        if surface is None:
            return 1, 1
        return surface.get_width(), surface.get_height()

    def _get_scaled_rgba(self, asset_path: str, width: int, height: int) -> Optional[np.ndarray]:
        # Retorna imagem RGBA redimensionada e cacheada.
        width = max(1, int(width))
        height = max(1, int(height))
        key = (self._normalize_path(asset_path), width, height)
        if key in self._scaled_rgba_cache:
            return self._scaled_rgba_cache[key]

        source = self._load_source_surface(asset_path)
        if source is None:
            self._scaled_rgba_cache[key] = None
            return None

        try:
            scaled = pygame.transform.smoothscale(source, (width, height))
            rgba_bytes = pygame.image.tostring(scaled, "RGBA")
            rgba = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((height, width, 4)).copy()
            self._scaled_rgba_cache[key] = rgba
            return rgba
        except Exception as exc:
            logger.warning(f"Failed to scale SVG asset '{asset_path}' to {width}x{height}: {exc}")
            self._scaled_rgba_cache[key] = None
            return None

    def draw_asset(
        self,
        frame: np.ndarray,
        asset_path: str,
        center: Tuple[int, int],
        size: Tuple[int, int],
        alpha_scale: float = 1.0,
    ) -> None:
        """Desenha o asset no frame BGR usando alpha compositing."""
        frame_h, frame_w = frame.shape[:2]
        width, height = max(1, int(size[0])), max(1, int(size[1]))
        rgba = self._get_scaled_rgba(asset_path, width, height)
        if rgba is None:
            return

        cx, cy = int(center[0]), int(center[1])
        x1 = cx - width // 2
        y1 = cy - height // 2
        x2 = x1 + width
        y2 = y1 + height

        # Clipa area alvo para manter composicao dentro do frame.
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(frame_w, x2)
        dst_y2 = min(frame_h, y2)
        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            return

        src_x1 = dst_x1 - x1
        src_y1 = dst_y1 - y1
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        src = rgba[src_y1:src_y2, src_x1:src_x2]
        alpha = (src[..., 3:4].astype(np.float32) / 255.0) * float(alpha_scale)
        if np.max(alpha) <= 0.0:
            return

        src_bgr = src[..., :3][:, :, ::-1].astype(np.float32)
        dst = frame[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        blended = src_bgr * alpha + dst * (1.0 - alpha)
        frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
