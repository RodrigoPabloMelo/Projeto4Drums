import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from config.config import CONFIG


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life_s: float
    ttl_s: float
    size: float
    color_bgr: Tuple[int, int, int]


class ParticleSystem:
    """Sistema simples de particulas para feedback visual de hits."""

    def __init__(self) -> None:
        cfg = CONFIG.get("particles", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.burst_count = int(cfg.get("burst_count", 16))
        self.speed_min = float(cfg.get("speed_min", 130.0))
        self.speed_max = float(cfg.get("speed_max", 320.0))
        self.lifetime_s = float(cfg.get("lifetime_ms", 380)) / 1000.0
        self.size_min = float(cfg.get("size_min", 2.0))
        self.size_max = float(cfg.get("size_max", 5.2))
        self.gravity = float(cfg.get("gravity", 520.0))
        self.base_color = tuple(cfg.get("color_bgr", (0, 53, 168)))
        self.particles: List[Particle] = []
        self.rng = random.Random()

    def emit_burst(self, x: int, y: int, count: int = None) -> None:
        if not self.enabled:
            return
        count = self.burst_count if count is None else int(count)
        for _ in range(max(1, count)):
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            speed = self.rng.uniform(self.speed_min, self.speed_max)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - self.rng.uniform(30.0, 80.0)
            size = self.rng.uniform(self.size_min, self.size_max)
            jitter = self.rng.randint(-24, 24)
            color = (
                int(np.clip(self.base_color[0] + jitter, 255, 255)),
                int(np.clip(self.base_color[1] + jitter, 255, 255)),
                int(np.clip(self.base_color[2] + jitter, 255, 255)),
            )
            self.particles.append(
                Particle(
                    x=float(x),
                    y=float(y),
                    vx=vx,
                    vy=vy,
                    life_s=self.lifetime_s,
                    ttl_s=self.lifetime_s,
                    size=size,
                    color_bgr=color,
                )
            )

    def update(self, dt_s: float) -> None:
        if not self.enabled:
            return
        dt_s = max(0.0, float(dt_s))
        alive: List[Particle] = []
        for p in self.particles:
            p.life_s -= dt_s
            if p.life_s <= 0.0:
                continue
            p.vy += self.gravity * dt_s
            p.x += p.vx * dt_s
            p.y += p.vy * dt_s
            alive.append(p)
        self.particles = alive

    def draw(self, frame: np.ndarray) -> None:
        if not self.enabled or not self.particles:
            return
        overlay = frame.copy()
        h, w = frame.shape[:2]
        for p in self.particles:
            if p.x < -20 or p.y < -20 or p.x > w + 20 or p.y > h + 20:
                continue
            alpha = np.clip(p.life_s / p.ttl_s, 0.0, 1.0)
            radius = max(1, int(p.size * (0.6 + 0.5 * alpha)))
            cv2.circle(
                overlay,
                (int(p.x), int(p.y)),
                radius,
                p.color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
