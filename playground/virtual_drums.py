import asyncio
import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer

from config.config import CONFIG
from playground.drum_kit import DrumKit
from playground.memory_game import MemoryGameController
from playground.particles import ParticleSystem
from playground.text_renderer import TextRenderer

# Logger do modulo.
logger = logging.getLogger(__name__)

# Constantes de modo e comandos.
MODE_PLAYGROUND = "playground"
MODE_MEMORY = "memory"
CMD_SWITCH_PLAYGROUND = "switch_playground"
CMD_SWITCH_MEMORY = "switch_memory"
CMD_RESTART_MEMORY = "restart_memory"
SCENE_INTRO = "intro"
SCENE_TRANSITION = "transition"
SCENE_GAME = "game"


class VirtualDrums:
    """Classe principal da aplicacao de bateria virtual."""
    def __init__(self):
        # Componentes do MediaPipe.
        self.mp_hands = mp.solutions.hands
        self.hands = None
        # Camera e kit de tambores.
        self.cap = None
        self.kit = None
        # Controlador do modo memory.
        self.memory_game: Optional[MemoryGameController] = None
        # Sons opcionais de feedback.
        self.fail_sound = None
        self.success_sound = None
        # Estado atual.
        self.current_mode = MODE_PLAYGROUND
        # Teclas configuradas.
        controls = CONFIG.get("app_mode", {}).get("controls", {})
        self.switch_mode_key = controls.get("switch_mode_key", "m").lower()
        self.restart_key = controls.get("restart_key", "r").lower()
        self.quit_key = controls.get("quit_key", "q").lower()
        # Memoria de posicoes anteriores para calcular velocidade.
        self.prev_positions: Dict[int, Tuple[int, int, float]] = {}
        # Controle de gestos ativos para transicoes.
        self.prev_gesture_active: Dict[Tuple[int, str], bool] = {}
        # Debounce de eventos por gesto.
        self.last_gesture_event: Dict[Tuple[int, int, str], float] = {}
        # Toasts de modo.
        self.mode_toast: Optional[Tuple[str, float]] = None
        # Debounce para comandos por gesto.
        self.last_command_time: Dict[str, float] = {}
        # Estado do "segurar gesto" para comandos.
        self.command_hold_candidate: Optional[str] = None
        self.command_hold_started_at: float = 0.0
        # Guardas do modo memory para entradas duplicadas.
        self.last_memory_input_index: Optional[int] = None
        self.last_memory_input_time: float = 0.0
        self.memory_safe_until: float = 0.0
        # Estatisticas da sessao.
        self.session_high_score: int = 0
        # Efeito visual de falha.
        self.fail_visual_until: float = 0.0
        self.fail_visual_message = "ERROU! GAME OVER"
        # Renderizadores auxiliares.
        self.text_renderer = TextRenderer()
        self.particles = ParticleSystem()
        # Estado da cena (intro -> transicao -> jogo).
        self.scene_state = SCENE_INTRO
        self.intro_started_at: float = 0.0
        self.intro_trigger_started_at: float = 0.0
        self.transition_started_at: float = 0.0
        # Tempo entre frames para simulacao das particulas.
        self.last_frame_time: float = 0.0

    def _reset_memory_safe_guard(self) -> None:
        # Reseta protecoes contra entradas duplicadas no modo memory.
        self.last_memory_input_index = None
        self.last_memory_input_time = 0.0
        self.memory_safe_until = 0.0
        self.fail_visual_until = 0.0

    def _resolve_initial_mode(self) -> str:
        # Resolve o modo inicial com compatibilidade retroativa.
        app_mode = CONFIG.get("app_mode", {})
        default_mode = app_mode.get("default_mode", MODE_PLAYGROUND)
        if default_mode in (MODE_PLAYGROUND, MODE_MEMORY):
            return default_mode

        # Fallback para configuracao antiga.
        legacy_enabled = CONFIG.get("game_mode_enabled")
        if legacy_enabled is True:
            return MODE_MEMORY
        return MODE_PLAYGROUND

    def _switch_mode(self, now: float) -> None:
        # Alterna entre playground e memory.
        if self.current_mode == MODE_PLAYGROUND:
            self._switch_to_mode(MODE_MEMORY, now, force_new_run=True)
        else:
            self._switch_to_mode(MODE_PLAYGROUND, now, force_new_run=False)

    def _switch_to_mode(self, mode: str, now: float, force_new_run: bool) -> None:
        # Aplica mudanca de modo e atualiza estado do jogo.
        if mode == MODE_MEMORY:
            self.current_mode = MODE_MEMORY
            if not self.memory_game:
                self.memory_game = MemoryGameController(
                    self.kit.get_drum_count(),
                    CONFIG["memory_game"]
                )
            if force_new_run:
                self.memory_game.start_new_run(now)
                self._reset_memory_safe_guard()
            self.mode_toast = ("Switched to Memory", now + 1.0)
            return

        self.current_mode = MODE_PLAYGROUND
        self._reset_memory_safe_guard()
        self.mode_toast = ("Switched to Playground", now + 1.0)

    def _key_matches(self, key_code: int, key_name: str) -> bool:
        # Compara tecla pressionada com a configurada.
        return key_code == ord(key_name.lower())

    def _load_optional_sound(self, path: str):
        # Carrega um som opcional sem quebrar a execucao em caso de falha.
        try:
            return mixer.Sound(path)
        except Exception as e:
            logger.warning(f"Could not load optional sound '{path}': {e}")
            return None

    def setup(self) -> None:
        """Inicializa pygame, MediaPipe e camera."""
        try:
            # Inicializa o mixer de audio.
            mixer.init()
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            raise

        try:
            # Inicializa o MediaPipe Hands.
            self.hands = self.mp_hands.Hands(**CONFIG['hands_config'])
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise

        try:
            # Abre a camera e valida leitura inicial.
            self.cap = cv2.VideoCapture(CONFIG['camera_index'])
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read from camera.")
            # Cria o kit de tambores com dimensoes do frame.
            h, w = frame.shape[:2]
            self.kit = DrumKit((w, h))
            # Carrega sons de feedback do modo memory.
            self.fail_sound = self._load_optional_sound(CONFIG["memory_game"].get("fail_sound", "sounds/fail_1.wav"))
            success_path = CONFIG["memory_game"].get("success_sound", "sounds/success_1.wav")
            self.success_sound = self._load_optional_sound(success_path)
            if self.success_sound is None:
                self.success_sound = self._load_optional_sound("sounds/crash_1.wav")
            # Define modo inicial e cria controlador se necessario.
            self.current_mode = self._resolve_initial_mode()
            if self.current_mode == MODE_MEMORY:
                self.memory_game = MemoryGameController(
                    self.kit.get_drum_count(),
                    CONFIG["memory_game"]
                )
                now = asyncio.get_event_loop().time()
                self.memory_game.start_new_run(now)
                self.intro_started_at = now
                self.last_frame_time = now
            else:
                now = asyncio.get_event_loop().time()
                self.intro_started_at = now
                self.last_frame_time = now
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise

    def _is_pinched(self, hand_landmarks, frame_w: int, frame_h: int) -> bool:
        # Detecta gesto de pinça usando distancia entre polegar e indicador.
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        pinch_dist = math.hypot(
            (thumb_tip.x - index_tip.x) * frame_w,
            (thumb_tip.y - index_tip.y) * frame_h
        )
        return pinch_dist < CONFIG["memory_game"]["gesture_pinched_threshold"]

    def _is_fist(self, hand_landmarks) -> bool:
        # Detecta punho fechado comparando dobra dos dedos.
        tips = (8, 12, 16, 20)
        knuckles = (5, 9, 13, 17)
        wrist = hand_landmarks.landmark[0]
        palm_ref = hand_landmarks.landmark[9]
        palm_scale = max(math.hypot(wrist.x - palm_ref.x, wrist.y - palm_ref.y), 1e-4)

        fold = 0.0
        for tip_idx, knuckle_idx in zip(tips, knuckles):
            tip = hand_landmarks.landmark[tip_idx]
            knuckle = hand_landmarks.landmark[knuckle_idx]
            fold += math.hypot(tip.x - knuckle.x, tip.y - knuckle.y) / palm_scale
        avg_fold = fold / len(tips)
        return avg_fold < CONFIG["memory_game"]["fist_fold_threshold"]

    def _finger_fold_ratios(self, hand_landmarks) -> Dict[str, float]:
        # Calcula relacao de dobra de cada dedo.
        finger_points = {
            "index": (8, 5),
            "middle": (12, 9),
            "ring": (16, 13),
            "pinky": (20, 17),
        }
        wrist = hand_landmarks.landmark[0]
        palm_ref = hand_landmarks.landmark[9]
        palm_scale = max(math.hypot(wrist.x - palm_ref.x, wrist.y - palm_ref.y), 1e-4)

        ratios: Dict[str, float] = {}
        for name, (tip_idx, knuckle_idx) in finger_points.items():
            tip = hand_landmarks.landmark[tip_idx]
            knuckle = hand_landmarks.landmark[knuckle_idx]
            ratios[name] = math.hypot(tip.x - knuckle.x, tip.y - knuckle.y) / palm_scale
        return ratios

    def _is_pointing_up(self, hand_landmarks) -> bool:
        # Detecta gesto de "apontar para cima" (apenas indicador estendido).
        ratios = self._finger_fold_ratios(hand_landmarks)
        threshold = CONFIG["gesture_controls"]["pointing_up_fold_threshold"]
        extended_threshold = threshold + 0.2
        return (
            ratios["index"] > extended_threshold
            and ratios["middle"] < threshold
            and ratios["ring"] < threshold
            and ratios["pinky"] < threshold
        )

    def _is_victory(self, hand_landmarks) -> bool:
        # Detecta gesto de "vitoria" (indicador e medio estendidos).
        ratios = self._finger_fold_ratios(hand_landmarks)
        threshold = CONFIG["gesture_controls"]["victory_fold_threshold"]
        extended_threshold = threshold + 0.2
        return (
            ratios["index"] > extended_threshold
            and ratios["middle"] > extended_threshold
            and ratios["ring"] < threshold
            and ratios["pinky"] < threshold
        )

    def _should_accept_command(self, command: str, now: float) -> bool:
        # Debounce para evitar comandos repetidos em pouco tempo.
        debounce_s = CONFIG["gesture_controls"]["gesture_command_debounce_ms"] / 1000.0
        last = self.last_command_time.get(command, 0.0)
        if now - last < debounce_s:
            return False
        self.last_command_time[command] = now
        return True

    def _detect_command_candidate(
        self,
        hand_inputs: List[Dict]
    ) -> Optional[str]:
        # Detecta possivel comando baseado em gestos fora da area dos tambores.
        if not CONFIG.get("gesture_controls", {}).get("enabled", True):
            return None

        outside_inputs = [
            info for info in hand_inputs
            if self.kit.get_drum_index_at_position(info["index_pos"]) is None
        ]
        if not outside_inputs:
            return None

        fists_outside = [info for info in outside_inputs if self._is_fist(info["landmarks"])]
        if (
            len(fists_outside) >= 2
            and self.current_mode == MODE_MEMORY
            and self.memory_game
            and self.memory_game.is_game_over()
        ):
            return CMD_RESTART_MEMORY

        for info in outside_inputs:
            if self._is_victory(info["landmarks"]):
                return CMD_SWITCH_MEMORY
            if self._is_pointing_up(info["landmarks"]):
                return CMD_SWITCH_PLAYGROUND
        return None

    def _detect_command_gesture(
        self,
        hand_inputs: List[Dict],
        current_time: float
    ) -> Optional[str]:
        # Valida o comando apenas apos segurar o gesto por um tempo.
        candidate = self._detect_command_candidate(hand_inputs)
        if candidate is None:
            self.command_hold_candidate = None
            self.command_hold_started_at = 0.0
            return None

        if candidate != self.command_hold_candidate:
            self.command_hold_candidate = candidate
            self.command_hold_started_at = current_time
            return None

        hold_s = CONFIG["gesture_controls"]["command_hold_ms"] / 1000.0
        if current_time - self.command_hold_started_at < hold_s:
            return None

        if not self._should_accept_command(candidate, current_time):
            return None

        self.command_hold_candidate = None
        self.command_hold_started_at = 0.0
        return candidate

    def _detect_gesture_zone_event(
        self,
        hand_idx: int,
        hand_landmarks,
        index_pos: Tuple[int, int],
        frame_w: int,
        frame_h: int,
        current_time: float
    ) -> Optional[Tuple[int, str]]:
        # Detecta evento de gesto (pinch ou fist) sobre um tambor.
        pinched = self._is_pinched(hand_landmarks, frame_w, frame_h)
        fist = self._is_fist(hand_landmarks)

        active_kind: Optional[str] = None
        if pinched:
            active_kind = "pinch"
        elif fist:
            active_kind = "fist"

        was_active = False
        if active_kind:
            was_active = self.prev_gesture_active.get((hand_idx, active_kind), False)

        # Atualiza o estado de gesto ativo para cada mao.
        for gesture_kind in ("pinch", "fist"):
            self.prev_gesture_active[(hand_idx, gesture_kind)] = (gesture_kind == active_kind)

        if not active_kind:
            return None

        if was_active:
            return None

        # Verifica se o gesto ocorreu sobre algum tambor.
        drum_index = self.kit.get_drum_index_at_position(index_pos)
        if drum_index is None:
            return None

        # Debounce para gestos por tambor.
        debounce_s = CONFIG["memory_game"]["gesture_debounce_ms"] / 1000.0
        event_key = (hand_idx, drum_index, active_kind)
        last_event_time = self.last_gesture_event.get(event_key, 0.0)
        if current_time - last_event_time < debounce_s:
            return None

        self.last_gesture_event[event_key] = current_time
        return drum_index, active_kind

    def _smoothstep(self, t: float) -> float:
        # Easing suave para transicoes visuais.
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        size: int,
        color_bgr: Tuple[int, int, int],
        align: str = "left",
        weight: str = "regular",
        alpha: float = 1.0,
        fallback_scale: float = 0.8,
    ) -> None:
        self.text_renderer.draw_text(
            frame=frame,
            text=text,
            pos=(x, y),
            size=size,
            color_bgr=color_bgr,
            align=align,
            weight=weight,
            alpha=alpha,
            fallback_scale=fallback_scale,
        )

    def _draw_rounded_rect(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        radius: int,
        color_bgr: Tuple[int, int, int],
    ) -> None:
        # Retangulo arredondado para componentes de interface.
        radius = max(1, min(radius, w // 2, h // 2))
        cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x + radius, y + radius), radius, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x + w - radius, y + radius), radius, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x + radius, y + h - radius), radius, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x + w - radius, y + h - radius), radius, color_bgr, -1, lineType=cv2.LINE_AA)

    def _draw_quit_hint(self, frame: np.ndarray) -> None:
        # Mantem apenas a dica de saida no canto superior esquerdo.
        text = f"{self.quit_key.upper()} para sair"
        self._draw_text(
            frame,
            text,
            20,
            40,
            size=28,
            color_bgr=(255, 255, 255),
            align="left",
            weight="bold",
            fallback_scale=0.8,
        )

    def _draw_mode_tabs(self, frame: np.ndarray) -> None:
        # Barra de modo no topo central com segmento ativo.
        ui_cfg = CONFIG.get("ui", {})
        tab_cfg = ui_cfg.get("tab", {})
        assets_cfg = CONFIG.get("assets", {}).get("mode_icons", {})
        frame_h, frame_w = frame.shape[:2]

        container_w = int(frame_w * float(tab_cfg.get("container_width_pct", 0.60)))
        container_h = int(tab_cfg.get("container_height_px", 92))
        top_margin = int(tab_cfg.get("top_margin_px", 22))
        radius = int(tab_cfg.get("corner_radius_px", container_h // 2))
        padding = int(tab_cfg.get("inner_padding_px", 8))
        container_color = tuple(tab_cfg.get("container_bg_bgr", (245, 245, 245)))
        active_color = tuple(tab_cfg.get("active_bg_bgr", (199, 0, 83)))
        active_text_color = tuple(tab_cfg.get("text_active_bgr", (255, 255, 255)))
        inactive_text_color = tuple(tab_cfg.get("text_inactive_bgr", (199, 0, 83)))

        x = (frame_w - container_w) // 2
        y = top_margin
        self._draw_rounded_rect(frame, x, y, container_w, container_h, radius, container_color)

        segment_w = (container_w - (padding * 3)) // 2
        segment_h = container_h - (padding * 2)
        left_x = x + padding
        right_x = left_x + segment_w + padding
        seg_y = y + padding
        seg_radius = max(8, radius - padding)

        left_active = self.current_mode == MODE_PLAYGROUND
        if left_active:
            self._draw_rounded_rect(frame, left_x, seg_y, segment_w, segment_h, seg_radius, active_color)
        else:
            self._draw_rounded_rect(frame, right_x, seg_y, segment_w, segment_h, seg_radius, active_color)

        icon_size = int(segment_h * 0.42)
        label_size = int(segment_h * 0.52)
        label_y = seg_y + (segment_h // 2) + int(label_size * 0.15)

        play_icon = assets_cfg.get("playground", "assets/playground.svg")
        mem_icon = assets_cfg.get("memory", "assets/memory.svg")

        left_icon_center = (left_x + int(segment_w * 0.14), seg_y + segment_h // 2)
        right_icon_center = (right_x + int(segment_w * 0.14), seg_y + segment_h // 2)

        self.kit.renderer.draw_asset(
            frame,
            play_icon,
            left_icon_center,
            (icon_size, icon_size),
            tint_bgr=active_text_color if left_active else inactive_text_color,
        )
        self.kit.renderer.draw_asset(
            frame,
            mem_icon,
            right_icon_center,
            (icon_size, icon_size),
            tint_bgr=inactive_text_color if left_active else active_text_color,
        )

        self._draw_text(
            frame,
            "Playground",
            left_x + int(segment_w * 0.22),
            label_y,
            size=label_size,
            color_bgr=active_text_color if left_active else inactive_text_color,
            align="left",
            weight="bold",
            fallback_scale=1.0,
        )
        self._draw_text(
            frame,
            "Jogo da Memoria",
            right_x + int(segment_w * 0.22),
            label_y,
            size=label_size,
            color_bgr=inactive_text_color if left_active else active_text_color,
            align="left",
            weight="bold",
            fallback_scale=1.0,
        )

    def _is_hand_raised(self, hand_landmarks) -> bool:
        # Mao levantada quando indicador fica acima do punho.
        margin = float(CONFIG.get("ui", {}).get("hand_raise_margin", 0.02))
        return hand_landmarks.landmark[8].y < (hand_landmarks.landmark[0].y - margin)

    def _update_intro_state(self, hand_inputs: List[Dict], now: float) -> None:
        # Avanca estados intro/transicao com base em duas maos levantadas.
        ui_cfg = CONFIG.get("ui", {})
        hold_s = float(ui_cfg.get("intro_hold_ms", 450)) / 1000.0
        transition_s = float(ui_cfg.get("intro_transition_ms", 650)) / 1000.0

        if self.scene_state == SCENE_INTRO:
            raised_count = sum(1 for info in hand_inputs if self._is_hand_raised(info["landmarks"]))
            if raised_count >= 2:
                if self.intro_trigger_started_at <= 0.0:
                    self.intro_trigger_started_at = now
                elif now - self.intro_trigger_started_at >= hold_s:
                    self.scene_state = SCENE_TRANSITION
                    self.transition_started_at = now
            else:
                self.intro_trigger_started_at = 0.0

        if self.scene_state == SCENE_TRANSITION:
            progress = (now - self.transition_started_at) / max(transition_s, 1e-4)
            if progress >= 1.0:
                self.scene_state = SCENE_GAME

    def _draw_intro_overlay(self, frame: np.ndarray, now: float) -> np.ndarray:
        # Desenha overlay de intro no centro com logo e instrucoes.
        ui_cfg = CONFIG.get("ui", {})
        intro_cfg = ui_cfg.get("intro", {})
        base_alpha = float(intro_cfg.get("panel_opacity", 0.75))
        text_gap = int(intro_cfg.get("text_gap_px", 16))
        transition_s = float(ui_cfg.get("intro_transition_ms", 650)) / 1000.0

        alpha = base_alpha
        if self.scene_state == SCENE_TRANSITION:
            progress = self._smoothstep((now - self.transition_started_at) / max(transition_s, 1e-4))
            alpha = base_alpha * (1.0 - progress)

        if alpha <= 0.0:
            return frame

        blended = frame.copy()
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (255, 255, 255), -1)
        cv2.addWeighted(panel, alpha, blended, 1.0 - alpha, 0, blended)

        h, w = frame.shape[:2]
        logo_h = int(min(w, h) * 0.20)
        logo_w = int(logo_h * 2.14)
        logo_center = (w // 2, int(h * 0.49))
        self.kit.renderer.draw_asset(
            blended,
            "assets/logo.svg",
            logo_center,
            (logo_w, logo_h),
            alpha_scale=max(0.0, min(1.0, 1.0 - ((1.0 - alpha) * 0.3))),
        )
        logo_bottom = logo_center[1] + (logo_h // 2)
        intro_label = "Levante as maos para comecar"
        text_size = max(18, int(min(w, h) * 0.04))
        _text_w, text_h = self.text_renderer.measure(intro_label, size=text_size, weight="regular")
        self._draw_text(
            blended,
            intro_label,
            w // 2,
            logo_bottom + text_gap + text_h,
            size=text_size,
            color_bgr=(45, 45, 45),
            align="center",
            weight="regular",
            alpha=max(0.0, min(1.0, 1.0 - ((1.0 - alpha) * 0.4))),
            fallback_scale=1.0,
        )
        return blended

    def _draw_memory_hud(self, frame, hud: Dict, safe_remaining_s: float = 0.0, countdown_value: Optional[int] = None) -> None:
        # Desenha o HUD do modo memory.
        frame_h, frame_w = frame.shape[:2]
        self._draw_quit_hint(frame)
        score_line = f"Score: {hud['score']}"
        high_line = f"High Score: {self.session_high_score}"
        self._draw_text(
            frame, score_line, frame_w // 2, 66, size=40, color_bgr=(255, 255, 255), align="center", weight="bold", fallback_scale=1.0
        )
        self._draw_text(
            frame, high_line, frame_w // 2, 108, size=28, color_bgr=(255, 255, 255), align="center", weight="regular", fallback_scale=0.75
        )

        if hud["message"]:
            # Mensagem de game over.
            self._draw_text(
                frame, f"{hud['message']} [{self.restart_key.upper()}]",
                20, 156, size=30, color_bgr=(0, 0, 255), align="left", weight="bold", fallback_scale=0.9
            )

        if safe_remaining_s > 0:
            # Indicador de janela segura ativa.
            self._draw_text(
                frame, f"SAFE ACTIVE: {safe_remaining_s:.1f}s",
                20, 188, size=28, color_bgr=(0, 255, 255), align="left", weight="bold", fallback_scale=0.8
            )

        if countdown_value is not None:
            # Contagem regressiva centralizada.
            text = str(countdown_value)
            self._draw_text(
                frame, text, frame_w // 2, frame_h // 2 + 60,
                size=max(80, int(min(frame_w, frame_h) * 0.22)),
                color_bgr=(255, 255, 255),
                align="center",
                weight="bold",
                fallback_scale=3.5,
            )

    def _draw_playground_hud(self, frame) -> None:
        # Desenha o HUD simples do modo playground.
        self._draw_quit_hint(frame)

    def _draw_mode_toast(self, frame, now: float) -> None:
        # Mostra toast temporario com o modo atual.
        if not self.mode_toast:
            return
        message, expires_at = self.mode_toast
        if now > expires_at:
            self.mode_toast = None
            return

        frame_h, frame_w = frame.shape[:2]
        text_w, text_h = self.text_renderer.measure(message, size=28, weight="bold")
        x = max((frame_w - text_w) // 2, 10)
        y = max(frame_h - 30, 30)

        # Painel semi-opaco para melhorar leitura.
        panel_x1 = max(x - 12, 0)
        panel_y1 = max(y - text_h - 12, 0)
        panel_x2 = min(x + text_w + 12, frame_w - 1)
        panel_y2 = min(y + 10, frame_h - 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        self._draw_text(
            frame, message, x, y, size=28, color_bgr=(255, 255, 255), align="left", weight="bold", fallback_scale=0.8
        )

    def _draw_failure_effect(self, frame: np.ndarray, now: float) -> np.ndarray:
        # Aplica overlay vermelho e tremor quando houver falha.
        if now > self.fail_visual_until:
            return frame

        frame_h, frame_w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w - 1, frame_h - 1), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        t = max(0.0, self.fail_visual_until - now)
        shake_px = int(CONFIG["memory_game"].get("fail_shake_px", 12))
        dx = int(math.sin(t * 40.0) * shake_px)
        dy = int(math.cos(t * 28.0) * (shake_px * 0.5))
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken = cv2.warpAffine(frame, transform, (frame_w, frame_h), borderMode=cv2.BORDER_REFLECT)

        self._draw_text(
            shaken,
            self.fail_visual_message,
            frame_w // 2,
            frame_h // 2 + 24,
            size=max(42, int(min(frame_w, frame_h) * 0.08)),
            color_bgr=(255, 255, 255),
            align="center",
            weight="bold",
            fallback_scale=1.4,
        )
        return shaken

    def _apply_command(self, command: Optional[str], now: float) -> None:
        # Executa o comando detectado por gesto.
        if command == CMD_SWITCH_PLAYGROUND:
            self._switch_to_mode(MODE_PLAYGROUND, now, force_new_run=False)
        elif command == CMD_SWITCH_MEMORY:
            self._switch_to_mode(MODE_MEMORY, now, force_new_run=True)
        elif command == CMD_RESTART_MEMORY and self.memory_game:
            self.memory_game.restart(now)
            self._reset_memory_safe_guard()
            self.mode_toast = ("Memory restarted", now + 1.0)

    def _apply_memory_safe_window(
        self,
        events: List[Tuple[int, str]],
        now: float
    ) -> List[Tuple[int, str]]:
        # Aplica janela segura para evitar penalidade por double input.
        if not self.memory_game or self.memory_game.phase != self.memory_game.WAIT_INPUT:
            return events

        filtered: List[Tuple[int, str]] = []
        expected = None
        if self.memory_game.expected_index < len(self.memory_game.sequence):
            expected = self.memory_game.sequence[self.memory_game.expected_index]

        duplicate_window_s = CONFIG["gesture_controls"]["double_input_window_ms"] / 1000.0
        safe_time_s = CONFIG["gesture_controls"]["safe_time_ms"] / 1000.0

        for drum_index, source in events:
            if (
                self.last_memory_input_index == drum_index
                and now - self.last_memory_input_time <= duplicate_window_s
            ):
                self.memory_safe_until = max(self.memory_safe_until, now + safe_time_s)

            self.last_memory_input_index = drum_index
            self.last_memory_input_time = now

            safe_active = now < self.memory_safe_until
            if safe_active and expected is not None and drum_index != expected:
                continue
            filtered.append((drum_index, source))

        return filtered

    def _play_sound_safe(self, sound) -> None:
        # Toca som ignorando falhas para nao quebrar o loop.
        if sound is None:
            return
        try:
            sound.play()
        except Exception as e:
            logger.warning(f"Failed to play effect sound: {e}")

    def _build_synthetic_canvas(self, frame_dim: Tuple[int, int]) -> np.ndarray:
        # Cria canvas neutro para exibir apenas o kit (sem feed da camera).
        frame_w, frame_h = frame_dim
        canvas = np.full((frame_h, frame_w, 3), 220, dtype=np.uint8)
        focus_radius = int(min(frame_w, frame_h) * 0.52)
        cv2.circle(
            canvas,
            (frame_w // 2, int(frame_h * 0.56)),
            focus_radius,
            (236, 236, 236),
            -1,
            lineType=cv2.LINE_AA,
        )
        return canvas

    def _draw_baquetas(self, frame: np.ndarray, hand_inputs: List[Dict]) -> None:
        # Renderiza cada mao detectada usando o asset de baqueta.
        if not self.kit:
            return

        baqueta_cfg = CONFIG.get("baqueta", {})
        asset_path = baqueta_cfg.get("asset")
        if not asset_path:
            return

        min_dim = min(frame.shape[1], frame.shape[0])
        stick_h = max(56, int(min_dim * baqueta_cfg.get("height_pct", 0.26)))
        native_w, native_h = self.kit.renderer.get_asset_size(asset_path)
        if native_h > 0:
            stick_w = max(8, int(stick_h * (native_w / native_h)))
        else:
            stick_w = max(8, int(stick_h * baqueta_cfg.get("width_ratio", 0.09)))
        tip_offset = int(stick_h * baqueta_cfg.get("tip_offset_pct", 0.18))
        stroke_cfg = {
            "enabled": bool(baqueta_cfg.get("stroke_enabled", True)),
            "color_bgr": tuple(baqueta_cfg.get("stroke_color_bgr", (255, 255, 255))),
            "px": int(baqueta_cfg.get("stroke_px", 2)),
        }
        tint_bgr = tuple(baqueta_cfg.get("tint_bgr", (199, 0, 83)))

        for idx, hand_info in enumerate(hand_inputs[:2]):
            tip_x, tip_y = hand_info["index_pos"]
            side_offset = int(stick_w * 0.4) * (-1 if idx == 0 else 1)
            center = (tip_x + side_offset, tip_y - tip_offset)
            self.kit.renderer.draw_asset(
                frame,
                asset_path,
                center,
                (stick_w, stick_h),
                alpha_scale=0.98,
                tint_bgr=tint_bgr,
                stroke=stroke_cfg,
            )

    def update_loop(self) -> None:
        """Processa um frame do video."""
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not initialized or closed.")
            return

        ret, camera_frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            return

        # Camera segue sendo usada para rastreamento, sem ser exibida.
        camera_frame = cv2.flip(camera_frame, 1)
        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        frame_h, frame_w = camera_frame.shape[:2]
        self.kit.update_frame_dim((frame_w, frame_h))
        frame = self._build_synthetic_canvas((frame_w, frame_h))

        current_time = asyncio.get_event_loop().time()
        dt_s = max(0.0, current_time - self.last_frame_time) if self.last_frame_time > 0.0 else 1.0 / 60.0
        self.last_frame_time = current_time
        zone_events_detailed: List[Tuple[int, str, Tuple[int, int]]] = []
        hand_inputs: List[Dict] = []

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Usa a ponta do indicador como posicao principal.
                lm = hand_landmarks.landmark[8]  # Ponta do dedo indicador
                x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                hand_inputs.append({
                    "hand_idx": idx,
                    "landmarks": hand_landmarks,
                    "index_pos": (x, y),
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                })

        self._update_intro_state(hand_inputs, current_time)
        allow_gameplay_input = self.scene_state == SCENE_GAME
        if allow_gameplay_input:
            # Detecta comando por gesto (trocar modo, reiniciar, etc).
            command = self._detect_command_gesture(hand_inputs, current_time)
            if command:
                self._apply_command(command, current_time)

        if allow_gameplay_input:
            for info in hand_inputs:
                idx = info["hand_idx"]
                hand_landmarks = info["landmarks"]
                x, y = info["index_pos"]
                w, h = info["frame_w"], info["frame_h"]
                # Calcula velocidade do movimento da mao.
                px, py, pt = self.prev_positions.get(idx, (x, y, current_time))
                dt = current_time - pt
                vel = (y - py) / dt if dt > 0 else 0.0
                self.prev_positions[idx] = (x, y, current_time)

                # Detecta strike baseado em velocidade e posicao.
                strike_idx = self.kit.get_hit_drum_index_by_strike((x, y), vel, current_time)
                if strike_idx is not None:
                    zone_events_detailed.append((strike_idx, "strike", (x, y)))

                # Detecta eventos por gesto (pinch/fist) sobre um tambor.
                gesture_event = self._detect_gesture_zone_event(
                    idx, hand_landmarks, (x, y), w, h, current_time
                )
                if gesture_event:
                    zone_events_detailed.append((gesture_event[0], gesture_event[1], (x, y)))

        indicator_states = None
        memory_hud: Optional[Dict] = None
        countdown_value: Optional[int] = None
        safe_remaining = 0.0
        draw_playground_hud = False
        emitted_positions: List[Tuple[int, int]] = []

        # Remove eventos duplicados para o mesmo tambor no mesmo frame.
        dedup_events: List[Tuple[int, str]] = []
        dedup_event_positions: List[Tuple[int, str, Tuple[int, int]]] = []
        seen = set()
        for drum_index, source, hit_pos in zone_events_detailed:
            event = (drum_index, source)
            if event[0] in seen:
                continue
            dedup_events.append(event)
            seen.add(event[0])
            dedup_event_positions.append((drum_index, source, hit_pos))

        if allow_gameplay_input and self.current_mode == MODE_MEMORY and self.memory_game:
            # Aplica janela segura e atualiza o jogo de memoria.
            safe_events = self._apply_memory_safe_window(dedup_events, current_time)
            safe_indices = {event[0] for event in safe_events}
            for drum_index, _source, hit_pos in dedup_event_positions:
                if drum_index in safe_indices:
                    emitted_positions.append(hit_pos)
            game_render_data = self.memory_game.update(current_time, safe_events)
            indicator_states = game_render_data["indicator_states"]
            # Toca tambores indicados pelo jogo.
            for drum_index in game_render_data["play_indices"]:
                self.kit.play_drum_by_index(drum_index, current_time)
            # Atualiza high score da sessao.
            self.session_high_score = max(self.session_high_score, game_render_data["hud"]["score"])
            if game_render_data.get("failed_this_frame"):
                # Som e efeito visual de falha.
                self._play_sound_safe(self.fail_sound)
                overlay_ms = CONFIG["memory_game"].get("fail_overlay_ms", 1000)
                self.fail_visual_until = current_time + (overlay_ms / 1000.0)
            if game_render_data.get("sequence_completed_this_frame"):
                # Som de sucesso ao completar sequencia.
                self._play_sound_safe(self.success_sound)
            memory_hud = game_render_data["hud"]
            safe_remaining = max(0.0, self.memory_safe_until - current_time)
            countdown_value = game_render_data.get("countdown_value")
        elif allow_gameplay_input:
            # Modo livre: toca tambores diretamente.
            for drum_index, _source in dedup_events:
                self.kit.play_drum_by_index(drum_index, current_time)
            emitted_positions.extend([hit_pos for _idx, _src, hit_pos in dedup_event_positions])
            if CONFIG.get("playground", {}).get("minimal_hud", True):
                draw_playground_hud = True

        if allow_gameplay_input and self.particles.enabled:
            for hit_pos in emitted_positions:
                self.particles.emit_burst(hit_pos[0], hit_pos[1])
        self.particles.update(dt_s)

        # Desenha tambores e overlays.
        self.kit.draw(frame, indicator_states)
        self.particles.draw(frame)
        self._draw_baquetas(frame, hand_inputs)

        show_tabs = self.scene_state in (SCENE_TRANSITION, SCENE_GAME)
        if show_tabs:
            self._draw_mode_tabs(frame)

        if self.scene_state in (SCENE_INTRO, SCENE_TRANSITION):
            frame = self._draw_intro_overlay(frame, current_time)
            if self.scene_state == SCENE_TRANSITION:
                progress = self._smoothstep(
                    (current_time - self.transition_started_at)
                    / max(float(CONFIG.get("ui", {}).get("intro_transition_ms", 650)) / 1000.0, 1e-4)
                )
                if memory_hud is not None:
                    self._draw_memory_hud(
                        frame,
                        memory_hud,
                        safe_remaining_s=safe_remaining,
                        countdown_value=countdown_value,
                    )
                elif draw_playground_hud:
                    self._draw_playground_hud(frame)
                if progress < 1.0:
                    veil = frame.copy()
                    cv2.rectangle(veil, (0, 0), (frame_w - 1, frame_h - 1), (220, 220, 220), -1)
                    cv2.addWeighted(veil, (1.0 - progress) * 0.10, frame, 1.0 - ((1.0 - progress) * 0.10), 0, frame)
                # Redesenha tabs acima do overlay durante a transicao.
                self._draw_mode_tabs(frame)
                self._draw_quit_hint(frame)
        elif memory_hud is not None:
            self._draw_memory_hud(
                frame,
                memory_hud,
                safe_remaining_s=safe_remaining,
                countdown_value=countdown_value,
            )
        elif draw_playground_hud:
            self._draw_playground_hud(frame)

        frame = self._draw_failure_effect(frame, current_time)
        self._draw_mode_toast(frame, current_time)
        cv2.imshow('DRUMble', frame)

        # Le teclas de controle.
        key = cv2.waitKey(1) & 0xFF
        if self._key_matches(key, self.quit_key):
            logger.info("Exit requested by user.")
            raise SystemExit
        if self._key_matches(key, self.switch_mode_key):
            self._switch_mode(current_time)
        if (
            self._key_matches(key, self.restart_key)
            and self.current_mode == MODE_MEMORY
            and self.memory_game
            and self.memory_game.is_game_over()
        ):
            self.memory_game.restart(current_time)
            self._reset_memory_safe_guard()

    def cleanup(self) -> None:
        """Libera recursos de camera, janelas e audio."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.hands:
            self.hands.close()
        mixer.quit()
        logger.info("Resources cleaned up.")
