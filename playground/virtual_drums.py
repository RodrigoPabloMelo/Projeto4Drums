import logging
from typing import Optional

from config.config import CONFIG

logger = logging.getLogger(__name__)

import asyncio
import math
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
from pygame import mixer
import numpy as np
from playground.drum_kit import DrumKit
from playground.memory_game import MemoryGameController
from config.config import CONFIG

# Logger do modulo.
logger = logging.getLogger(__name__)

# Constantes de modo e comandos.
MODE_PLAYGROUND = "playground"
MODE_MEMORY = "memory"
CMD_SWITCH_PLAYGROUND = "switch_playground"
CMD_SWITCH_MEMORY = "switch_memory"
CMD_RESTART_MEMORY = "restart_memory"


class VirtualDrums:
    """Classe principal da aplicacao de bateria virtual."""
    def __init__(self):
        # Componentes do MediaPipe.
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
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
                self.memory_game.start_new_run(asyncio.get_event_loop().time())
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
                return CMD_SWITCH_PLAYGROUND
            if self._is_pointing_up(info["landmarks"]):
                return CMD_SWITCH_MEMORY
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

    def _draw_memory_hud(self, frame, hud: Dict, safe_remaining_s: float = 0.0, countdown_value: Optional[int] = None) -> None:
        # Desenha o HUD do modo memory.
        font_scale = CONFIG["memory_game"]["font_scale"]
        frame_h, frame_w = frame.shape[:2]
        score_line = f"Score: {hud['score']}"
        high_line = f"High Score: {self.session_high_score}"
        score_w, score_h = cv2.getTextSize(score_line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        high_w, _ = cv2.getTextSize(high_line, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv2.putText(
            frame,
            score_line,
            ((frame_w - score_w) // 2, 32 + score_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            high_line,
            ((frame_w - high_w) // 2, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        lines = [
            "Mode: Memory",
            f"Combo: x{hud['combo_multiplier']:.2f}",
            f"Round: {hud['round']}",
            f"State: {hud['state']}",
            f"[{self.switch_mode_key.upper()}] Switch  [{self.quit_key.upper()}] Quit",
        ]
        y = 30
        for line in lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            y += 32

        if hud["message"]:
            # Mensagem de game over.
            cv2.putText(
                frame,
                f"{hud['message']} [{self.restart_key.upper()}]",
                (20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        if safe_remaining_s > 0:
            # Indicador de janela segura ativa.
            cv2.putText(
                frame,
                f"SAFE ACTIVE: {safe_remaining_s:.1f}s",
                (20, y + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        if countdown_value is not None:
            # Contagem regressiva centralizada.
            text = str(countdown_value)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 7)
            tx = (frame_w - text_size[0]) // 2
            ty = (frame_h + text_size[1]) // 2
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 7)

    def _draw_playground_hud(self, frame) -> None:
        # Desenha o HUD simples do modo playground.
        lines = [
            "Mode: Playground",
            f"[{self.switch_mode_key.upper()}] Switch to Memory  [{self.quit_key.upper()}] Quit",
        ]
        y = 30
        for line in lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            y += 32

    def _draw_mode_toast(self, frame, now: float) -> None:
        # Mostra toast temporario com o modo atual.
        if not self.mode_toast:
            return
        message, expires_at = self.mode_toast
        if now > expires_at:
            self.mode_toast = None
            return

        frame_h, frame_w = frame.shape[:2]
        text_size, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size
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
        cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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

        text = self.fail_visual_message
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
        tx = (frame_w - text_size[0]) // 2
        ty = (frame_h + text_size[1]) // 2
        cv2.putText(shaken, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
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

    def update_loop(self) -> None:
        """Processa um frame do video."""
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not initialized or closed.")
            return

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            return

        # Espelha a imagem para ficar mais natural para o usuario.
        frame = cv2.flip(frame, 1)
        # Converte para RGB para o MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        current_time = asyncio.get_event_loop().time()
        zone_events: List[Tuple[int, str]] = []
        hand_inputs: List[Dict] = []

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Usa a ponta do indicador como posicao principal.
                lm = hand_landmarks.landmark[8]  # Ponta do dedo indicador
                w, h = frame.shape[1], frame.shape[0]
                x, y = int(lm.x * w), int(lm.y * h)
                hand_inputs.append({
                    "hand_idx": idx,
                    "landmarks": hand_landmarks,
                    "index_pos": (x, y),
                    "frame_w": w,
                    "frame_h": h,
                })

                # Desenha os pontos da mao.
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Detecta comando por gesto (trocar modo, reiniciar, etc).
        command = self._detect_command_gesture(hand_inputs, current_time)
        if command:
            self._apply_command(command, current_time)
        else:
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
                    zone_events.append((strike_idx, "strike"))

                # Detecta eventos por gesto (pinch/fist) sobre um tambor.
                gesture_event = self._detect_gesture_zone_event(
                    idx, hand_landmarks, (x, y), w, h, current_time
                )
                if gesture_event:
                    zone_events.append(gesture_event)

        indicator_states = None
        # Remove eventos duplicados para o mesmo tambor no mesmo frame.
        dedup_events: List[Tuple[int, str]] = []
        seen = set()
        for event in zone_events:
            if event[0] in seen:
                continue
            dedup_events.append(event)
            seen.add(event[0])

        if self.current_mode == MODE_MEMORY and self.memory_game:
            # Aplica janela segura e atualiza o jogo de memoria.
            safe_events = self._apply_memory_safe_window(dedup_events, current_time)
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
            safe_remaining = max(0.0, self.memory_safe_until - current_time)
            self._draw_memory_hud(
                frame,
                game_render_data["hud"],
                safe_remaining_s=safe_remaining,
                countdown_value=game_render_data.get("countdown_value"),
            )
        else:
            # Modo livre: toca tambores diretamente.
            for drum_index, _source in dedup_events:
                self.kit.play_drum_by_index(drum_index, current_time)
            if CONFIG.get("playground", {}).get("minimal_hud", True):
                self._draw_playground_hud(frame)

        # Desenha tambores e overlays.
        self.kit.draw(frame, indicator_states)
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
