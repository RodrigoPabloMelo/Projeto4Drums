import asyncio
import logging
import math
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
from pygame import mixer
from playground.drum_kit import DrumKit
from playground.memory_game import MemoryGameController
from config.config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class VirtualDrums:
    """Main class for the virtual drums application."""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.cap = None
        self.kit = None
        self.memory_game: Optional[MemoryGameController] = None
        self.game_mode_enabled = CONFIG.get("game_mode_enabled", False)
        self.prev_positions: Dict[int, Tuple[int, int, float]] = {}
        self.prev_gesture_active: Dict[Tuple[int, str], bool] = {}
        self.last_gesture_event: Dict[Tuple[int, int, str], float] = {}

    def setup(self) -> None:
        """Initialize pygame, MediaPipe, and camera."""
        try:
            mixer.init()
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            raise

        try:
            self.hands = self.mp_hands.Hands(**CONFIG['hands_config'])
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise

        try:
            self.cap = cv2.VideoCapture(CONFIG['camera_index'])
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read from camera.")
            h, w = frame.shape[:2]
            self.kit = DrumKit((w, h))
            if self.game_mode_enabled:
                self.memory_game = MemoryGameController(
                    self.kit.get_drum_count(),
                    CONFIG["memory_game"]
                )
                self.memory_game.start_new_run(asyncio.get_event_loop().time())
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise

    def _is_pinched(self, hand_landmarks, frame_w: int, frame_h: int) -> bool:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        pinch_dist = math.hypot(
            (thumb_tip.x - index_tip.x) * frame_w,
            (thumb_tip.y - index_tip.y) * frame_h
        )
        return pinch_dist < CONFIG["memory_game"]["gesture_pinched_threshold"]

    def _is_fist(self, hand_landmarks) -> bool:
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

    def _detect_gesture_zone_event(
        self,
        hand_idx: int,
        hand_landmarks,
        index_pos: Tuple[int, int],
        frame_w: int,
        frame_h: int,
        current_time: float
    ) -> Optional[Tuple[int, str]]:
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

        for gesture_kind in ("pinch", "fist"):
            self.prev_gesture_active[(hand_idx, gesture_kind)] = (gesture_kind == active_kind)

        if not active_kind:
            return None

        if was_active:
            return None

        drum_index = self.kit.get_drum_index_at_position(index_pos)
        if drum_index is None:
            return None

        debounce_s = CONFIG["memory_game"]["gesture_debounce_ms"] / 1000.0
        event_key = (hand_idx, drum_index, active_kind)
        last_event_time = self.last_gesture_event.get(event_key, 0.0)
        if current_time - last_event_time < debounce_s:
            return None

        self.last_gesture_event[event_key] = current_time
        return drum_index, active_kind

    def _draw_hud(self, frame, hud: Dict) -> None:
        font_scale = CONFIG["memory_game"]["font_scale"]
        lines = [
            f"Mode: Memory",
            f"Score: {hud['score']}",
            f"Combo: x{hud['combo_multiplier']:.2f}",
            f"Round: {hud['round']}",
            f"State: {hud['state']}",
        ]

        y = 30
        for line in lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            y += 32

        if hud["message"]:
            cv2.putText(frame, hud["message"], (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    def update_loop(self) -> None:
        """Process one frame of the video feed."""
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not initialized or closed.")
            return

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        current_time = asyncio.get_event_loop().time()
        zone_events: List[Tuple[int, str]] = []

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                lm = hand_landmarks.landmark[8]  # Index finger tip
                w, h = frame.shape[1], frame.shape[0]
                x, y = int(lm.x * w), int(lm.y * h)

                # Compute vertical velocity
                vel = 0.0
                if idx in self.prev_positions:
                    px, py, pt = self.prev_positions[idx]
                    dt = current_time - pt
                    vel = (y - py) / dt if dt > 0 else 0.0
                self.prev_positions[idx] = (x, y, current_time)

                if self.game_mode_enabled and self.memory_game:
                    strike_idx = self.kit.get_hit_drum_index_by_strike((x, y), vel, current_time)
                    if strike_idx is not None:
                        zone_events.append((strike_idx, "strike"))

                    gesture_event = self._detect_gesture_zone_event(
                        idx, hand_landmarks, (x, y), w, h, current_time
                    )
                    if gesture_event:
                        zone_events.append(gesture_event)
                else:
                    self.kit.interact((x, y), vel)

                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        indicator_states = None
        if self.game_mode_enabled and self.memory_game:
            dedup_events: List[Tuple[int, str]] = []
            seen = set()
            for event in zone_events:
                if event[0] in seen:
                    continue
                dedup_events.append(event)
                seen.add(event[0])

            game_render_data = self.memory_game.update(current_time, dedup_events)
            indicator_states = game_render_data["indicator_states"]
            for drum_index in game_render_data["play_indices"]:
                self.kit.play_drum_by_index(drum_index, current_time)
            self._draw_hud(frame, game_render_data["hud"])

        self.kit.draw(frame, indicator_states)
        cv2.imshow('Virtual Drums', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Exit requested by user.")
            raise SystemExit
        if (
            key == ord('r')
            and self.game_mode_enabled
            and self.memory_game
            and self.memory_game.is_game_over()
        ):
            self.memory_game.restart(asyncio.get_event_loop().time())

    def cleanup(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.hands:
            self.hands.close()
        mixer.quit()
        logger.info("Resources cleaned up.")
