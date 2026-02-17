CONFIG = {
    'hit_velocity_threshold': 1000,
    'drum_cooldown': 0.3,
    'game_mode_enabled': True,
    'drums': [
        {'name': 'Snare', 'pos': (0.5, 0.7), 'radius': 80, 'sound': 'sounds/snare_1.wav'},
        {'name': 'HiHat', 'pos': (0.3, 0.7), 'radius': 70, 'sound': 'sounds/hihat_1.wav'},
        {'name': 'Crash', 'pos': (0.7, 0.5), 'radius': 70, 'sound': 'sounds/crash_1.wav'},
    ],
    'memory_game': {
        'highlight_ms': 500,
        'between_steps_ms': 220,
        'input_timeout_ms': 2500,
        'base_points': 10,
        'combo_step': 0.15,
        'max_combo_multiplier': 4.0,
        'gesture_pinched_threshold': 35.0,
        'fist_fold_threshold': 0.45,
        'font_scale': 0.8,
        'gesture_debounce_ms': 250,
        'feedback_ms': 220,
        'round_bonus': 25,
        'indicator_colors': {
            'idle': (70, 70, 70),
            'active': (0, 220, 255),
            'correct': (0, 255, 0),
            'wrong': (0, 0, 255)
        }
    },
    'camera_index': 0,
    'hands_config': {
        'max_num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.7
    },
    'fps': 60
}
