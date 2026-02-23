CONFIG = {
    # Velocidade minima para considerar um golpe no tambor.
    'hit_velocity_threshold': 1000,
    # Tempo de espera entre toques no mesmo tambor.
    'drum_cooldown': 0.3,
    # Configuracao geral do modo da aplicacao.
    'app_mode': {
        # Modo inicial: "playground" ou "memory".
        'default_mode': 'playground',  # playground | memory
        # Teclas de controle do app.
        'controls': {
            # Alterna entre os modos.
            'switch_mode_key': 'm',
            # Reinicia o modo memory.
            'restart_key': 'r',
            # Sai da aplicacao.
            'quit_key': 'q'
        }
    },
    # Configuracoes do modo playground.
    'playground': {
        # Mostra HUD minimo quando True.
        'minimal_hud': True
    },
    # Configuracoes para comandos por gestos.
    'gesture_controls': {
        # Liga/desliga reconhecimento de gestos de comando.
        'enabled': True,
        # Janela segura para evitar penalidade em toques repetidos.
        'safe_time_ms': 1000,
        # Janela para detectar double input do mesmo tambor.
        'double_input_window_ms': 250,
        # Limite para detectar "apontando para cima".
        'pointing_up_fold_threshold': 0.5,
        # Limite para detectar gesto "vitoria".
        'victory_fold_threshold': 0.5,
        # Debounce para evitar comandos repetidos.
        'gesture_command_debounce_ms': 700,
        # Tempo de segurar gesto para confirmar comando.
        'command_hold_ms': 3000
    },
    # Lista de tambores, suas posicoes e sons.
    'drums': [
        {'name': 'Snare', 'pos': (0.5, 0.8), 'radius': 160, 'sound': 'sounds/snare_1.wav'},
        {'name': 'HiHat', 'pos': (0.3, 0.7), 'radius': 120, 'sound': 'sounds/hihat_1.wav'},
        {'name': 'Crash', 'pos': (0.7, 0.5), 'radius': 100, 'sound': 'sounds/crash_1.wav'},
    ],
    # Configuracoes do jogo de memoria.
    'memory_game': {
        # Contagem antes de iniciar a rodada.
        'round_ready_countdown_ms': 3000,
        # Duracao do destaque de cada passo.
        'highlight_ms': 500,
        # Intervalo entre passos da sequencia.
        'between_steps_ms': 220,
        # Tempo maximo para o usuario responder.
        'input_timeout_ms': 2500,
        # Pontos base por acerto.
        'base_points': 10,
        # Crescimento do multiplicador de combo.
        'combo_step': 0.15,
        # Limite do multiplicador de combo.
        'max_combo_multiplier': 4.0,
        # Som de falha.
        'fail_sound': 'sounds/fail_1.wav',
        # Som de sucesso.
        'success_sound': 'sounds/success_1.wav',
        # Duracao do overlay de falha.
        'fail_overlay_ms': 1000,
        # Intensidade do tremor na falha.
        'fail_shake_px': 12,
        # Limite de distancia para gesto de pinça.
        'gesture_pinched_threshold': 35.0,
        # Limite para detectar punho fechado.
        'fist_fold_threshold': 0.45,
        # Escala da fonte do HUD.
        'font_scale': 0.8,
        # Debounce para gestos do jogo.
        'gesture_debounce_ms': 250,
        # Duracao do feedback visual.
        'feedback_ms': 220,
        # Bonus por completar a rodada.
        'round_bonus': 25,
        # Cores dos indicadores por estado.
        'indicator_colors': {
            'idle': (70, 70, 70),
            'active': (0, 220, 255),
            'correct': (0, 255, 0),
            'wrong': (0, 0, 255)
        }
    },
    # Indice da camera usada pelo OpenCV.
    'camera_index': 0,
    # Configuracoes do MediaPipe Hands.
    'hands_config': {
        # Numero maximo de maos detectadas.
        'max_num_hands': 2,
        # Confianca minima para deteccao inicial.
        'min_detection_confidence': 0.7,
        # Confianca minima para rastreamento.
        'min_tracking_confidence': 0.7
    },
    # FPS alvo do loop principal.
    'fps': 60
}
