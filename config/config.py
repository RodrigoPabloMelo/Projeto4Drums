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
    # Configuracoes visuais globais.
    'ui': {
        # Cor principal da marca para o kit.
        'brand_color_hex': '#A83500',
        # OpenCV usa BGR.
        'brand_color_bgr': (0, 53, 168),
        # Opacidade base do overlay da intro.
        'intro_overlay_alpha_idle': 0.65,
        # Duracao da transicao intro -> jogo.
        'intro_transition_ms': 650,
        # Tempo minimo com duas maos levantadas para iniciar.
        'intro_hold_ms': 450,
        # Limite em Y para considerar mao levantada (indicador acima do punho).
        'hand_raise_margin': 0.02,
        # Configuracao da tab de modo no topo.
        'tab': {
            'container_width_pct': 0.60,
            'container_height_px': 92,
            'top_margin_px': 22,
            'corner_radius_px': 46,
            'inner_padding_px': 8,
            'container_bg_bgr': (245, 245, 245),
            'active_bg_bgr': (199, 0, 83),  # #5300C7
            'text_active_bgr': (255, 255, 255),
            'text_inactive_bgr': (199, 0, 83),
        },
        # Configuracao especifica da intro.
        'intro': {
            'panel_opacity': 0.75,
            'text_gap_px': 16,
        },
    },
    'assets': {
        'mode_icons': {
            'playground': 'assets/playground.svg',
            'memory': 'assets/memory.svg',
        },
    },
    # Configuracoes de tipografia.
    'fonts': {
        'instrument_sans_regular': 'assets/fonts/InstrumentSans-Regular.ttf',
        'instrument_sans_bold': 'assets/fonts/InstrumentSans-Bold.ttf',
    },
    # Configuracoes do sistema de particulas.
    'particles': {
        'enabled': True,
        'burst_count': 16,
        'speed_min': 130.0,
        'speed_max': 320.0,
        'lifetime_ms': 380,
        'size_min': 2.0,
        'size_max': 5.2,
        'gravity': 520.0,
        'color_bgr': (0, 53, 168),
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
    # Vinculos de som por papel logico do tambor.
    'sound_bindings': {
        'snare': 'sounds/snare_1.wav',
        'hihat': 'sounds/hihat_1.wav',
        'crash': 'sounds/crash_1.wav',
        'bass': 'sounds/bass.wav',
    },
    # Configuracao visual das baquetas (maos).
    'baqueta': {
        # Asset SVG da baqueta.
        'asset': 'assets/baqueta.svg',
        # Cor da baqueta em BGR (#5300C7).
        'tint_bgr': (199, 0, 83),
        # Altura relativa em funcao do menor lado do canvas.
        'height_pct': 0.26,
        # Largura = height * width_ratio.
        'width_ratio': 0.09,
        # Distancia vertical do centro da baqueta para a ponta do indicador.
        'tip_offset_pct': 0.18,
        # Stroke branco ao redor da baqueta.
        'stroke_enabled': True,
        'stroke_color_bgr': (255, 255, 255),
        'stroke_px': 2,
    },
    # Lista de tambores em visao superior.
    'drums': [
        {
            'id': 'cymbal_left',
            'asset': 'assets/cymbal-left.svg',
            'position_pct': (0.30, 0.21),
            'size_pct': 0.16,
            'radius_pct': 0.08,
            'sound_binding': 'hihat',
        },
        {
            'id': 'cymbal_right',
            'asset': 'assets/cymbal-right.svg',
            'position_pct': (0.68, 0.18),
            'size_pct': 0.24,
            'radius_pct': 0.10,
            'sound_binding': 'crash',
        },
        {
            'id': 'snare_mid_left',
            'asset': 'assets/snare.svg',
            'position_pct': (0.46, 0.48),
            'size_pct': 0.19,
            'radius_pct': 0.09,
            'sound_binding': 'snare',
        },
        {
            'id': 'snare_mid_right',
            'asset': 'assets/snare.svg',
            'position_pct': (0.58, 0.48),
            'size_pct': 0.17,
            'radius_pct': 0.085,
            'sound_binding': 'snare',
        },
        {
            'id': 'kick_left',
            'asset': 'assets/tom1.svg',
            'position_pct': (0.24, 0.74),
            'size_pct': 0.30,
            'radius_pct': 0.12,
            'sound_binding': 'bass',
        },
        {
            'id': 'kick_right',
            'asset': 'assets/tom2.svg',
            'position_pct': (0.73, 0.74),
            'size_pct': 0.36,
            'radius_pct': 0.14,
            'sound_binding': 'bass',
        },
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
