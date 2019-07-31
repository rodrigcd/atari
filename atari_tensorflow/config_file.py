import atari_py

""" Available Games """
available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))

SELECTED_GAME = "Breakout"
SELECTED_MODEL = "ddqn"
SELECTED_MODE = "training"
RENDER_GAME = False
STEP_LIMIT = 5000000
RUN_LIMIT = None
CLIP_REWARD = True
CHANNEL_FIRST = True

""" Input Options """
FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)

""" Agent Options """
GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS


env_options_kwargs = {"game_name": SELECTED_GAME,
                      "game_model": SELECTED_MODEL,
                      "game_mode": SELECTED_MODE,
                      "render": RENDER_GAME,
                      "total_step_limit": STEP_LIMIT,
                      "total_run_limit": RUN_LIMIT,
                      "clip_reward": CLIP_REWARD,
                      "channel_first": CHANNEL_FIRST,
                      "frames_per_obs": FRAMES_IN_OBSERVATION,
                      "frame_size": FRAME_SIZE,
                      "input_shape": INPUT_SHAPE}

agent_options_kwargs = {"gamma": GAMMA,
                        "memory_size": MEMORY_SIZE,
                        "batch_size": BATCH_SIZE,
                        "training_frequency": TRAINING_FREQUENCY,
                        "target_network_update_freq": TARGET_NETWORK_UPDATE_FREQUENCY,
                        "model_persistence_update_freq": MODEL_PERSISTENCE_UPDATE_FREQUENCY,
                        "replay_start_size": REPLAY_START_SIZE,
                        "exploration_max": EXPLORATION_MAX,
                        "exploration_min": EXPLORATION_MIN,
                        "exploration_test": EXPLORATION_TEST,
                        "exploration_steps": EXPLORATION_STEPS,
                        "exploration_decay": EXPLORATION_DECAY}