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

options_kwargs = {"game_name": SELECTED_GAME,
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