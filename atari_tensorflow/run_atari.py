from atari_tensorflow.atari_env import Atari_env
from atari_tensorflow.config_file import *


def main():
    atari_env = Atari_env(**options_kwargs)
    atari_env.run_rl_agent()


if __name__ == "__main__":
    main()
