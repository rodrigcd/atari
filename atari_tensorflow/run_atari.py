from atari_tensorflow.atari_env import AtariEnv
from atari_tensorflow.config_file import env_options_kwargs, agent_options_kwargs
from atari_tensorflow.agents.base_agent import BaseAgent


def main():
    rl_agent = BaseAgent(**agent_options_kwargs)
    atari_env = AtariEnv(**env_options_kwargs)
    atari_env.run_rl_agent()


if __name__ == "__main__":
    main()
