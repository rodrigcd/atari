import atari_py
import numpy as np
import gym
from gym_wrappers import MainGymWrapper
import time


def run_game(n_rep=2000):
    env_name = 'BreakoutDeterministic-v4'
    print(env_name)
    env = MainGymWrapper.wrap(gym.make(env_name))
    env.reset()
    print("Metadata", env.metadata)
    print("Actions:", env.action_space)
    print("Observation:", env.observation_space)

    actions = env.action_space
    env.render()
    for i in range(n_rep):
        next_state, reward, terminal, info = env.step(np.random.randint(low=0, high=4))
        env.render()
        time.sleep(0.4)
        print("Four consecutive frames")
        print(info)
        print(np.asarray(next_state).shape)
        print(reward)
        print(terminal)


if __name__ == "__main__":
    run_game()


