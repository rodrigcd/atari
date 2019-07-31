import gym
import argparse
import numpy as np
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from game_models.ge_game_model import GETrainer, GESolver
from gym_wrappers import MainGymWrapper


class Atari_env(object):

    def __init__(self, **kwargs):

        self.game_name = kwargs["game_name"]
        self.agent_model = kwargs["game_model"]
        self.game_mode = kwargs["game_mode"]
        self.render = kwargs["render"]
        self.total_step_limit = kwargs["total_step_limit"]
        self.total_run_limit = kwargs["total_run_limit"]
        self.clip_reward = kwargs["clip_reward"]
        self.channel_first = kwargs["channel_first"]
        self.frames_per_obs = kwargs["frames_per_obs"]
        self.frame_size = kwargs["frame_size"]
        self.input_shape = kwargs["input_shape"]

        self.env_name = self.game_name + "Deterministic-v4"
        self._initialize_environmnt()

    def _initialize_environmnt(self):
        self.env = MainGymWrapper.wrap(gym.make(self.env_name))

    def _initialize_agent(self):
        self.game_model = self._game_model(self.agent_model + "_" + self.game_mode,
                                           self.game_name,
                                           self.env.action_space.n,
                                           self.input_shape)

    def run_rl_agent(self):
        self._initialize_agent()

        if isinstance(self.game_model, GETrainer):
            self.game_model.genetic_evolution(self.env)

        run = 0
        total_step = 0
        while True:
            if self.total_run_limit is not None and run >= self.total_run_limit:
                print("Reached total run limit of: " + str(self.total_run_limit))
                exit(0)

            run += 1
            current_state = self.env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= self.total_step_limit:
                    print("Reached total step limit of: " + str(self.total_step_limit))
                    exit(0)
                total_step += 1
                step += 1

                if self.render:
                    self.env.render()

                action = self.game_model.move(current_state)
                next_state, reward, terminal, info = self.env.step(action)
                if self.clip_reward:
                    np.sign(reward)
                score += reward
                self.game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                self.game_model.step_update(total_step)

                if terminal:
                    self.game_model.save_run(score, step, run)
                    break

    def _game_model(self, game_mode,game_name, action_space, input_shape):
        if game_mode == "ddqn_training":
            return DDQNTrainer(game_name, input_shape, action_space)
        elif game_mode == "ddqn_testing":
            return DDQNSolver(game_name, input_shape, action_space)
        elif game_mode == "ge_training":
            return GETrainer(game_name, input_shape, action_space)
        elif game_mode == "ge_testing":
            return GESolver(game_name, input_shape, action_space)
        else:
            print("Unrecognized mode. Use --help")
            exit(1)

    def play_as_human(self):
        pass