"""Continual World environment with a safety constraint on not spilling a mug."""

from gymnasium import Env
# import metaworld
import numpy as np
import random

TASK_CYCLE = ['safe-hammer', 'safe-push-wall', 'safe-faucet', 'safe-push-back', 'safe-stick-pull', 
              'safe-hammer', 'safe-push-wall', 'safe-faucet', 'safe-push-back', 'safe-stick-pull',
              'safe-hammer', 'safe-push-wall', 'safe-faucet', 'safe-push-back', 'safe-stick-pull'] # recommend 15 million

TASK_NUMS = {
    'hammer': 0,
    'push-wall': 1,
    'faucet': 2,
    'push-back': 3,
    'stick-pull': 4
}

TASK_LENGTH = 1_000_000 / 10 # divide by number of parallel processes

class SafetyContinualWorldEnv(Env):
    """HalfCheetah environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        self.current_task = 0
        self.steps_since_change = 0
        self.current_task_name = TASK_CYCLE[self.current_task]
        super().__init__(**kwargs)
        self.task_nums = TASK_NUMS

        self.change_task()

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        cost = info['unscaled_cost']
        return observation, reward, cost, terminated, truncated, info

    def reset(self):
        return self.env.reset()
    
    def check_task(self):
        if self.steps_since_change > TASK_LENGTH:
            self.steps_since_change = 0
            self.current_task = (self.current_task + 1) % len(TASK_CYCLE)
            self.change_task()

    def change_task(self):
        task_name = TASK_CYCLE[self.current_task]
        self.current_task_name = task_name

        ml1 = metaworld.ML1(self.current_task_name)
        env = ml1.train_classes[self.current_task_name]() 
        task = random.choice(ml1.train_tasks)

        env.set_task(task)  
        env._partially_observable = False
        self.env = env

        self.reset()
