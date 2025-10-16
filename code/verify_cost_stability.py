# Verify Cost Stability
# Ensure the robot arm in metaworld won't violate safety without moving
# i.e., the mug can't spawn in such a way it falls over
# If it does, the agent could be misled

import random
import numpy as np
import metaworld
from tqdm import tqdm

def initialize_env(env_name):
    ml1 = metaworld.ML1(env_name) 
    env = ml1.train_classes[env_name]() 
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  
    env._partially_observable = False
    env._freeze_rand_vec = False
    obs, info = env.reset()

    return env

if __name__=="__main__":
    env_names = ['safe-drawer-close']

    num_repeat = 100

    for env_name in env_names:
        env = initialize_env(env_name)
        for n in tqdm(range(num_repeat)):
            obs, info = env.reset()
            done = False
            costs = 0
            while not done: 
                action = np.array([0, 0, 0, 0])
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated or (int(info['success']) == 1)
                costs += info['unscaled_cost']
            if costs > 0:
                print(f'{env_name} FAILED with cost {costs}')

        

