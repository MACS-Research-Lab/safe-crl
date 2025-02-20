# Optimizing Safe RL algorithm hyperparameters on proxy tasks
import optuna
import argparse
import numpy as np

from safepo.single_agent.train_functions import train_cpo, train_cppo

def objective(trial, alg, task):
    # suggest hyperparameter dict here, depends on the algorithm selected
    hyperparams = {}

    if alg == 'cpo':
        hyperparams = {}
        policy = train_cpo(hyperparams, task)
    elif alg == 'ppo_lag':
        hyperparams = None
        policy = None 
    elif alg == 'cppo_pid':
        hyperparams = None
        policy = None
    else:
        raise Exception(f'{alg} not supported, should be one of cpo, ppo_lag, cppo_pid')

    value = evaluate_safe_alg(policy, task)
    return value

def evaluate_safe_alg(policy, task):
    if task == 'cheetah':
        pass
    elif task == 'cw':
        pass
    else:
        raise Exception(f'{task} not supported, should be one of cheetah, cw')

    n_repeat = 5
    for repeat_num in range(n_repeat):
        done = False
        while not done:
            pass
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, help="Choose from 'cpo', 'ppo_lag', 'cppo_pid'")
    parser.add_argument('--task', type=str, help="Choose from 'cheetah' or 'cw'")
    args = parser.parse_args()

    objective(None, args.alg, args.task)