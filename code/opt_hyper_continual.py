# Optimizing Safe RL algorithm hyperparameters on proxy tasks
import optuna
import argparse
import numpy as np
import metaworld
import safety_gymnasium
import random
import torch

from safepo.single_agent.train_functions import train_ppo_ewc

def objective(trial, alg, task):
    # suggest hyperparameter dict here, depends on the algorithm selected
    neurons = trial.suggest_categorical("neurons", [32, 64, 128])
    hyperparams = {
        'hidden_sizes': [neurons, neurons],
        'gamma': 0.99,
        'batch_size': trial.suggest_categorical("batch_size", [64, 128, 256]),
        'max_grad_norm': 40
    }

    if alg == 'ppo_ewc':
        hyperparams['target_kl'] = 0.02
        hyperparams['learning_iters'] = int(trial.suggest_float("learning_iters", 35, 45, step=1))
        hyperparams['lambda'] = trial.suggest_float("lambda", 1, 25) # We found that ~10 works well through lit review, so let's search around this area
        policy = train_ppo_ewc(hyperparams, task)
    else:
        raise Exception(f'{alg} not supported, should be ppo_ewc')

    value = evaluate_continual_alg(policy, task)
    return value

def evaluate_continual_alg(policy, task):
    if task == 'SafetyHalfCheetahVelocity-v4':
        env1 = safety_gymnasium.make('SafetyHalfCheetahVelocity-v4') # nominal
        env2 = safety_gymnasium.make('SafetyHalfCheetahVelocity-v5') # back leg missing
    elif task == 'SafetyAntVelocity-v2':
        env1 = safety_gymnasium.make('SafetyAntVelocity-v2') # nominal
        env2 = safety_gymnasium.make('SafetyAntVelocity-v3') # back legs missing
    elif task == 'SafetyContinualWorld':
        ml1 = metaworld.ML1('safe-hammer') 
        env1 = ml1.train_classes['safe-hammer']() 
        task = random.choice(ml1.train_tasks)
        env1.set_task(task)  
        env1._partially_observable = False
        env1._freeze_rand_vec = False

        ml1 = metaworld.ML1('safe-push-wall') 
        env2 = ml1.train_classes['safe-push-wall']() 
        task = random.choice(ml1.train_tasks)
        env2.set_task(task)  
        env2._partially_observable = False
        env2._freeze_rand_vec = False

    n_repeat = 5
    overall_rewards = []
    # First, evaluate n times on task A (forgetting)
    for repeat_num in range(n_repeat):
        obs, info = env1.reset()
        done = False
        rewards, costs = [], []
        while not done:
            action, _, _, _ = policy.step(torch.tensor(obs, dtype=torch.float32), deterministic=True)
            obs, reward, cost, terminated, truncated, info = env1.step(action.detach().numpy())
            rewards.append(reward)
            done = terminated or truncated

        overall_rewards.append(np.sum(rewards))

    # Then, evaluate n times on task B (learning)
    for repeat_num in range(n_repeat):
        obs, info = env2.reset()
        done = False
        rewards, costs = [], []
        while not done:
            action, _, _, _ = policy.step(torch.tensor(obs, dtype=torch.float32), deterministic=True)
            obs, reward, cost, terminated, truncated, info = env2.step(action.detach().numpy())
            rewards.append(reward)
            done = terminated or truncated

        overall_rewards.append(np.sum(rewards))

    return np.mean(overall_rewards)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, help="Choose from 'ppo_ewc'")
    parser.add_argument('--task', type=str, help="Choose from 'cheetah', 'cw', or 'ant'")
    args = parser.parse_args()

    if args.task == 'cheetah':
        task = 'SafetyHalfCheetahVelocity-v4'
    elif args.task == 'cw':
        task = 'SafetyContinualWorld'
    elif args.task == 'ant':
        task = 'SafetyAntVelocity-v2'
    else:
        raise Exception(f'{args.task} not supported, should be one of cheetah, cw, or ant')

    storage_url = f"sqlite:///hyperparams/{args.alg}_{args.task}.db"

    study = optuna.create_study(storage=storage_url, study_name=f"{args.alg}_{args.task}", direction="maximize")
    study.optimize(lambda trial: objective(trial, args.alg, task), n_trials=50, n_jobs=1)