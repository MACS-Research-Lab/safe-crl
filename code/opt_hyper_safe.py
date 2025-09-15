# Optimizing Safe RL algorithm hyperparameters on proxy tasks
import optuna
import argparse
import numpy as np
import metaworld
import safety_gymnasium
import random
import torch

from safepo.single_agent.train_functions import train_cpo, train_cppo, train_ppo_lag

def objective(trial, alg, task):
    # suggest hyperparameter dict here, depends on the algorithm selected
    neurons = trial.suggest_categorical("neurons", [32, 64, 128])
    hyperparams = {
        'hidden_sizes': [neurons, neurons],
        'gamma': 0.99,
        'batch_size': trial.suggest_categorical("batch_size", [64, 128, 256]),
        'max_grad_norm': 40
    }

    if alg == 'cpo':
        hyperparams['target_kl'] = 0.01
        hyperparams['step_fraction'] = trial.suggest_float("step_fraction", 0.6, 1)
        hyperparams['learning_iters'] = 10
        policy = train_cpo(hyperparams, task)
    elif alg == 'ppo_lag':
        hyperparams['learning_iters'] = 40
        hyperparams['target_kl'] = 0.02
        hyperparams['lagrangian_multiplier_lr'] = trial.suggest_float("lagrangian_multiplier_lr", 0.02, 0.04)
        policy = train_ppo_lag(hyperparams, task) 
    elif alg == 'cppo_pid':
        hyperparams['target_kl'] = 0.02
        hyperparams['learning_iters'] = 40
        policy = train_cppo(hyperparams, task)
    else:
        raise Exception(f'{alg} not supported, should be one of cpo, ppo_lag, cppo_pid')

    value = evaluate_safe_alg(policy, task)
    return value

def evaluate_safe_alg(policy, task):
    if task == 'SafetyHalfCheetahVelocity-v4':
        env = safety_gymnasium.make('SafetyHalfCheetahVelocity-v4')
    elif task == 'SafetyContinualWorld':
        ml1 = metaworld.ML1('safe-hammer') 
        env = ml1.train_classes['safe-hammer']() 
        task = random.choice(ml1.train_tasks)
        env.set_task(task)  
        env._partially_observable = False
        env._freeze_rand_vec = False

    n_repeat = 5
    overall_rewards, overall_costs = [], []
    for repeat_num in range(n_repeat):
        obs, info = env.reset()
        done = False
        rewards, costs = [], []
        while not done:
            action, _, _, _ = policy.step(torch.tensor(obs, dtype=torch.float32), deterministic=True)
            if task =='SafetyHalfCheetahVelocity-v4':
                obs, reward, cost, terminated, truncated, info = env.step(action.detach().numpy())
            else:
                obs, reward, terminated, truncated, info = env.step(action.detach().numpy())
                cost = info['unscaled_cost']
            rewards.append(reward)
            costs.append(cost)
            done = terminated or truncated

        overall_rewards.append(np.sum(rewards))
        overall_costs.append(np.sum(costs))

    return np.mean(overall_rewards) - np.mean(overall_costs)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, help="Choose from 'cpo', 'ppo_lag', 'cppo_pid'")
    parser.add_argument('--task', type=str, help="Choose from 'cheetah' or 'cw'")
    args = parser.parse_args()

    if args.task == 'cheetah':
        task = 'SafetyHalfCheetahVelocity-v4'
    elif args.task == 'cw':
        task = 'SafetyContinualWorld'
    else:
        raise Exception(f'{args.task} not supported, should be one of cheetah, cw')

    storage_url = f"sqlite:///hyperparams/{args.alg}_{args.task}.db"

    study = optuna.create_study(storage=storage_url, study_name=f"{args.alg}_{args.task}", direction="maximize")
    study.optimize(lambda trial: objective(trial, args.alg, task), n_trials=50, n_jobs=1)