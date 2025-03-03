import argparse
import subprocess
import multiprocessing


# -------------Train Agent------------------
# Helper script to train multiple agents
# Calls the safepo training scripts
# Uses saved configs to ease reproducibility

# Define constants
SEEDS = ['0', '1', '2', '3', '4']

def run_script(algorithm, arguments):
    command = ['python', f'./Safe-Policy-Optimization/safepo/single_agent/{algorithm}.py'] + arguments
    result = subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, help="Choose from 'cpo', 'ppo_lag', 'cppo_pid', 'ppo_ewc', 'ppo_ewc_cost', 'clear")
    parser.add_argument('--task', type=str, help="Choose from 'cheetah' or 'cw'")
    parser.add_argument('--name', type=str, help="Experiment name to prepend results file")

    args = parser.parse_args()
    processes = []

    for seed in SEEDS:
        if args.task == 'cheetah':
            env_name = 'SafetyHalfCheetahVelocity-v4'
            task_list = '[0, 1, 0, 2, 1, 0, 2]'
            total_steps = '8_000_000'
        elif args.task == 'cw':
            env_name = 'SafetyContinualWorld'
            task_list = '[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]'
            total_steps = '15_000_000'
        else:
            raise Exception("Choose a task in the allowed task list. Run --help to see the full list.")

        arguments = ['--seed', seed,
                     '--task', env_name,
                     '--tasks', task_list,
                     '--total-steps', total_steps,
                     '--experiment', f'{args.name}_{seed}',
                     '--num-envs', '1',
                     '--device', 'cpu']
        
        algorithm = args.alg

        process = multiprocessing.Process(target=run_script, args=(algorithm, arguments))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
