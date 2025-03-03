# Safe Continual Reinforcement Learning

Code for CoLLAs 2025 submission.

## Repository Structure

- [./code/](./code/) contains all code for the project
- [./code/Metaworld](./code/Metaworld/) contains a modified version of the [Meta World](https://github.com/Farama-Foundation/Metaworld) (Continual World) environment
- [./code/Metaworld/metaworld/envs/mujoco/sawyer_xyz/safe_cw/](./code/Metaworld/metaworld/envs/mujoco/sawyer_xyz/safe_cw/) contains the custom Safe Continual World environments
- [./code/Safe-Policy-Optimization/](./code/Safe-Policy-Optimization/) contains the safe RL algorithms, impelemented by [safepo](https://github.com/PKU-Alignment/Safe-Policy-Optimization)
- [./code/Safe-Policy-Optimization/safepo/single_agent/](./code/Safe-Policy-Optimization/safepo/single_agent/) contains all the algorithms benchmarked in this study
- [./code/safety-gymnasium/](./code/safety-gymnasium/) is a safe RL gymnasium library from [this repo](https://github.com/PKU-Alignment/safety-gymnasium)
- [./code/safety-gymnasium/safety-gymnasium/tasks/](./code/safety-gymnasium/safety_gymnasium/tasks) contains the [Safe Continual World](./code/safety-gymnasium/safety_gymnasium/tasks/safe_continual_world/safety_continual_world.py) and [Damaged HalfCheetah Velocity](./code/safety-gymnasium/safety_gymnasium/tasks/safe_velocity/safety_half_cheetah_velocity_v4.py) environments, used by the algorithms above
- [./code/Analyze Results.ipynb](./code/Analyze%20Results.ipynb) contains analysis code for the Damaged HalfCheetah Velocity task. All results from this paper can be produced in this notebook
- [./code/Analyze Results CW.ipynb](./code/Analyze%20Results%20CW.ipynb) contains analysis code for the Safe Continual World task. All results from this paper can be produced in this notebook
- [./code/opt_hyper_continual.py](./code/opt_hyper_continual.py) code to optimize hyperparameters for continual RL algorithms
- [./code/opt_hyper_rl.py](./code/opt_hyper_rl.py) code to optimize hyperparameters for regular RL algorithms
- [./code/opt_hyper_safe.py](./code/opt_hyper_safe.py) code to optimize hyperparameters for safe RL algorithms
- [./code/train_agent.py](./code/train_agent.py) trains an agent on 5 random seeds
- [./code/verify_cost_stability.py](./code/verify_cost_stability.py) simple experiment to verify the mug will not tip when the agent takes no action

## Installation

1. **[Recommended, but not required]** Create a conda env with python version 3.10 (`conda create -n safe_continual python=3.10.15`) and activate it (`conda activate safe_continual`). Install pip in that environment using `conda install pip`.
2. Enter the Metaworld directory (`cd code/Metaworld`) and `python -m pip install -e .`
3. Enter the Safe-Policy-Optimization directory (`cd code/Safe-Policy-Optimization`) and `python -m pip install -e .`
3. Enter the safety-gymnasium directory (`cd code/safety-gymnasium`) and `python -m pip install -e .`
4. Install optuna `python -m pip install optuna`
5. Install memory_profiler `python -m pip install memory_profiler`

## Reproducing Results

Run `python train_agent.py --alg {ppo, cpo, ppo_lag, cppo_pid, ppo_ewc, ppo_ewc_cost, clear} --env {cheetah or cw} --name {Experiment name}`. The results will be saved to `./runs` and the relevant CSV will be called `progress.csv`. However, since these results can take over a week total for all algorithms, depending on your computational resources, we recommend to use the saved results in the `./code/results/` folder instead. The notebook files in the `./code/` directory uses these for analysis. 

To repeat or extend the hyperparameter experiments, run the relevant hyperparameter optimization scripts. Use the `--help` flag to get the arguments for those files. The hyperparameters will be saved in `./code/hyperparams/`. 