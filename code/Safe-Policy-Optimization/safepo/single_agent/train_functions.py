from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from typing import Callable

import numpy as np
try: 
    from isaacgym import gymutil
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from torch.optim.lr_scheduler import LinearLR
from safepo.common.lagrange import PIDLagrangian as Lagrange
from safepo.common.lagrange import Lagrange as LagLagrange


from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params
device = "cpu"

def train_cpo(hyperparams, task):
    if 'step_fraction' in hyperparams:
        STEP_FRACTION=hyperparams['step_fraction']
    else:
        STEP_FRACTION=0.8
    
    CPO_SEARCHING_STEPS=15
    CONJUGATE_GRADIENT_ITERS=15

    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    default_cfg = {
        'hidden_sizes': [64, 64],
        'gamma': 0.99,
        'target_kl': 0.01,
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
    }

    if len(hyperparams.keys()) == 0:
        hyperparams = default_cfg


    def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
        flat_params = []
        for _, param in model.named_parameters():
            if param.requires_grad:
                data = param.data
                data = data.view(-1)  # flatten tensor
                flat_params.append(data)
        assert flat_params, "No gradients were found in model parameters."
        return torch.cat(flat_params)


    def conjugate_gradients(
        fisher_product: Callable[[torch.Tensor], torch.Tensor],
        policy: ActorVCritic,
        fvp_obs: torch.Tensor,
        vector_b: torch.Tensor,
        num_steps: int = 10,
        residual_tol: float = 1e-10,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        vector_x = torch.zeros_like(vector_b)
        vector_r = vector_b - fisher_product(vector_x, policy, fvp_obs)
        vector_p = vector_r.clone()
        rdotr = torch.dot(vector_r, vector_r)

        for _ in range(num_steps):
            vector_z = fisher_product(vector_p, policy, fvp_obs)
            alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
            vector_x += alpha * vector_p
            vector_r -= alpha * vector_z
            new_rdotr = torch.dot(vector_r, vector_r)
            if torch.sqrt(new_rdotr) < residual_tol:
                break
            vector_mu = new_rdotr / (rdotr + eps)
            vector_p = vector_r + vector_mu * vector_p
            rdotr = new_rdotr
        return vector_x


    def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
        assert isinstance(vals, torch.Tensor)
        i: int = 0
        for _, param in model.named_parameters():
            if param.requires_grad:  # param has grad and, hence, must be set
                orig_size = param.size()
                size = np.prod(list(param.size()))
                new_values = vals[i : int(i + size)]
                # set new param values
                new_values = new_values.view(orig_size)
                param.data = new_values
                i += int(size)  # increment array position
        assert i == len(vals), f"Lengths do not match: {i} vs. {len(vals)}"

    def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
        grads = []
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                grads.append(grad.view(-1))  # flatten tensor and append
        assert grads, "No gradients were found in model parameters."
        return torch.cat(grads)

    def fvp(
        params: torch.Tensor,
        policy: ActorVCritic,
        fvp_obs: torch.Tensor,
    ) -> torch.Tensor:
        policy.actor.zero_grad()
        current_distribution = policy.actor(fvp_obs)
        with torch.no_grad():
            old_distribution = policy.actor(fvp_obs)
        kl = torch.distributions.kl.kl_divergence(
            old_distribution, current_distribution
        ).mean()

        grads = torch.autograd.grad(kl, tuple(policy.actor.parameters()), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * params).sum()
        grads = torch.autograd.grad(
            kl_p,
            tuple(policy.actor.parameters()),
            retain_graph=False,
        )

        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return flat_grad_grad_kl + params * 0.1

    args = {
        'seed': 0,
        'task': task
    }

    # set the random seed, device and number of threads
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=1, env_id=task, seed=args['seed']
    )
    config = hyperparams

    # set training steps
    steps_per_epoch = 20_000
    total_steps = 500_000
    local_steps_per_epoch = 20_000
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=1e-3
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=1e-3
    )

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=1,
        gamma=config["gamma"],
    )

    if task=='SafetyContinualWorld':
        env.current_task = 2
        env.change_task()
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len, ep_success = (
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if args['task'] in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args['task'] in isaac_gym_map.keys() else reward
            if 'success' in info and int(info['success']) == 1 and terminated:
                ep_success += 1
            ep_cost += cost.cpu().numpy() if args['task'] in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        ep_ret[idx] = 0.0
                        prev_ep_cost = ep_cost[idx]
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        ep_success[idx] = 0.0
    
                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        eval_start_time = time.time()

        # update policy
        data = buffer.get()
        fvp_obs = data["obs"][:: 1]
        theta_old = get_flat_params_from(policy.actor)
        policy.actor.zero_grad()
        # compute loss_pi
        temp_distribution = policy.actor(data["obs"])
        log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
        ratio = torch.exp(log_prob - data["log_prob"])
        loss_pi_r = -(ratio * data["adv_r"]).mean()
        loss_reward_before = loss_pi_r.item()
        old_distribution = policy.actor(data["obs"])

        loss_pi_r.backward()

        grads = -get_flat_gradients_from(policy.actor)
        x = conjugate_gradients(fvp, policy, fvp_obs, grads, CONJUGATE_GRADIENT_ITERS)
        assert torch.isfinite(x).all(), "x is not finite"
        xHx = torch.dot(x, fvp(x, policy, fvp_obs))
        assert xHx.item() >= 0, "xHx is negative"
        alpha = torch.sqrt(2 * config['target_kl'] / (xHx + 1e-8))

        policy.actor.zero_grad()
        temp_distribution = policy.actor(data["obs"])
        log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
        ratio = torch.exp(log_prob - data["log_prob"])
        loss_pi_c = (ratio * data["adv_c"]).mean()
        loss_cost_before = loss_pi_c.item()

        loss_pi_c.backward()

        b_grads = get_flat_gradients_from(policy.actor)
        ep_costs = prev_ep_cost - 25

        p = conjugate_gradients(fvp, policy, fvp_obs, b_grads, CONJUGATE_GRADIENT_ITERS)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all(), "r is not finite"
            assert torch.isfinite(s).all(), "s is not finite"

            A = q - r**2 / (s + 1e-8)
            B = 2 * config['target_kl'] - ep_costs**2 / (s + 1e-8)

            if ep_costs < 0 and B < 0:
                optim_case = 3
            elif ep_costs < 0 <= B:
                optim_case = 2
            elif ep_costs >= 0 and B >= 0:
                optim_case = 1
            else:
                optim_case = 0

        if optim_case in (3, 4):
            alpha = torch.sqrt(2 * config['target_kl'] / (xHx + 1e-8))
            nu_star = torch.zeros(1)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x

        elif optim_case in (1, 2):

            def project(
                data: torch.Tensor, low: torch.Tensor, high: torch.Tensor
            ) -> torch.Tensor:
                """Project data to [low, high] interval."""
                return torch.clamp(data, low, high)

            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * config['target_kl']))
            r_num = r.item()
            eps_cost = ep_costs + 1e-8
            if ep_costs < 0:
                lambda_a_star = project(
                    lambda_a, torch.as_tensor(0.0), r_num / eps_cost
                )
                lambda_b_star = project(
                    lambda_b, r_num / eps_cost, torch.as_tensor(torch.inf)
                )
            else:
                lambda_a_star = project(
                    lambda_a, r_num / eps_cost, torch.as_tensor(torch.inf)
                )
                lambda_b_star = project(
                    lambda_b, torch.as_tensor(0.0), r_num / eps_cost
                )

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * config['target_kl'] * lam)

            lambda_star = (
                lambda_a_star
                if f_a(lambda_a_star) >= f_b(lambda_b_star)
                else lambda_b_star
            )

            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)

            step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

        else:
            lambda_star = torch.zeros(1)
            nu_star = torch.sqrt(2 * config['target_kl'] / (s + 1e-8))
            step_direction = -nu_star * p

        step_frac = 1.0
        theta_old = get_flat_params_from(policy.actor)
        expected_reward_improve = grads.dot(step_direction)

        kl = torch.zeros(1)
        for step in range(CPO_SEARCHING_STEPS):
            new_theta = theta_old + step_frac * step_direction
            set_param_values_to_model(policy.actor, new_theta)
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    temp_distribution = policy.actor(data["obs"])
                    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                    ratio = torch.exp(log_prob - data["log_prob"])
                    loss_reward = -(ratio * data["adv_r"]).mean()
                except ValueError:
                    step_frac *= STEP_FRACTION
                    continue
                temp_distribution = policy.actor(data["obs"])
                log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                ratio = torch.exp(log_prob - data["log_prob"])
                loss_cost = (ratio * data["adv_c"]).mean()
                current_distribution = policy.actor(data["obs"])
                kl = torch.distributions.kl.kl_divergence(
                    old_distribution, current_distribution
                ).mean()
            loss_reward_improve = loss_reward_before - loss_reward.item()
            loss_cost_diff = loss_cost.item() - loss_cost_before

            step_frac *= STEP_FRACTION
        else:
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        theta_new = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_new)

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["target_value_r"],
                data["target_value_c"],
            ),
            batch_size=config.get("batch_size", 20_000//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                target_value_r_b,
                target_value_c_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                total_loss = 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()

    return policy

def train_cppo(hyperparams, task):
    
    default_cfg = {
        'hidden_sizes': [64, 64],
        'gamma': 0.99,
        'target_kl': 0.02,
        'batch_size': 64,
        'learning_iters': 40,
        'max_grad_norm': 40.0,
    }

    if hyperparams is None:
        hyperparams = default_cfg

    args = {
        'seed': 0,
        'task': task,
        'num_envs': 1
    }

    # set the random seed, device and number of threads
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=1, env_id=task, seed=args['seed']
    )
    config = hyperparams

    # set training steps
    steps_per_epoch = 20_000
    total_steps = 500_000
    local_steps_per_epoch = 20_000
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4
    )

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=1,
        gamma=config["gamma"],
    )
    # setup lagrangian multiplier
    lagrange = Lagrange(
        cost_limit=25,
        lagrangian_multiplier_init=0.001, # should this be a hyperparameter?
    )

    if task=='SafetyContinualWorld':
        env.current_task = 2
        env.change_task()
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len, ep_success = (
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if args['task'] in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args['task'] in isaac_gym_map.keys() else reward
            if 'success' in info and int(info['success']) == 1 and terminated:
                ep_success += 1
            ep_cost += cost.cpu().numpy() if args['task'] in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        ep_ret[idx] = 0.0
                        prev_ep_cost = ep_cost[idx]
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        ep_success[idx] = 0.0

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        eval_start_time = time.time()

        # update lagrange multiplier
        ep_costs = prev_ep_cost
        lagrange.update_lagrange_multiplier(ep_costs)

        # update policy
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])

        # comnpute advantage
        advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
        advantage /= (lagrange.lagrangian_multiplier + 1)

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
            ),
            batch_size=config.get("batch_size", 20_000//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                actor_optimizer.zero_grad()
                total_loss = loss_pi + 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()


            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()

    return policy

def train_ppo_lag(hyperparams, task):
    default_cfg = {
        'hidden_sizes': [64, 64],
        'gamma': 0.99,
        'target_kl': 0.02,
        'batch_size': 64,
        'learning_iters': 40,
        'max_grad_norm': 40.0,
    }

    args = {
        'seed': 0,
        'task': task,
    }

    # set the random seed, device and number of threads
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=1, env_id=args['task'], seed=args['seed']
    )
    eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args['task'], seed=None)
    config = hyperparams

    # set training steps
    steps_per_epoch = 20_000
    total_steps = 500_000
    local_steps_per_epoch = steps_per_epoch
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4
    )

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=1,
        gamma=config["gamma"],
    )
    # setup lagrangian multiplier
    lagrange = LagLagrange(
        cost_limit=25,
        lagrangian_multiplier_init=0.001, # should this be a hyperparameter?
        lagrangian_multiplier_lr=config['lagrangian_multiplier_lr'],
    )

    if task=='SafetyContinualWorld':
        env.current_task = 2
        env.change_task()
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len, ep_success = (
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if args['task'] in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args['task'] in isaac_gym_map.keys() else reward
            if 'success' in info and int(info['success']) == 1 and terminated:
                ep_success += 1
            ep_cost += cost.cpu().numpy() if args['task'] in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        ep_ret[idx] = 0.0
                        prev_ep_cost = ep_cost[idx]
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        ep_success[idx] = 0.0

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        eval_start_time = time.time()

        # update lagrange multiplier
        ep_costs = prev_ep_cost
        lagrange.update_lagrange_multiplier(ep_costs)

        # update policy
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])

        # comnpute advantage
        advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
        advantage /= (lagrange.lagrangian_multiplier + 1)

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
            ),
            batch_size=config.get("batch_size", 20_000//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                actor_optimizer.zero_grad()
                total_loss = loss_pi + 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()


            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()

    return policy

def train_ppo(hyperparams, task):
    default_cfg = {
        'hidden_sizes': [64, 64],
        'gamma': 0.99,
        'target_kl': 0.02,
        'batch_size': 64,
        'learning_iters': 40,
        'max_grad_norm': 40.0,
    }

    args = {
        'seed': 0,
        'task': task
    }

    # set the random seed, device and number of threads
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=1, env_id=args['task'], seed=args['seed']
    )
    eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args['task'], seed=None)
    config = hyperparams

    # set training steps
    steps_per_epoch = 20_000
    total_steps = 500_000
    local_steps_per_epoch = steps_per_epoch 
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4
    )

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=1,
        gamma=config["gamma"],
    )

    if task=='SafetyContinualWorld':
        env.current_task = 2
        env.change_task()
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len, ep_success = (
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if args['task'] in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args['task'] in isaac_gym_map.keys() else reward
            if 'success' in info and int(info['success']) == 1 and terminated:
                ep_success += 1
            ep_cost += cost.cpu().numpy() if args['task'] in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        ep_ret[idx] = 0.0
                        prev_ep_cost = ep_cost[idx]
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        ep_success[idx] = 0.0

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        # update lagrange multiplier
        ep_costs = prev_ep_cost

        # update policy
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])

        # comnpute advantage
        advantage = data["adv_r"]

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
            ),
            batch_size=config.get("batch_size", 20_000//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                actor_optimizer.zero_grad()
                total_loss = loss_pi + 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()

            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()

    return policy

def train_ppo_ewc(hyperparams, task):
    def compute_fisher_info(data, policy):
        fisher_info = {}
        for obs, act in zip(data['obs'], data['act']):
            policy.zero_grad()
            log_prob = policy.actor(obs).log_prob(act).sum()
            log_prob.backward()
            for name, param in policy.actor.named_parameters():
                if name not in fisher_info:
                    # print(f'param grad requires {param.grad.requires_grad}, is detach {param.grad is None}, grad {param.grad}')
                    fisher_info[name] = param.grad.clone().pow(2)
                    # print(f'param grad clone requires {fisher_info[name].requires_grad}, is detach {fisher_info[name] is None}, grad {fisher_info[name].grad}')
                else:
                    fisher_info[name] += param.grad.clone().pow(2)

        for name in fisher_info:
            fisher_info[name] /= len(data['obs'])

        return fisher_info

    def save_old_params(model):
        return {n: p.clone() for n, p in model.named_parameters()}
        
    default_cfg = {
        'hidden_sizes': [64, 64],
        'gamma': 0.99,
        'target_kl': 0.02,
        'batch_size': 64,
        'learning_iters': 40,
        'max_grad_norm': 40.0,
    }

    args = {
        'seed': 0,
    }
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    if task == 'SafetyHalfCheetahVelocity-v4':
        tasks = [0, 1]
    elif task == 'SafetyContinualWorld':
        tasks = [0, 1]

    num_tasks = len(tasks)
    # EWC over multiple tasks
    fisher_matrices = [None] * num_tasks
    old_params_list = [None] * num_tasks
    lambda_ewc = hyperparams['lambda']

    current_task_index = 0
    steps_since_change = 0

    env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=1, env_id=task, seed=args['seed']
    )
    env.TASK_LENGTH = 500_000
    config = hyperparams

    # set training steps
    steps_per_epoch = 20_000
    total_steps = 1_000_000
    local_steps_per_epoch = steps_per_epoch 
    epochs = total_steps // steps_per_epoch
    steps_per_task = 500_000

    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4
    )

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=1,
        gamma=config["gamma"],
    )

    if task=='SafetyContinualWorld':
        env.current_task = 2
        env.change_task()
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len, ep_success = (
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if task in isaac_gym_map.keys() else reward
            if 'success' in info and int(info['success']) == 1 and terminated:
                ep_success += 1
            ep_cost += cost.cpu().numpy() if task in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        ep_ret[idx] = 0.0
                        prev_ep_cost = ep_cost[idx]
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        ep_success[idx] = 0.0

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        # update lagrange multiplier
        ep_costs = prev_ep_cost

        # update policy
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])

        # comnpute advantage
        advantage = data["adv_r"]

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
            ),
            batch_size=config.get("batch_size", 20_000//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                actor_optimizer.zero_grad()

                ewc_loss = 0.

                for i, (fisher_info, old_params) in enumerate(zip(fisher_matrices, old_params_list)):
                    if fisher_info is not None:
                        for name, param in policy.actor.named_parameters():
                            fisher = fisher_info[name]
                            ewc_loss = ewc_loss + (fisher * (param - old_params[name]).pow(2)).sum() 

                total_loss = loss_pi + 2*loss_r + loss_c + ewc_loss * lambda_ewc\
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c + ewc_loss * lambda_ewc

                total_loss.backward()

                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()

            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()

        steps_since_change += steps_per_epoch

        # At the end of each epoch, check if the task has changed
        if steps_since_change >= steps_per_task:
            current_task_num = tasks[current_task_index]
            current_fisher = compute_fisher_info(buffer.get(), policy)
            old_params = save_old_params(policy.actor)
            
            # Note: the fisher matrix for the previous task is overwritten 
            # This may make it highly dependent on the buffer of data
            fisher_matrices[current_task_num] = current_fisher
            old_params_list[current_task_num] = old_params

            current_task_index = (current_task_index + 1) % num_tasks
            steps_since_change = 0
        
    return policy
