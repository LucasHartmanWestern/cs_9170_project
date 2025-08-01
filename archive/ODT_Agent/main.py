"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import argparse
import pickle
import random
import time
import torch
import numpy as np
import wandb
from .critic import Q_Critic, Critic, V_Critic, VQ_Critic
import os
from .get_args import get_args
from tqdm import tqdm
from . import utils
from .replay_buffer import ReplayBuffer
from .lamb import Lamb
from .delayedreward import CumulativeRewardWrapper
from pathlib import Path
from .data import create_dataloader, delay_reward
from .decision_transformer.models.decision_transformer import DecisionTransformer
from .evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from .trainer import SequenceTrainer
from .logger import Logger

MAX_EPISODE_LEN = 1000
HAMMER_REWARD_SCALE_CONST, PEN_REWARD_SCALE_CONST, MUJOCO_REWARD_SCALE_CONST = 1000, 100, 1

class Experiment:
    def __init__(self, environment, seed, variant, device):
        
        self.seed = seed
        self.environment = environment
        self.USE_GRAD_CLIP, self.USE_LR_SCHEDULER = (variant['grad_clip'] == 1), (variant['lr_scheduler'] == 1)
        self.variant = variant  
        if variant['actor_rl_coeff'] < 1e-10: assert variant['stoc'] == 1 and variant['use_entropy_reg'] == 1, "Supervised learning must be standard!"
        self.env_name = variant['env']
        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)


        self.offline_trajs, self.state_mean, self.state_std, self.action_mean, self.action_std = self._load_dataset(
            variant["env"]
        )
        self.environment.action_mean = self.action_mean
        self.environment.action_std = self.action_std
        
        self.delayed_reward_flag = variant['delayed_reward']
        
        assert self.delayed_reward_flag != 1, "Error!"
        
        if self.delayed_reward_flag > 1:
            self.offline_trajs = delay_reward(self.offline_trajs, self.delayed_reward_flag)
        else:
            print("not delayed!")
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)
        
        if variant['rl_algo'] in ['AWR', 'PPO']:
            if variant['rl_algo'] != 'AWR':
                self.online_buffer = ReplayBuffer(99999999) # will be manually cleared
            else:
                self.online_buffer = ReplayBuffer(99999999, self.offline_trajs) # initialization with offline buffer
                returns = [self.online_buffer.trajectories[i]["rewards"].sum() for i in range(len(self.online_buffer.trajectories))]
                sorted_inds = np.argsort(returns)  # lowest to highest
                self.online_buffer.trajectories = [
                    self.online_buffer.trajectories[ii] for ii in sorted_inds
                ]

        self.aug_trajs = []

        if variant['stoc'] == 0: assert variant['use_entropy_reg'] == 0, "Error!"

        self.device = device
        self.target_entropy = -self.act_dim
        print(f"K: {variant['K']}")
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=variant["K"],
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=3 * variant['K'],
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=(variant['stoc'] == 1),
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        if self.USE_LR_SCHEDULER == 1:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
            )
        else: self.scheduler = None
        if variant['stoc'] == 1:
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4 * variant['temperature_learnable'],
                betas=[0.9, 0.999],
            )

        else:
            self.log_temperature_optimizer = torch.nn.ModuleList() #torch.nn.parameter.Parameter(torch.zeros(1)) 
        
        is_mujoco_flag = ((variant["env"].find("hopper-") != -1) or (variant["env"].find("ant-") != -1) or (variant["env"].find("walker2d-") != -1) or (variant["env"].find("halfcheetah-") != -1))
         
        if variant['rl_algo'] in ["TD3", "SAC", "AWAC"]:
            self.critic = Q_Critic(state_dim=self.state_dim, action_dim=self.act_dim, time_dim=variant['critic_time_dim'], time_aware=variant['critic_time_aware'], activation=variant['critic_activation'], normalization=variant['critic_normalization'], with_layernorm = ((variant["override_layernorm"] != -1) and ((is_mujoco_flag == False) or (variant["override_layernorm"] == 1)))).to(device=self.device)
        elif variant['rl_algo'] == 'IQL':
            self.critic = VQ_Critic(state_dim=self.state_dim, action_dim=self.act_dim, time_dim=variant['critic_time_dim'], time_aware=variant['critic_time_aware'], activation=variant['critic_activation'], normalization=variant['critic_normalization'], with_layernorm = ((variant["override_layernorm"] != -1) and ((is_mujoco_flag == False) or (variant["override_layernorm"] == 1)))).to(device=self.device)
        else:
            self.critic = V_Critic(state_dim=self.state_dim, action_dim=self.act_dim, time_dim=variant['critic_time_dim'], time_aware=variant['critic_time_aware'], activation=variant['critic_activation'], normalization=variant['critic_normalization'], with_layernorm = ((variant["override_layernorm"] != -1) and ((is_mujoco_flag == False) or (variant["override_layernorm"] == 1)))).to(device=self.device)
        
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=variant['critic_learning_rate'])

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        
        self.reward_scale = 1.0 if ("relocate" in variant['env'] or "pen" in variant['env'] or 'hammer' in variant['env'] or 'maze2d' in variant['env']) else 0.001 
        
        # warning: relocate and door might need to retest - reward scale modified from 0.001 to 1.0 !
        
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):
        env = self.environment
        state_dim = env.state_shape[0]
        act_dim = env.action_dim.shape[0]

        action_range = env.action_range
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }
        if self.USE_LR_SCHEDULER == 1:
            to_save["scheduler_state_dict"] = self.scheduler.state_dict()
        
        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.USE_LR_SCHEDULER == 1:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):
        dataset_path = f"/home/epigou/cs_9170_project/Dataset_Large2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        totrets, totshape = [], []
        
        for i in range(len(trajectories)):
            #print("REW-before:", trajectories[i]['rewards'].sum())
            #print("traj:", trajectories[i])
            if env_name.find("hammer-") != -1: trajectories[i]['rewards'] /= HAMMER_REWARD_SCALE_CONST 
            elif (env_name.find("pen-") != -1 and env_name.find("open") == -1) or env_name.find('relocate-') != -1 or env_name.find('door')!= -1: trajectories[i]['rewards'] /= PEN_REWARD_SCALE_CONST
            else: trajectories[i]['rewards'] /= MUJOCO_REWARD_SCALE_CONST
            #print("REW-after:", trajectories[i]['rewards'].sum(), trajectories[i]['rewards'].shape)
            totrets.append(trajectories[i]['rewards'].sum())
            totshape.append(trajectories[i]['rewards'].shape)
        
        totrets, totshape = np.array(totrets), np.array(totshape)
        print("totrets-max:", totrets.max(), "totrets-mean:", totrets.mean())
        
        def count_values_in_array(arr):
            # Find unique values and their counts
            values, counts = np.unique(arr, return_counts=True)
            
            # Combine the counts and values into a list of tuples
            count_value_pairs = list(zip(values, counts))
            
            # Sort the list of tuples by the value (the first element of each tuple)
            sorted_count_value_pairs = sorted(count_value_pairs, key=lambda x: x[0])
            
            # Print the sorted counts and values
            for value, count in sorted_count_value_pairs:
                print(f"Value: {value}, Count: {count}")
            
        count_values_in_array(totshape)
        #exit(0)
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        all_actions = np.concatenate([path["actions"] for path in trajectories], axis=0)
        action_mean = np.mean(all_actions, axis=0)
        action_std  = np.std(all_actions,  axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std, action_mean, action_std

    def determine_rl_params(self, variant):
        if variant['rl_algo'] == 'TD3':
            rl_params = {"algo": "TD3", "TD3_policy_noise": self.variant['TD3_policy_noise'], "TD3_tau": self.variant['TD3_tau'], "TD3_noise_clip": self.variant["TD3_noise_clip"]}
        elif variant['rl_algo'] == 'SAC': 
            assert variant['num_actor_update_interval'] == 1, "Error!"
            rl_params = {"algo": "SAC", "SAC_tau": self.variant['SAC_tau']} 
        elif variant["rl_algo"] == 'AWAC':
            assert variant['num_actor_update_interval'] == 1, "Error!"
            rl_params = {"algo": "AWAC", "AWAC_normalize_adv": self.variant['AWAC_normalize_adv'], "AWAC_tau":self.variant['AWAC_tau'], "AWAC_beta": self.variant['AWAC_beta'], 'AWAC_soft_flag': self.variant['AWAC_soft_flag']}
        elif variant["rl_algo"] == 'PPO':
            assert variant['num_actor_update_interval'] == 1, "Error!"
            rl_params = {"algo": "PPO", "PPO_eps_clip":self.variant['PPO_eps_clip'], "PPO_td_lambda": self.variant['PPO_td_lambda'], "PPO_old_logprob_generated_in_training": self.variant['PPO_old_logprob_generated_in_training']}
        elif variant["rl_algo"] == 'AWR':
            assert variant['num_actor_update_interval'] == 1, "Error!"
            rl_params = {"algo": "AWR", "AWR_beta": self.variant["AWR_beta"], "AWR_normalize_adv": self.variant['AWR_normalize_adv'], 'AWR_td_lambda': self.variant['AWR_td_lambda']}
        elif variant['rl_algo'] == 'IQL':
            assert variant['num_actor_update_interval'] == 1, "Error!"
            rl_params = {"algo": 'IQL', 'IQL_ratio': self.variant['IQL_ratio'], 'IQL_beta': self.variant['IQL_beta'], "IQL_tau": self.variant['IQL_tau']}
        
        rl_params["normalized_rl_coeff"], rl_params['gamma'] = self.variant['normalized_rl_coeff'], self.variant['gamma']
        
        return rl_params 


    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN
        t0 = time.time()  
        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale]

            print("target-return:", target_return)
            #exit(0)
            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                self.variant,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                noise_level=(self.variant['expl_noise'] if self.variant['stoc'] == 0 else 0),
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
            )
        t1 = time.time()
        self.replay_buffer.add_new_trajs(trajs)
        if self.variant['rl_algo'] in ["PPO", "AWR"]:
            self.online_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += lengths.sum().item()
        print("collect:", t1 - t0, "addtraj:", time.time() - t1, "total-return:", returns)
        return {
            "aug_traj/return": returns.mean().item(),
            "aug_traj/length": lengths.float().mean().item(),
        }



    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env=self.environment,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        
        rl_params = self.determine_rl_params(self.variant)

        trainer = SequenceTrainer(
            model=self.model,
            critic=self.critic,
            optimizer=self.optimizer,
            critic_optimizer=self.critic_optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            actor_update_interval=self.variant['num_actor_update_interval'],
            rl_params=rl_params,
            entropy_flag=self.variant["use_entropy_reg"],
            scheduler=(self.scheduler if self.variant['lr_scheduler'] else None),
            device=self.device,
            use_grad_clip_flag=self.USE_GRAD_CLIP,
            use_lr_scheduler_flag=self.USE_LR_SCHEDULER,
            variant=self.variant,
            state_mean = self.state_mean,
            state_std = self.state_std,
            action_range=self.action_range,
            reward_scale=self.reward_scale,
            state_dim=self.state_dim,
            action_dim=self.act_dim 
        )

        writer = None
        #(SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None)
        
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"] * self.variant['num_actor_update_interval'],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                action_mean=self.action_mean,
                action_std=self.action_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range
            )
            
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
                update_critic=True,
                rl_coeff=(0 if (self.variant['RL_from_start'] == 0) else self.variant['actor_rl_coeff']),
                sup_coeff=(1 if (self.variant['RL_from_start'] == 0) else self.variant['actor_sup_coeff']),#self.variant['actor_sup_coeff'],
                pretrain_flag=True 
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            
            outputs_wandb = outputs
            outputs_wandb["iter_num"], outputs_wandb["total_transitions_sampled"] = self.pretrain_iter, self.total_transitions_sampled,
            wandb.log(outputs_wandb)
            
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1

    def evaluate(self, eval_fns, video_debug=0, video_name=""):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        
        for eval_fn in eval_fns: 
            o = eval_fn(self.model, video_debug, video_name)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        if self.env_name.find('antmaze') != -1:
            outputs['evaluation/SR'] = np.sum(o['evaluation/length'] < (700 if (self.env_name.find("umaze") != -1) else 1000)) 

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        rl_params = self.determine_rl_params(self.variant)

        trainer = SequenceTrainer(
            model=self.model,
            critic=self.critic,
            optimizer=self.optimizer,
            critic_optimizer=self.critic_optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            actor_update_interval=self.variant['num_actor_update_interval'],
            rl_params=rl_params, 
            entropy_flag=self.variant["use_entropy_reg"],
            scheduler=(self.scheduler if self.variant['lr_scheduler'] else None),
            device=self.device,
            use_grad_clip_flag=self.USE_GRAD_CLIP,
            use_lr_scheduler_flag=self.USE_LR_SCHEDULER,
            variant=self.variant,
            state_mean = self.state_mean,
            state_std = self.state_std,
            action_range=self.action_range,
            reward_scale=self.reward_scale,
            state_dim=self.state_dim,
            action_dim=self.act_dim   
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                env=self.environment,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = None # (SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None)
        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            
            tot_len = 0
            t0 = time.time()
            while True:
            
                augment_outputs = self._augment_trajectories(
                    online_envs,
                    self.variant["online_rtg"],
                    n=self.variant["num_online_rollouts"],
                )
                outputs.update(augment_outputs)
            
                tot_len += augment_outputs['aug_traj/length']
                if tot_len >= self.variant['minimum_sapairs_per_iter']: break 

            t1 = time.time()

            print("replay_buffer_rew:", [self.replay_buffer.trajectories[i]['rewards'].sum() for i in range(len(self.replay_buffer.trajectories))])
            # exit(0)
            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"] * self.variant['num_actor_update_interval'],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                action_mean=self.action_mean,
                action_std=self.action_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False
 
            t2 = time.time()
             
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
                update_critic=True,
                rl_coeff=self.variant['actor_rl_coeff'],
                sup_coeff=self.variant['actor_sup_coeff'],
                online_buffer = (self.online_buffer if self.variant['rl_algo'] in ['PPO', 'AWR'] else None)
            )
            t25 = time.time()
            outputs.update(train_outputs)

            if self.variant['rl_algo'] == "PPO":
                self.online_buffer.clear()
            elif self.variant['rl_algo'] == "AWR":
                self.online_buffer.prune(self.variant['AWR_buffer_size'])
            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns, 0, self.EXP_NAME+"-iter"+str(self.online_iter))
                outputs.update(eval_outputs)

            t3 = time.time()

            outputs["time/total"] = time.time() - self.start_time

            outputs_wandb = outputs
            outputs_wandb["iter_num"], outputs_wandb["total_transitions_sampled"] = self.pretrain_iter, self.total_transitions_sampled,
            wandb.log(outputs_wandb)
            
            is_mujoco_flag = ((self.variant["env"].find("hopper-") != -1) or (self.variant["env"].find("ant-") != -1) or (self.variant["env"].find("walker2d-") != -1) or (self.variant["env"].find("halfcheetah-") != -1))
            
            if outputs_wandb['total_transitions_sampled'] >= (600000 if is_mujoco_flag else 1050000): exit(0)
            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )
            t4 = time.time()
            if self.online_iter % 10 == 0:
                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=False,
                )

            t5 = time.time()
            print("outerloop - collectdata:", t1 - t0, "construct dataloader:", t2 - t1, "train:", t25 - t2, "prune:", t3 - t25, "metric:", t4 - t3, "save:", t5 - t4)

            self.online_iter += 1

    def __call__(self, EXP_NAME):
        
        self.EXP_NAME = EXP_NAME
        
        utils.set_seed_everywhere(self.seed)

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_flag,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_prob(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_flag * entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )
            
        def mse_loss_fn(a_hat, a, attention_mask):
            return ((a_hat - a) ** 2)[attention_mask > 0].mean()

        print("\n\nMaking Eval Env.....")

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(self.environment, (loss_fn if self.variant['stoc'] == 1 else mse_loss_fn))

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            self.online_tuning(self.environment, self.environment, (loss_fn if self.variant['stoc'] == 1 else mse_loss_fn))


def call_odt(env):

    # args = get_args()
    # if "random" in args.env and args.force_no_minimum == 0:
    #     assert args.minimum_sapairs_per_iter > 0, \
    #         "You must set minimum_sapairs_per_iter > 0 in random envs!"
    # if args.force_no_minimum != 0:
    #     print("forced no minimum!")

    utils.set_seed_everywhere(env.seed)

    variant = {
        # ─── environment & data ──────────────────────────────────────────
        "env":                   "Synth-Data",
        "exp_name":              "Synth-Data",
        "custom_dataset":        0,
        "delayed_reward":        0,
        "remove_trivial_trajs":  1,
        "replay_size":           10,
        "save_dir":              "./Temp",

        # ─── model & architecture ────────────────────────────────────────
        "K":                     1001,
        "eval_context_length":   1001,
        "embed_dim":             32,
        "n_layer":               1,
        "n_head":                1,
        "dropout":               0.1,
        "activation_function":   "relu",
        "ordering":              "krt",
        "init_temperature":      1.0,

        # ─── optimization ────────────────────────────────────────────────
        "learning_rate":         3e-4,
        "weight_decay":          1e-4,
        "grad_clip":             1,
        "lr_scheduler":          1,
        "warmup_steps":          25,

        # ─── data-loader & loops ─────────────────────────────────────────
        "batch_size":            64,
        "num_updates_per_pretrain_iter": 1,
        "num_updates_per_online_iter":   1,
        "max_pretrain_iters":    0,
        "max_online_iters":      200,
        "num_online_rollouts":   1,
        "eval_interval":         20,

        # ─── supervised vs RL mix ───────────────────────────────────────
        # Pretrain: RL_from_start=0 → rl_coeff=0, sup_coeff=1 (pure supervised)
        # Online:  actor_rl_coeff=1.0, actor_sup_coeff=0.0 (pure RL)
        "actor_rl_coeff":        1.0,
        "actor_sup_coeff":       0.0,
        "normalized_rl_coeff":   1.0,
        "gamma":                 0.99,
        "RL_from_start":         1,

        # ─── RL algorithm choice ─────────────────────────────────────────
        "rl_algo":               "SAC",
        "num_actor_update_interval": 1,

        # ─── SAC params ─────────────────────────────────────────────────
        "SAC_tau":               0.005,

        # ─── TD3 params ─────────────────────────────────────────────────
        "TD3_policy_noise":      0.2,
        "TD3_noise_clip":        0.5,
        "TD3_tau":               0.005,

        # ─── AWAC params ────────────────────────────────────────────────
        "AWAC_beta":             0.1,
        "AWAC_normalize_adv":    1,
        "AWAC_tau":              0.005,
        "AWAC_soft_flag":        1,

        # ─── PPO params ─────────────────────────────────────────────────
        "PPO_eps_clip":                  0.2,
        "PPO_td_lambda":                 0.95,
        "PPO_old_logprob_generated_in_training": 0,

        # ─── AWR params ─────────────────────────────────────────────────
        "AWR_td_lambda":         0.95,

        # ─── IQL params ─────────────────────────────────────────────────
        "IQL_ratio":             0.5,
        "IQL_beta":              1.0,
        "IQL_tau":               0.7,

        # ─── evaluation returns ──────────────────────────────────────────
        "online_rtg":            1.0,
        "eval_rtg":              0.0,
        "minimum_sapairs_per_iter": 5,

        # ─── critic settings ────────────────────────────────────────────
        "critic_time_dim":       1,
        "critic_time_aware":     0,
        "critic_activation":     "relu",
        "critic_normalization":  "layernorm",
        "override_layernorm":    -1,
        "critic_learning_rate":  1e-3,

        # ─── stochastic policy flags ────────────────────────────────────
        "stoc":                  1,    # need a stochastic policy for SAC
        "use_entropy_reg":       0,    # SAC has its own entropy term
        "temperature_learnable": 1,

        # ─── logging & misc ─────────────────────────────────────────────
        "log_to_tb":             1,
        "force_no_minimum":      0,
    }

    variant['K'] = env.horizon + 1



    experiment = Experiment(environment=env, seed=env.seed, variant=variant, device=env.device)

    # suffix_alg = "-none" if args.actor_rl_coeff < 1e-10 else args.rl_algo
    # suffix_det = "-stoc" if args.stoc == 1 else "-det"
    EXP_NAME = "Synth-Data"
    wandb.init(
        project="odt-TD3branch",    # pick any project name you like
        name=EXP_NAME,  
    )

    print("=" * 50)
    # 7) Run pretrain + online finetuning
    experiment(EXP_NAME)

    return env, experiment
