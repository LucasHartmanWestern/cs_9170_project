"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
    env,
    eval_rtg,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    device,
    use_mean=False,
    reward_scale=0.001,
):
    def eval_episodes_fn(model, video_debug=0, name=None):
        target_return = [eval_rtg * reward_scale]
        ret = vec_evaluate_episode_rtg(
            env,
            state_dim,
            act_dim,
            model,
            online_data_mode='eval',
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
            video_debug=video_debug
        )
        
        returns, lengths, trajs = ret
        
        suffix = "_gm" if use_mean else ""
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.float32, device=device)

        return {
            f"evaluation/return_mean{suffix}": returns_t.mean().item(),
            f"evaluation/return_std{suffix}":  returns_t.std(unbiased=False).item(),
            f"evaluation/length_mean{suffix}": lengths_t.mean().item(),
            f"evaluation/length_std{suffix}":  lengths_t.std(unbiased=False).item(),
            f"evaluation/return": returns,
            f"evaluation/length": lengths
        }

    return eval_episodes_fn



def vec_evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    online_data_mode,
    target_return: list,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    use_mean=False,
    noise_level=0,
    video_debug=0
):
    assert len(target_return) == 1
    t0 = time.time()
    if online_data_mode == "eval": model.eval()
    else: model.train()
    model.to(device=device)
    
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = 1
    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = state.view(num_envs, -1, state_dim)
    
    next_states = torch.zeros(0, device=device, dtype=torch.float32)
    if model.stochastic_policy:
        action_log_probs = []
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
    episode_length = torch.full((num_envs,), max_ep_len, dtype=torch.long, device=device)  
    unfinished = torch.ones((num_envs,), dtype=torch.bool, device=device)

    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )
        with torch.no_grad():
            state_pred, action_dist, reward_pred = model.get_predictions(
                (states - state_mean) / state_std,
                actions,
                rewards,
                target_return,
                timesteps,
                num_envs=num_envs,
            )
        state_pred  = state_pred.detach().view(num_envs, -1)
        reward_pred = reward_pred.detach().view(num_envs)

        # sample or pick the action tensor as before…
        if model.stochastic_policy:
            if use_mean:
                action = action_dist.mean.view(num_envs, -1, act_dim)
            else:
                action = action_dist.sample().view(num_envs, -1, act_dim)

            # log‐probs for the last step
            action_log_probs.append(
                action_dist.log_prob(action.clamp(*model.action_range))[:, -1]
            )

            # take only the newest action
            action = action[:, -1]
        else:
            action = action_dist[:, -1] + noise_level * torch.randn_like(action_dist[:, -1])


        action = action.clamp(*model.action_range)

        # step the env; returns tensors now
        state, reward, done, info = env.step(action)

        # accumulate return for unfinished envs
        episode_return[unfinished] += reward[unfinished].view(-1, 1)

        # write the action into history
        actions[:, -1] = action

        # choose terminal vs next observation
        next_state = info.get("terminal_observation", state)

        # append the new state into states history
        state = state.view(num_envs, -1, state_dim)
        states = torch.cat([states, state], dim=1)

        
        # append the chosen next state
        next_state = next_state.to(device=device, dtype=torch.float32) \
                               .view(num_envs, -1, state_dim)
        next_states = torch.cat([next_states, next_state], dim=1)

        # insert the new reward
        reward = reward.to(device=device, dtype=torch.float32).view(num_envs, 1)
        rewards[:, -1] = reward

        # update the predicted return history
        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.view(num_envs, -1, 1)],
            dim=1,
        )

        # bump the timestep counter
        timesteps = torch.cat([
            timesteps,
            torch.full((num_envs, 1), t+1, dtype=torch.long, device=device),
        ], dim=1)

        # if we’re at the very last step, force done=True
        if t == max_ep_len - 1:
            done = torch.ones_like(unfinished)

        # update which envs just finished
        done = done.to(device=device).bool()
        newly_done = done & unfinished

        # record the length for those that just finished
        episode_length = torch.where(
            newly_done,
            torch.full_like(episode_length, t+1),
            episode_length
        )

        # mark them as finished
        unfinished = unfinished & (~done)

        if not unfinished.any():
            break

    t1 = time.time()
    if model.stochastic_policy:
        action_log_probs = torch.vstack(action_log_probs).T  # still on device

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].item()

        # terminals stays on device
        terminals = torch.zeros(ep_len, dtype=torch.uint8, device=device)
        terminals[-1] = 1

        traj = {
            "next_observations": next_states[ii, :ep_len].detach(),
            "observations":      states[ii, :ep_len].detach(),
            "actions":           actions[ii, :ep_len].detach(),
            "rewards":           rewards[ii, :ep_len].detach(),
            "terminals":         terminals,
        }

        if model.stochastic_policy:
            traj["action_log_probs"] = action_log_probs[ii, :ep_len].detach()

        trajectories.append(traj)

    t2 = time.time()
    print("collecttraj:", t1 - t0, "deal:", t2 - t1)

    return (
        episode_return.view(num_envs),
        episode_length.view(num_envs),
        trajectories,
    )


