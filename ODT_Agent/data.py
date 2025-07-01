"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import torch
import numpy as np
import random
from tqdm import tqdm

MAX_EPISODE_LEN = 1000

def delay_reward(trajectories, num):
    for i in tqdm(range(len(trajectories))):
        N = trajectories[i]['observations'].shape[0]        
        total_reward = 0
        
        for j in range(N):
            total_reward += trajectories[i]['rewards'][j]
            trajectories[i]['rewards'][j] = 0
            if j % num == num-1 or j == N-1:
                trajectories[i]['rewards'][j] = total_reward
                total_reward = 0
    return trajectories



class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
        transform=None,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        if self.transform:
            return self.transform(traj)
        else:
            return traj

    def __len__(self):
        return len(self.sampling_ind)

class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        action_mean,
        action_std,
        reward_scale,
        action_range,
        logprob_flag=False,
        V_flag=False,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        # store normalization statistics as numpy arrays for easy arithmetic
        self.state_mean = np.array(state_mean, dtype=np.float32)
        self.state_std  = np.array(state_std,  dtype=np.float32)
        self.action_mean = np.array(action_mean, dtype=np.float32)
        self.action_std  = np.array(action_std,  dtype=np.float32)
        self.reward_scale = reward_scale
        self.logprob_flag = logprob_flag
        self.V_flag = V_flag
        self.action_range = action_range

    def __call__(self, traj):
        # 1) sample a random starting index
        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # optional fields
        if self.logprob_flag:
            lplp = traj['action_log_probs'][si:si + self.max_len].reshape(-1, 1)
        if self.V_flag:
            V  = traj['old_V'][si:si + self.max_len].reshape(-1, 1)
            VV = traj['target'][si:si + self.max_len].reshape(-1, 1)

        # 2) extract raw slices
        ss_raw   = traj["observations"][si:si + self.max_len].reshape(-1, self.state_dim)
        nsns_raw = traj["next_observations"][si:si + self.max_len].reshape(-1, self.state_dim)
        aa_raw   = traj["actions"][si:si + self.max_len].reshape(-1, self.act_dim)
        rr_raw   = traj["rewards"][si:si + self.max_len].reshape(-1, 1)
        dd_raw   = traj.get("terminals", traj.get("dones"))[si:si + self.max_len]

        # actual length
        tlen = ss_raw.shape[0]

        # 3) padding: prepend zeros up to max_len
        pad_states = np.zeros((self.max_len - tlen, self.state_dim), dtype=np.float32)
        pad_actions= np.zeros((self.max_len - tlen, self.act_dim),   dtype=np.float32)
        pad_rewards= np.zeros((self.max_len - tlen, 1),              dtype=np.float32)
        pad_dones  = np.ones((self.max_len - tlen,),                 dtype=np.int64) * 2

        ss   = np.concatenate([pad_states, ss_raw],   axis=0)
        nsns = np.concatenate([pad_states, nsns_raw], axis=0)
        aa   = np.concatenate([pad_actions, aa_raw],  axis=0)
        rr   = np.concatenate([pad_rewards, rr_raw],  axis=0) * self.reward_scale
        dd   = np.concatenate([pad_dones, dd_raw],    axis=0)

        # 4) normalize states and actions
        ss   = (ss   - self.state_mean)  / self.state_std
        nsns = (nsns - self.state_mean)  / self.state_std
        aa   = (aa   - self.action_mean) / self.action_std

        # 5) compute return-to-go
        rtg_raw = discount_cumsum(traj["rewards"][si:], gamma=1.0)[: tlen + 1].reshape(-1, 1)
        if rtg_raw.shape[0] <= tlen:
            rtg_raw = np.concatenate([rtg_raw, np.zeros((1, 1), dtype=np.float32)], axis=0)
        pad_rtg = np.zeros((self.max_len - tlen, 1), dtype=np.float32)
        rtg = np.concatenate([pad_rtg, rtg_raw], axis=0) * self.reward_scale

        # 6) timesteps, ordering, and mask
        timesteps = np.arange(si, si + tlen)
        timesteps = np.concatenate([np.zeros(self.max_len - tlen, dtype=np.int64),
                                    np.clip(timesteps, 0, MAX_EPISODE_LEN - 1)], axis=0)
        ordering  = np.arange(tlen)
        ordering  = np.concatenate([np.zeros(self.max_len - tlen, dtype=np.int64), ordering], axis=0)
        padding_mask = np.concatenate([np.zeros(self.max_len - tlen, dtype=np.int64),
                                       np.ones(tlen, dtype=np.int64)], axis=0)

        # 7) convert to torch tensors
        ss   = torch.from_numpy(ss).float()
        aa   = torch.from_numpy(aa).float().clamp(*self.action_range)
        nsns = torch.from_numpy(nsns).float()
        rr   = torch.from_numpy(rr).float()
        dd   = torch.from_numpy(dd).long()
        rtg  = torch.from_numpy(rtg).float()
        timesteps   = torch.from_numpy(timesteps).long()
        ordering    = torch.from_numpy(ordering).long()
        padding_mask= torch.from_numpy(padding_mask).long()

        ret = [ss, aa, nsns, rr, dd, rtg, timesteps, ordering, padding_mask]

        if self.logprob_flag:
            lplp = np.concatenate([np.zeros((self.max_len - tlen, 1), dtype=np.float32), lplp], axis=0)
            ret.append(torch.from_numpy(lplp).float())
        if self.V_flag:
            V  = np.concatenate([np.zeros((self.max_len - tlen, 1), dtype=np.float32), V], axis=0)
            VV = np.concatenate([np.zeros((self.max_len - tlen, 1), dtype=np.float32), VV], axis=0)
            ret.append(torch.from_numpy(V).float())
            ret.append(torch.from_numpy(VV).float())

        return ret



def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    action_mean,
    action_std,
    reward_scale,
    action_range,
    num_workers=4,
    logprob_flag=False,
    V_flag=False
):
    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters
    sampling_ind = sample_trajs(trajectories, sample_size) 
    
    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        reward_scale=reward_scale,
        action_range=action_range,
        logprob_flag=logprob_flag,
        V_flag=V_flag
    )

    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind, transform=transform)

    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )


def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(trajectories, sample_size):

    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    p_sample = traj_lens / np.sum(traj_lens)

    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds
