# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch
from rl_utils.common import DictDataset
import json

def log_finished_rewards(
    rollouts,
    rolling_ep_rewards: torch.Tensor,
    logger,
) -> torch.Tensor:
    """
    :param rolling_ep_rewards: tensor of shape (num_envs,)
    """
    num_steps, num_envs = rollouts.rewards.shape[:2]
    done_episodes_rewards = []
    for env_i in range(num_envs):
        for step_i in range(num_steps):
            rolling_ep_rewards[env_i] += rollouts.rewards[step_i, env_i].item()
            if rollouts.masks[step_i + 1, env_i].item() == 0.0:
                done_episodes_rewards.append(rolling_ep_rewards[env_i].item())
                rolling_ep_rewards[env_i] = 0
    logger.collect_info_list("inferred_episode_reward", done_episodes_rewards)
    return rolling_ep_rewards


def extract_transition_batch(rollouts):
    obs = next(iter(rollouts.obs.values()))
    cur_obs = obs[:-1]
    masks = rollouts.masks[1:]
    next_obs = (masks * obs[1:]) + ((1 - masks) * rollouts.final_obs)
    actions = rollouts.actions
    return cur_obs, actions, next_obs, masks


def create_next_obs(dataset) -> Dict[str, torch.Tensor]:
    obs = dataset["observations"].detach()

    final_final_obs = dataset["infos"][-1]["final_obs"]

    next_obs = torch.cat([obs[1:], final_final_obs.unsqueeze(0)], 0)
    num_eps = 1
    for i in range(obs.shape[0] - 1):
        cur_info = dataset["infos"][i]
        if "final_obs" in cur_info:
            num_eps += 1
            next_obs[i] = cur_info["final_obs"].detach()

    if num_eps != dataset["terminals"].sum():
        raise ValueError(
            f"Inconsistency in # of episodes {num_eps} vs {dataset['terminals'].sum()}"
        )
    dataset["next_observations"] = next_obs.detach()

    return dataset


def get_dataset_data(dataset_path: str, env_name: str):
    
    with open(dataset_path) as f:

        loaded = json.load(f)

        tensored = {key: torch.tensor(value) for key,value in loaded.items() if key != "infos"}

        tensored["infos"] = [{key: torch.tensor(value) if (isinstance(value,list) or isinstance(value,float) ) else {k:torch.tensor(val) for k,val in value.items()} for key,value in info.items()} for info in loaded["infos"]]

        return create_next_obs(tensored)



def get_transition_dataset(dataset_path: str, env_name: str):
    dataset = get_dataset_data(dataset_path, env_name)

    return DictDataset(
        dataset,
        [
            "observations",
            "actions",
            "rewards",
            "terminals",
            "next_observations",
        ],
    )
