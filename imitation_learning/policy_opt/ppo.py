# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate

from imitation_learning.policy_opt.policy import Policy


class PPO:
    def __init__(
        self,
        use_gae: bool,
        gae_lambda: float,
        gamma: float,
        use_clipped_value_loss: bool,
        clip_param: bool,
        value_loss_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        num_mini_batch: int,
        num_epochs: int,
        optimizer_params: Dict[str, Any],
        num_steps: int,
        num_envs: int,
        policy: Policy,
        **kwargs,
    ):
        super().__init__()
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_mini_batch = num_mini_batch
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_envs = num_envs

        self.opt: torch.optim.Optimizer = hydra_instantiate(
            optimizer_params, params=policy.parameters()
        )
        self.returns = torch.zeros(self.num_steps + 1, self.num_envs, 1)

    def state_dict(self):
        return {"opt": self.opt.state_dict()}

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("opt")
        if should_load_opt:
            self.opt.load_state_dict(opt_state)

    def update(
        self,
        policy,
        storage,
        logger,
        **kwargs,
    ):
        with torch.no_grad():
            last_value = policy.get_value(
                storage.get_obs(-1),
                storage.recurrent_hidden_states[-1],
                storage.masks[-1],
            )

        advantages = self.compute_derived(
            policy,
            storage.rewards,
            storage.masks,
            storage.bad_masks,
            storage.value_preds,
            last_value,
        )

        for _ in range(self.num_epochs):
            data_gen = storage.data_generator(
                self.num_mini_batch, returns=self.returns[:-1], advantages=advantages
            )
            for sample in data_gen:
                ac_eval = policy.evaluate_actions(
                    sample["obs"],
                    sample["hxs"],
                    sample["mask"],
                    sample["action"],
                )

                ratio = torch.exp(ac_eval["log_prob"] - sample["prev_log_prob"])
                surr1 = ratio * sample["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                    * sample["advantages"]
                )
                action_loss = -torch.min(surr1, surr2).mean(0)

                if self.use_clipped_value_loss:
                    value_pred_clipped = sample["value"] + (
                        ac_eval["value"] - sample["value"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (ac_eval["value"] - sample["returns"]).pow(2)
                    value_losses_clipped = (value_pred_clipped - sample["returns"]).pow(
                        2
                    )
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = (
                        0.5 * (sample["returns"] - ac_eval["value"]).pow(2).mean()
                    )

                loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - ac_eval["dist_entropy"].mean() * self.entropy_coef
                )

                self.opt.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                self.opt.step()

                logger.collect_info("value_loss", value_loss.mean().item())
                logger.collect_info("action_loss", action_loss.mean().item())
                logger.collect_info(
                    "dist_entropy", ac_eval["dist_entropy"].mean().item()
                )

    def compute_derived(
        self,
        policy,
        rewards,
        masks,
        bad_masks,
        value_preds,
        last_value,
    ):
        if self.use_gae:
            value_preds[-1] = last_value
            gae = 0
            for step in reversed(range(rewards.size(0))):
                delta = (
                    rewards[step]
                    + self.gamma * value_preds[step + 1] * masks[step + 1]
                    - value_preds[step]
                )
                gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae
                gae = gae * bad_masks[step + 1]
                self.returns[step] = gae + value_preds[step]
        else:
            self.returns[-1] = last_value
            for step in reversed(range(rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * self.gamma * masks[step + 1]
                    + rewards[step]
                ) * bad_masks[step + 1] + (1 - bad_masks[step + 1]) * value_preds[step]
        advantages = self.returns[:-1] - value_preds[:-1]

        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        return advantages
