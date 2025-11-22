"""
GSPO and GSPO-token (vectorized) trainer utilities.

This is a self-contained port of the validated implementation originally prototyped in
`Abstractions/RL`. It supports both the sequence-level GSPO objective and the GSPO-token
extension described in:

    Zheng, C., et al. "Group Sequence Policy Optimization." arXiv:2507.18071 (2025).

Usage summary:
    - Configure via `GSPOTokenConfig` (set `use_token_level=True` for GSPO-token).
    - Provide policy/ref models, optimizer, prompts, and reward_fn.
    - `reward_fn` may return:
        * Tensor (sequence rewards) => GSPO baseline
        * Dict with {"rewards": Tensor, "token_advantages": Tensor} => GSPO-token
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


Tensor = torch.Tensor


class GSPOTokenError(RuntimeError):
    """Base exception for GSPO/GSPOToken utilities."""


@dataclass
class GSPOTokenConfig:
    """Configuration for GSPO / GSPO-token training."""

    group_size: int = 4
    epsilon: float = 0.2
    max_length: int = 512
    eos_token: int = 2
    pad_token: int = 0
    use_token_level: bool = False

    def validate(self) -> None:
        assert self.group_size > 0, "group_size must be positive"
        assert 0 < self.epsilon < 1, "epsilon must be in (0,1)"
        assert self.max_length > 0, "max_length must be positive"
        assert self.eos_token != self.pad_token, "EOS and pad tokens must differ"


RewardOutput = Union[Tensor, Dict[str, Tensor]]


class GSPOTokenTrainer:
    """Vectorized GSPO / GSPO-token trainer."""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        optimizer: optim.Optimizer,
        config: Optional[GSPOTokenConfig],
        reward_fn: Callable[[Tensor, Tensor], RewardOutput],
    ) -> None:
        self.config = config or GSPOTokenConfig()
        self.config.validate()

        self.policy = policy_model
        self.reference = reference_model
        self.optimizer = optimizer
        self.reward_fn = reward_fn

        for p in self.reference.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample_responses(self, prompts: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        prompts_exp = prompts.repeat_interleave(cfg.group_size, dim=0)
        total = prompts_exp.size(0)
        device = prompts.device

        responses = torch.full(
            (total, cfg.max_length), cfg.pad_token, dtype=torch.long, device=device
        )
        lengths = torch.zeros(total, dtype=torch.long, device=device)
        active = torch.ones(total, dtype=torch.bool, device=device)
        current = prompts_exp.clone()

        self.reference.eval()
        with torch.no_grad():
            for step in range(cfg.max_length):
                logits = self.reference(current)[:, -1, :]
                dist = Categorical(logits=logits)
                nxt = dist.sample()
                nxt[~active] = cfg.pad_token
                responses[:, step] = nxt
                current = torch.cat([current, nxt.unsqueeze(1)], dim=1)
                finished = (nxt == cfg.eos_token) & active
                lengths[finished] = step + 1
                active = active & (nxt != cfg.eos_token)
                if not active.any():
                    break
            lengths[active] = cfg.max_length

        max_len = lengths.max().item()
        responses = responses[:, :max_len]
        full_sequences = torch.cat([prompts_exp, responses], dim=1)
        return responses, lengths, full_sequences

    # ------------------------------------------------------------------
    # Log-prob helpers
    # ------------------------------------------------------------------
    def _log_probs(
        self,
        model: nn.Module,
        full_sequences: Tensor,
        prompt_len: int,
        response_lengths: Tensor,
    ) -> Tensor:
        inp = full_sequences[:, :-1]
        targets = full_sequences[:, 1:]
        logits = model(inp)
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        positions = torch.arange(gathered.size(1), device=gathered.device).unsqueeze(0)
        resp_mask = (positions >= prompt_len) & (
            positions < (prompt_len + response_lengths.unsqueeze(1))
        )
        masked = gathered * resp_mask.float()
        result = masked.sum(dim=1)
        if torch.isnan(result).any() or torch.isinf(result).any():
            raise GSPOTokenError("Invalid log probabilities encountered.")
        return result

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(self, prompts: Tensor) -> Dict[str, float]:
        cfg = self.config
        batch = prompts.size(0)
        responses, lengths, full_sequences = self._sample_responses(prompts)
        prompts_exp = prompts.repeat_interleave(cfg.group_size, dim=0)

        reward_output = self.reward_fn(prompts_exp, responses)
        if isinstance(reward_output, dict):
            rewards = reward_output.get("rewards")
            token_adv = reward_output.get("token_advantages")
        else:
            rewards = reward_output
            token_adv = None

        if rewards is None:
            raise GSPOTokenError("Reward function must provide `rewards`.")

        rewards_grouped = rewards.view(batch, cfg.group_size)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_grouped - mean_r) / std_r

        ref_log = self._log_probs(self.reference, full_sequences, prompts.size(1), lengths)
        pol_log = self._log_probs(self.policy, full_sequences, prompts.size(1), lengths)
        log_ratios = (pol_log - ref_log) / lengths.float()
        ratios = torch.exp(log_ratios)
        clipped = torch.clamp(ratios, 1 - cfg.epsilon, 1 + cfg.epsilon)

        if cfg.use_token_level:
            if token_adv is None:
                raise GSPOTokenError(
                    "GSPO-token requires `token_advantages` in reward output. "
                    "Expected dict with {'rewards': Tensor, 'token_advantages': Tensor}."
                )
            if token_adv.size() != responses.size():
                raise GSPOTokenError(
                    f"Token advantages shape {token_adv.size()} does not match responses {responses.size()}."
                )
            mask = (
                torch.arange(token_adv.size(1), device=token_adv.device)
                .unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            per_seq_adv = (token_adv * mask.float()).sum(dim=1) / lengths.float().clamp_min(
                1.0
            )
            adv_vector = per_seq_adv
        else:
            adv_vector = advantages.view(-1)

        term1 = ratios * adv_vector
        term2 = clipped * adv_vector
        loss = -torch.min(term1, term2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "mean_reward": float(rewards.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "mean_ratio": float(ratios.mean().item()),
            "clipped_fraction": float((ratios != clipped).float().mean().item()),
        }
        if cfg.use_token_level:
            metrics["token_adv_mean"] = float(per_seq_adv.mean().item())

        return metrics


def run_gspo_token_vectorized_step(
    policy_model: nn.Module,
    reference_model: nn.Module,
    optimizer: optim.Optimizer,
    prompts: Tensor,
    reward_fn: Callable[[Tensor, Tensor], RewardOutput],
    config: Optional[GSPOTokenConfig] = None,
) -> Dict[str, float]:
    """Convenience wrapper to execute a single GSPO / GSPO-token step."""

    trainer = GSPOTokenTrainer(policy_model, reference_model, optimizer, config, reward_fn)
    return trainer.update(prompts)

