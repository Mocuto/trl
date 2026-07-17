"""ReplayGRPOTrainer — ReSum-GRPO with an experience-replay buffer.

This is a thin SUBCLASS of ReSumGRPOTrainer. It changes exactly one thing:
how training batches are produced. Instead of generating fresh rollouts
every optimizer step (and discarding them after num_iterations), it
maintains a FIFO replay buffer, generates only every `replay_generate_every`
steps, and samples training minibatches from the buffer — recomputing GRPO
advantages over each sampled minibatch.

Everything else — the loss, the chunked training_step, generation, reward
funcs, logging — is inherited unchanged. The base on-policy ReSum path is
never modified; if you don't instantiate this class, nothing here runs.

Off-policy correction: old_per_token_logps can either be recomputed under the
current policy (default) or read from the stored generation-time logprobs.
Recomputing takes a fresh PPO snapshot at the start of each update, so the IS
ratio starts at ~1 and only diverges across the inner GRPO iterations — the
same behavior as on-policy GRPO. Using stored logprobs instead makes the ratio
measure the *full* policy drift since generation, which inflates the clip
fraction as buffer staleness grows. Toggle via `replay_recompute_old_logprobs`;
both paths are instrumented via replay/* metrics.

Pure buffer logic lives in replay_buffer.py (framework-agnostic, unit-tested).
"""
from __future__ import annotations

import torch

from ..models.utils import disable_gradient_checkpointing
from .resum_grpo_trainer import ReSumGRPOTrainer
from .utils import pad
from .replay_buffer import (
    ReplayBuffer,
    build_records_from_rollout,
    recompute_trajectory_advantages,
    flatten_records,
)


class ReplayGRPOTrainer(ReSumGRPOTrainer):
    """ReSum-GRPO with experience replay. See module docstring.

    Args (beyond the base trainer's):
        replay_buffer_size (`int`):
            Capacity N of the FIFO buffer, in trajectories.
        replay_batch_trajectories (`int`):
            Number of trajectories B to sample per training step.
        replay_generate_every (`int`, *optional*, defaults to `1`):
            Generate fresh rollouts every this many training steps. With 1,
            every step generates AND samples (max freshness, least savings).
            Larger values amortize generation across more gradient steps.
        replay_positive_bias_fraction (`float`, *optional*, defaults to `0.0`):
            Fraction of each sampled minibatch drawn from highest-reward
            trajectories (the paper's positive-bias sampling). 0 → uniform.
        replay_recompute_old_logprobs (`bool`, *optional*, defaults to `True`):
            If `True`, recompute `old_per_token_logps` under the current policy
            for each sampled minibatch (fresh PPO snapshot; IS ratio starts at
            ~1). If `False`, use the stored generation-time logprobs, so the IS
            ratio measures full policy drift since generation.
        replay_seed (`int`, *optional*, defaults to `0`):
            Seed for the buffer's sampling RNG.
        terminal_reward_weight (`float`, *optional*, defaults to `1.0`):
            Weight applied to terminal reward when aggregating a trajectory's
            scalar reward for advantage computation. Must match the terminal
            reward function's weight (sim9_reward_completion uses 1.0).
    """

    def __init__(
        self,
        *args,
        replay_buffer_size: int,
        replay_batch_trajectories: int,
        replay_generate_every: int = 1,
        replay_positive_bias_fraction: float = 0.0,
        replay_retention: str = "fifo",
        replay_deviant_fraction: float = 0.0,
        replay_recompute_old_logprobs: bool = True,
        replay_seed: int = 0,
        terminal_reward_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            seed=replay_seed,
            retention=replay_retention,
            deviant_fraction=replay_deviant_fraction,
        )
        self._replay_batch_trajectories = replay_batch_trajectories
        self._replay_generate_every = replay_generate_every
        self._replay_positive_bias_fraction = replay_positive_bias_fraction
        self._replay_recompute_old_logprobs = replay_recompute_old_logprobs
        self._replay_terminal_reward_weight = terminal_reward_weight

        # Wrap rollout_func to capture its raw output. The base calls
        # self.rollout_func(prompts, self) inside generation; we intercept
        # the return so we can build buffer records from it without
        # touching the base method.
        self._inner_rollout_func = self.rollout_func
        self._last_raw_rollout = None

        def _capturing_rollout_func(prompts, trainer):
            out = self._inner_rollout_func(prompts, trainer)
            self._last_raw_rollout = out
            return out

        self.rollout_func = _capturing_rollout_func

        # Startup summary + reuse-factor sanity check. R = trajectories per
        # rollout = generation_batch_size (one episode per prompt). The
        # reuse factor B*K/R is how many times each generated trajectory is
        # trained on, on average — the whole point of replay.
        R = self.args.generation_batch_size
        B = replay_batch_trajectories
        K = replay_generate_every
        reuse = (B * K / R) if R else float("nan")
        print(
            f"[ReplayGRPOTrainer] buffer_size(N)={replay_buffer_size} "
            f"sample/step(B)={B} generate_every(K)={K} "
            f"trajectories/rollout(R=generation_batch_size)={R}\n"
            f"[ReplayGRPOTrainer] retention={replay_retention} "
            f"deviant_fraction={replay_deviant_fraction} "
            f"sample_positive_bias={replay_positive_bias_fraction} "
            f"terminal_reward_weight={terminal_reward_weight}\n"
            f"[ReplayGRPOTrainer] => reuse_factor (B*K/R) = {reuse:.2f}  "
            f"(each generated trajectory trained on ~{reuse:.1f}x); "
            f"every {K} steps generates {R}, samples {B}/step"
        )
        if R and reuse < 1.0:
            print(
                f"[ReplayGRPOTrainer] WARNING: reuse_factor {reuse:.2f} < 1 — you "
                f"generate more trajectories per cycle ({R}) than you consume "
                f"({B}*{K}={B*K}); some generated trajectories will be evicted "
                f"before ever being sampled (wasted generation). Raise "
                f"--replay-batch-trajectories or --replay-generate-every, or "
                f"lower the generation batch."
            )
        if R and replay_buffer_size < 2 * R:
            print(
                f"[ReplayGRPOTrainer] WARNING: buffer_size {replay_buffer_size} < 2*R "
                f"({2 * R}) — the buffer barely spans one rollout, so uniform "
                f"sampling is ~subsampling the latest rollout (little staleness "
                f"diversity). Raise --replay-buffer-size to several multiples of "
                f"R ({R}) for real replay."
            )

    def _prepare_inputs(self, generation_batch):
        # Eval is unchanged — defer entirely to the base on-policy path.
        mode = "train" if self.model.training else "eval"
        if mode != "train":
            return super()._prepare_inputs(generation_batch)

        # Generation step: produce fresh rollouts (also runs reward funcs +
        # logs fresh-batch metrics via the inherited method) and add them to
        # the buffer. We discard the inherited method's assembled batch — we
        # train on a buffer sample instead. Always generate while the buffer
        # is still empty so step 0 has data.
        is_generation_step = (
            self._step % self._replay_generate_every == 0 or len(self._replay_buffer) == 0
        )
        if is_generation_step:
            self._last_raw_rollout = None
            # Side effects: runs the (wrapped) rollout_func → captures raw,
            # computes/logs fresh-batch reward metrics, fills the completions
            # parquet + memory/compaction tables.
            _ = self._generate_and_score_completions(generation_batch)
            if self._last_raw_rollout is not None:
                records = build_records_from_rollout(
                    self._last_raw_rollout,
                    generation_step=self._step,
                    terminal_reward_weight=self._replay_terminal_reward_weight,
                )
                self._replay_buffer.extend(records)

        # Sample a training minibatch from the buffer and recompute GRPO
        # advantages over it (re-injects reward variance from stale
        # trajectories; keeps the baseline current).
        sampled = self._replay_buffer.sample(
            self._replay_batch_trajectories,
            positive_bias_fraction=self._replay_positive_bias_fraction,
        )
        if not sampled:
            # Buffer somehow empty (shouldn't happen after a generation step) —
            # fall back to the base path so we never hand back an empty batch.
            return super()._prepare_inputs(generation_batch)

        advantages = recompute_trajectory_advantages([r.reward for r in sampled])
        flat = flatten_records(sampled, advantages)
        inputs = self._assemble_inputs(flat)
        self._log_replay_stats(sampled)
        return inputs

    def _assemble_inputs(self, flat: dict) -> dict:
        """Pad the framework-agnostic flat lists into the batch dict the
        loss expects. Matches the base's padding conventions exactly:
        prompt left-padded, completion + logprobs right-padded.
        """
        device = self.accelerator.device
        pad_id = self._tokenizer.pad_token_id

        prompt_ids = [torch.tensor(ids, dtype=torch.long) for ids in flat["prompt_ids"]]
        prompt_mask = [torch.ones_like(t, dtype=torch.long) for t in prompt_ids]
        prompt_ids = pad(
            prompt_ids, padding_value=pad_id, padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)

        completion_ids = [torch.tensor(ids, dtype=torch.long) for ids in flat["completion_ids"]]
        completion_mask = [torch.ones_like(t, dtype=torch.long) for t in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=pad_id, padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)

        # old_per_token_logps: either recomputed under the current policy (a
        # fresh PPO snapshot, so the IS ratio starts at ~1 and only diverges
        # across the inner GRPO iterations) or read from the stored
        # generation-time logprobs (ratio measures full drift since generation,
        # which inflates the clip fraction as buffer staleness grows).
        if self._replay_recompute_old_logprobs:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            # Mirrors the base GRPOTrainer's old_per_token_logps computation:
            # no_grad + checkpointing disabled (avoids the requires_grad warning).
            with torch.no_grad(), disable_gradient_checkpointing(
                self.model, self.args.gradient_checkpointing_kwargs
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=self.args.per_device_train_batch_size,
                )
        else:
            # Generation-time logprobs become old_per_token_logps (right-padded,
            # aligned with completion_ids). len(logprobs)==len(completion_ids)
            # per segment (one logprob per completion token).
            old_logps = [torch.tensor(lp, dtype=torch.float32) for lp in flat["old_logprobs"]]
            old_per_token_logps = pad(
                old_logps, padding_value=0.0, padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device)

        advantages = torch.tensor(flat["advantages"], dtype=torch.float32, device=device)
        segment_groups = torch.tensor(flat["segment_groups"], dtype=torch.long, device=device)
        num_items_in_batch = completion_mask.sum()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "segment_groups": segment_groups,
            "num_items_in_batch": num_items_in_batch,
        }

    def _log_replay_stats(self, sampled) -> None:
        """Emit replay-specific metrics: buffer fill, staleness, sampled-batch
        reward spread, and how much reward variance the sample restored."""
        mode = "train" if self.model.training else "eval"
        stats = self._replay_buffer.stats(self._step)
        staleness = self._replay_buffer.staleness(sampled, self._step)
        sample_rewards = [r.reward for r in sampled]

        self._metrics[mode]["replay/buffer_size"].append(float(stats.get("size", 0)))
        self._metrics[mode]["replay/buffer_fill_fraction"].append(float(stats.get("fill_fraction", 0.0)))
        self._metrics[mode]["replay/buffer_staleness_mean"].append(float(stats.get("staleness_mean", 0.0)))
        self._metrics[mode]["replay/buffer_staleness_max"].append(float(stats.get("staleness_max", 0.0)))
        # buffer_reward_std is the headline signal for retention policy: with
        # "deviance" it should stay healthily nonzero even as the policy
        # improves and fresh-batch reward variance collapses. If it decays
        # toward 0 like fifo, the std-collapse crash risk returns.
        self._metrics[mode]["replay/buffer_reward_mean"].append(float(stats.get("reward_mean", 0.0)))
        self._metrics[mode]["replay/buffer_reward_std"].append(float(stats.get("reward_std", 0.0)))
        # min/max expose the buffer's reward spread directly — e.g. whether
        # deviance retention is pinning a stale very-negative trajectory that
        # drags buffer_reward_mean below live rollout performance.
        self._metrics[mode]["replay/buffer_reward_min"].append(float(stats.get("reward_min", 0.0)))
        self._metrics[mode]["replay/buffer_reward_max"].append(float(stats.get("reward_max", 0.0)))
        if staleness:
            self._metrics[mode]["replay/sample_staleness_mean"].append(sum(staleness) / len(staleness))
            self._metrics[mode]["replay/sample_staleness_max"].append(float(max(staleness)))
        if sample_rewards:
            n = len(sample_rewards)
            mean_r = sum(sample_rewards) / n
            std_r = (sum((x - mean_r) ** 2 for x in sample_rewards) / (n - 1)) ** 0.5 if n > 1 else 0.0
            # sample_reward_std is the key one: if it stays healthy while the
            # FRESH-batch reward std collapses (success streak), the buffer is
            # doing its job — re-injecting variance and keeping advantages
            # well-conditioned.
            self._metrics[mode]["replay/sample_reward_mean"].append(mean_r)
            self._metrics[mode]["replay/sample_reward_std"].append(std_r)
            self._metrics[mode]["replay/sample_trajectories"].append(float(n))
