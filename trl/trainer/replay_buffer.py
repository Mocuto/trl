"""Experience-replay buffer for off-policy LLM RL training.

This module is intentionally framework-agnostic and self-contained: it
holds plain Python data (token-id lists, float rewards/logprobs) and has
no torch / transformers / TRL dependency. That keeps it unit-testable on
CPU and reusable across environments (the multi-env training goal).

It is consumed by ReplayGRPOTrainer (replay_grpo_trainer.py), which is a
SUBCLASS of ReSumGRPOTrainer — the base on-policy ReSum path is never
modified or imported here. When the replay buffer is disabled, none of
this code runs.

Design (per Arnal et al. 2026, "Efficient RL Training for LLMs with
Experience Replay"):
  - FIFO buffer of the N freshest trajectories.
  - Uniform sampling (optionally positive-biased toward high reward).
  - GRPO advantages are RECOMPUTED over each sampled minibatch, not the
    generation-time batch — this both keeps the baseline current and
    re-injects reward variance from stale trajectories, counteracting the
    std-collapse-on-success failure mode of pure on-policy GRPO.

What this module does NOT handle: the importance-weight staleness
(policy drift since generation). The stored generation-time logprobs are
used as old_per_token_logps by the trainer; correcting/mitigating that
gap (recompute-under-current-policy, AsymRE, ReVal) is the trainer's
concern, deliberately kept out of the buffer.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SegmentData:
    """One LLM segment (one generate() call) within a trajectory."""
    prompt_ids: List[int]
    completion_ids: List[int]
    logprobs: List[float]  # generation-time per-completion-token logprobs


@dataclass
class TrajectoryRecord:
    """One trajectory (one episode), possibly spanning multiple segments.

    All segments of a trajectory share the trajectory's reward — that's
    the ReSum advantage-broadcasting contract. The reward stored here is
    the final scalar reward used for advantage computation (terminal +
    summed step rewards), already aggregated by the caller.
    """
    segments: List[SegmentData]
    reward: float
    generation_step: int                 # trainer global step at generation; for staleness
    metadata: Dict = field(default_factory=dict)  # seed / episode_idx / etc., for logging only

    @property
    def num_segments(self) -> int:
        return len(self.segments)


class ReplayBuffer:
    """Replay buffer of TrajectoryRecords with pluggable retention + sampling.

    Retention (which records survive when over capacity) — three variants,
    all A/B-comparable:
      - "fifo": keep the N freshest, drop the oldest. Simple, but as the
        policy improves the buffer fills with uniformly-good (low-variance)
        trajectories → GRPO advantage std collapses → the success-streak
        crash. This is the failure mode the paper and our runs observe.
      - "deviance": keep the freshest (1-δ)N PLUS the δN MOST-DEVIANT of
        the remaining (older) trajectories, where deviance = |reward -
        buffer_mean_reward|. Deliberately retains the reward extremes
        (both tails) to keep buffer variance alive even when fresh
        generations have collapsed to uniform success, keeping advantage
        normalization well-conditioned. Tradeoff: retained deviant
        trajectories are typically OLDER → more policy drift; watch
        replay/buffer_staleness_max + the IS-ratio / clip metrics.
      - "positive": the paper's variant — keep the freshest (1-δ)N PLUS
        the δN HIGHEST-REWARD of the remainder. Likely overlaps the recent
        set heavily as the policy improves (recent ≈ best), so adds less
        variance than "deviance", but stays closer to on-policy (less
        drift). Kept so we can compare it head-to-head against deviance.

    The δ knob is `deviant_fraction` for both "deviance" and "positive".

    Sampling (which records a training minibatch draws) is independent:
    uniform or positive-bias, controlled per sample() call.
    """

    _RETENTION_MODES = ("fifo", "deviance", "positive")

    def __init__(
        self,
        capacity: int,
        seed: int = 0,
        retention: str = "fifo",
        deviant_fraction: float = 0.0,
    ):
        if capacity <= 0:
            raise ValueError(f"ReplayBuffer capacity must be > 0, got {capacity}")
        if retention not in self._RETENTION_MODES:
            raise ValueError(
                f"retention must be one of {self._RETENTION_MODES}, got {retention!r}"
            )
        if not (0.0 <= deviant_fraction < 1.0):
            raise ValueError(f"deviant_fraction must be in [0, 1), got {deviant_fraction}")
        self.capacity = capacity
        self.retention = retention
        self.deviant_fraction = deviant_fraction
        # Plain list, ordered oldest -> newest. We prune explicitly rather
        # than relying on deque(maxlen) so retention can be non-FIFO.
        self._buf: List[TrajectoryRecord] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, record: TrajectoryRecord) -> None:
        self._buf.append(record)
        self._prune()

    def extend(self, records: List[TrajectoryRecord]) -> None:
        self._buf.extend(records)
        self._prune()

    def _prune(self) -> None:
        """Drop records down to capacity using the configured retention policy."""
        if len(self._buf) <= self.capacity:
            return
        if self.retention in ("deviance", "positive") and self.deviant_fraction > 0.0:
            self._buf = self._prune_freshest_plus_scored()
        else:
            # FIFO (or deviance/positive with δ=0): keep the N freshest.
            self._buf = self._buf[-self.capacity:]

    def _prune_freshest_plus_scored(self) -> List[TrajectoryRecord]:
        """Keep freshest (1-δ)N + the top-δN of the remainder by a score.

        Score depends on retention mode:
          - "deviance": |reward - buffer_mean| (retains reward extremes →
            preserves variance)
          - "positive": reward (retains highest-reward older trajectories,
            the paper's variant)

        Insertion order is preserved in the result so future prunes still
        see correct freshness ordering.
        """
        n = self.capacity
        n_extra = round(self.deviant_fraction * n)
        n_fresh = n - n_extra
        if n_fresh <= 0:  # degenerate; shouldn't happen since δ < 1
            n_fresh = 1
            n_extra = n - 1

        fresh = self._buf[-n_fresh:]
        older = self._buf[:-n_fresh]
        if n_extra <= 0 or not older:
            return self._buf[-n:]

        if self.retention == "deviance":
            rewards = [r.reward for r in self._buf]
            mean = sum(rewards) / len(rewards)
            key = lambda r: abs(r.reward - mean)
        else:  # "positive"
            key = lambda r: r.reward

        older_ranked = sorted(older, key=key, reverse=True)
        extra = older_ranked[:n_extra]

        retained = {id(x) for x in fresh} | {id(x) for x in extra}
        # Preserve oldest->newest order among retained records.
        return [x for x in self._buf if id(x) in retained]

    def sample(
        self,
        n: int,
        positive_bias_fraction: float = 0.0,
    ) -> List[TrajectoryRecord]:
        """Sample n trajectory records.

        positive_bias_fraction δ in [0, 1): of the n drawn, ceil(δ·n) come
        from the highest-reward trajectories in the buffer and the rest are
        sampled uniformly from the remainder (the paper's "positive-bias
        sampling"). δ=0 → pure uniform.

        Without replacement when n <= len(buffer); with replacement only if
        the caller asks for more than the buffer holds (early training,
        before the buffer fills).
        """
        size = len(self._buf)
        if size == 0:
            return []
        if not (0.0 <= positive_bias_fraction < 1.0):
            raise ValueError(
                f"positive_bias_fraction must be in [0, 1), got {positive_bias_fraction}"
            )

        items = list(self._buf)

        if positive_bias_fraction > 0.0:
            n_pos = math.ceil(positive_bias_fraction * n)
            ranked = sorted(items, key=lambda r: r.reward, reverse=True)
            pos_pool = ranked[:n_pos]
            rest_pool = ranked[n_pos:]
            chosen = list(pos_pool[: min(n_pos, len(pos_pool))])
            remaining = n - len(chosen)
            if remaining > 0:
                pool = rest_pool if rest_pool else items
                chosen += self._draw(pool, remaining)
            return chosen

        return self._draw(items, n)

    def _draw(self, pool: List[TrajectoryRecord], n: int) -> List[TrajectoryRecord]:
        if not pool:
            return []
        if n <= len(pool):
            return self._rng.sample(pool, n)
        # Need more than available — sample with replacement to fill.
        return [self._rng.choice(pool) for _ in range(n)]

    def staleness(self, records: List[TrajectoryRecord], current_step: int) -> List[int]:
        """Per-record staleness = steps elapsed since the record was generated."""
        return [current_step - r.generation_step for r in records]

    def stats(self, current_step: int) -> Dict[str, float]:
        """Summary stats for logging."""
        if not self._buf:
            return {"size": 0, "capacity": self.capacity}
        ages = [current_step - r.generation_step for r in self._buf]
        rewards = [r.reward for r in self._buf]
        return {
            "size": len(self._buf),
            "capacity": self.capacity,
            "fill_fraction": len(self._buf) / self.capacity,
            "staleness_mean": sum(ages) / len(ages),
            "staleness_max": max(ages),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std": _sample_std(rewards),
        }


def _sample_std(xs: List[float]) -> float:
    """Sample standard deviation (Bessel-corrected, /(N-1)).

    Matches torch.std default (unbiased=True) so advantage normalization
    here is numerically consistent with the on-policy ReSum path.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def recompute_trajectory_advantages(rewards: List[float], eps: float = 1e-4) -> List[float]:
    """GRPO advantage over a set of trajectory rewards: (r - mean) / (std + eps).

    Mirrors the on-policy ReSum computation (resum_grpo_trainer.py lines
    2364-2373) but is applied to the SAMPLED minibatch's trajectory
    rewards rather than the generation batch. Uses sample std to match
    torch.std. With a single trajectory, std is 0 → advantage 0.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    std = _sample_std(rewards)
    return [(r - mean) / (std + eps) for r in rewards]


def flatten_records(
    records: List[TrajectoryRecord],
    advantages: List[float],
) -> Dict[str, list]:
    """Expand sampled trajectories into per-segment, framework-agnostic lists.

    Returns dict with aligned, segment-ordered lists ready for the trainer
    to pad into tensors:
      - prompt_ids:     list[list[int]]
      - completion_ids: list[list[int]]
      - old_logprobs:   list[list[float]]   (generation-time logprobs)
      - advantages:     list[float]         (trajectory advantage broadcast per segment)
      - segment_groups: list[int]           (segment -> sampled-trajectory index)

    advantages[i] is the advantage for records[i], broadcast to every
    segment of that trajectory.
    """
    if len(records) != len(advantages):
        raise ValueError(
            f"records ({len(records)}) and advantages ({len(advantages)}) length mismatch"
        )
    prompt_ids: List[List[int]] = []
    completion_ids: List[List[int]] = []
    old_logprobs: List[List[float]] = []
    seg_advantages: List[float] = []
    segment_groups: List[int] = []

    for traj_idx, (record, adv) in enumerate(zip(records, advantages)):
        for seg in record.segments:
            prompt_ids.append(seg.prompt_ids)
            completion_ids.append(seg.completion_ids)
            old_logprobs.append(seg.logprobs)
            seg_advantages.append(adv)
            segment_groups.append(traj_idx)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "old_logprobs": old_logprobs,
        "advantages": seg_advantages,
        "segment_groups": segment_groups,
    }


def build_records_from_rollout(
    rollout_output: Dict,
    generation_step: int,
    terminal_reward_weight: float = 1.0,
) -> List[TrajectoryRecord]:
    """Group a rollout_func output dict into per-trajectory records.

    Expects the sim9 rollout output format:
      - prompt_ids:        list[list[int]]   per segment
      - completion_ids:    list[list[int]]   per segment
      - logprobs:          list[list[float]] per segment
      - segment_groups:    list[int]         segment -> trajectory index
      - completion_reward: list[float]       per segment (broadcast within trajectory)
      - step_rewards:      list[list[float]] per segment (the episode's step rewards, broadcast)

    Trajectory reward = terminal_reward_weight * completion_reward
                        + sum(step_rewards), matching what
    sim9_reward_completion + sim9_reward_step produce with unit reward
    weights. step_rewards are already self-weighted by their
    StepRewardComputer, so no extra scale is applied here.
    """
    prompt_ids = rollout_output["prompt_ids"]
    completion_ids = rollout_output["completion_ids"]
    logprobs = rollout_output["logprobs"]
    segment_groups = rollout_output["segment_groups"]
    completion_reward = rollout_output["completion_reward"]
    step_rewards = rollout_output["step_rewards"]

    n_seg = len(prompt_ids)
    for name, seq in [
        ("completion_ids", completion_ids), ("logprobs", logprobs),
        ("segment_groups", segment_groups), ("completion_reward", completion_reward),
        ("step_rewards", step_rewards),
    ]:
        if len(seq) != n_seg:
            raise ValueError(f"rollout field '{name}' length {len(seq)} != n_segments {n_seg}")

    # Group segment indices by trajectory id.
    by_traj: Dict[int, List[int]] = {}
    for seg_idx, g in enumerate(segment_groups):
        by_traj.setdefault(int(g), []).append(seg_idx)

    records: List[TrajectoryRecord] = []
    for g in sorted(by_traj):
        seg_idxs = by_traj[g]
        segments = [
            SegmentData(
                prompt_ids=list(prompt_ids[i]),
                completion_ids=list(completion_ids[i]),
                logprobs=list(logprobs[i]),
            )
            for i in seg_idxs
        ]
        # All segments in a trajectory share completion_reward + step_rewards;
        # read from the first segment of the group.
        first = seg_idxs[0]
        step_sum = float(sum(step_rewards[first])) if step_rewards[first] else 0.0
        reward = terminal_reward_weight * float(completion_reward[first]) + step_sum
        records.append(TrajectoryRecord(
            segments=segments,
            reward=reward,
            generation_step=generation_step,
            metadata={"rollout_traj_index": g},
        ))
    return records
