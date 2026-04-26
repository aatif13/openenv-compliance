"""
Maps post-penalty raw grader values into a final score in [0, 1].
All graders use this as the last step on their computed value.

reward_to_score() is strictly non-decreasing: linear clamp to [0.0, 1.0].
"""

from __future__ import annotations


def reward_to_score(reward: float | None) -> float:
    if reward is None:
        return 0.0
    try:
        reward = float(reward)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, reward))
