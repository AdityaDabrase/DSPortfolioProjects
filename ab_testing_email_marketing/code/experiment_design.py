"""Experiment design toolkit: power, minimum detectable effect, sample size.

In a real company this work happens BEFORE the experiment launches -
it is the "how big does the test need to be?" conversation. We include
it (and run it against the Hillstrom test's actual dimensions) to show
the full workflow, not just the after-the-fact analysis.

Run directly for a design report:
    python experiment_design.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def sample_size_two_proportions(
    baseline_rate: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Sample size PER ARM to detect an absolute lift of `mde_abs`.

    This is the standard two-proportion formula:

        n = (z_{1-a/2} * sqrt(2*p*(1-p)) + z_{power} * sqrt(p1(1-p1)+p2(1-p2)))^2
            -----------------------------------------------------------------
                                   (p2 - p1)^2

    Intuition for each piece:
      - alpha (via z_{1-a/2}): how often we tolerate a false positive.
      - power (via z_{power}): how often we want to CATCH a real effect.
        80% power means a real effect of exactly MDE size is missed 1 time in 5.
      - The denominator (effect size squared) is why small effects are
        expensive: halving the MDE quadruples the required sample.
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde_abs
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-sided test
    z_power = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(numerator / (p2 - p1) ** 2))


def mde_two_proportions(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Invert the question: given the sample we HAVE, what's the smallest
    absolute lift we can reliably detect?

    This is the honest way to interpret a finished experiment: if the MDE
    at n=21,306/arm is 0.4pp and we measured +0.2pp "not significant",
    the right conclusion is "the test couldn't see effects that small",
    NOT "there is no effect". Solved numerically by bisection because the
    sample-size formula has no closed-form inverse.
    """
    lo, hi = 1e-6, 1 - baseline_rate - 1e-6
    for _ in range(100):  # bisection converges fast; 100 iters is overkill but cheap
        mid = (lo + hi) / 2
        if sample_size_two_proportions(baseline_rate, mid, alpha, power) > n_per_arm:
            lo = mid   # need a bigger effect to be detectable at this n
        else:
            hi = mid
    return hi


def power_achieved(
    baseline_rate: float,
    effect_abs: float,
    n_per_arm: int,
    alpha: float = 0.05,
) -> float:
    """Probability this experiment detects an effect of the given size.

    Used by the auditor's power check: a "no significant difference"
    verdict from an underpowered test is worthless, and this number is
    how you prove it either way.
    """
    p1, p2 = baseline_rate, baseline_rate + effect_abs
    se_null = np.sqrt(2 * p1 * (1 - p1) / n_per_arm)
    se_alt = np.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # Power = P(z-statistic exceeds the critical value | the effect is real)
    return float(stats.norm.sf((z_alpha * se_null - abs(effect_abs)) / se_alt))


def design_report(
    baseline_rate: float = 0.005,
    n_per_arm: int = 21_306,
    mde_targets: tuple[float, ...] = (0.001, 0.002, 0.003, 0.005),
) -> str:
    """Pre-registration style design report for the Hillstrom conversion metric.

    Defaults reflect the actual experiment: ~0.5% baseline conversion
    (control arm) and ~21.3k customers per arm (64k / 3).
    """
    lines = [
        "EXPERIMENT DESIGN REPORT (conversion metric)",
        "=" * 60,
        f"Baseline conversion rate : {baseline_rate:.2%}",
        f"Alpha (two-sided)        : 0.05",
        f"Target power             : 80%",
        "",
        "Sample size required per arm, by target MDE:",
    ]
    for mde in mde_targets:
        n = sample_size_two_proportions(baseline_rate, mde)
        feasible = "OK at current size" if n <= n_per_arm else "NOT reachable at current size"
        lines.append(f"  detect {mde*100:+.2f}pp lift -> n = {n:>9,} per arm   [{feasible}]")

    mde_now = mde_two_proportions(baseline_rate, n_per_arm)
    lines += [
        "",
        f"With the actual n = {n_per_arm:,} per arm:",
        f"  minimum detectable effect = {mde_now*100:.3f}pp absolute "
        f"({mde_now/baseline_rate:.0%} relative lift)",
        "",
        "Reading this: effects smaller than the MDE can exist and this",
        "experiment would usually miss them. 'Not significant' != 'no effect'.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(design_report())
