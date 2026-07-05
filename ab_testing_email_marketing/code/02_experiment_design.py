"""
Experiment design: the work you do BEFORE looking at outcomes.

Real experimentation teams write this document before the test launches
(pre-registration). It commits us to a primary metric, a significance level,
and a minimum effect worth detecting -- so that after the data arrives we
can't unconsciously move the goalposts to whatever looks significant.

Run:  python 02_experiment_design.py
"""

import sys

from utils import (achieved_power, minimum_detectable_effect,
                   required_sample_size)

# ---------------------------------------------------------------------------
# 1. The pre-registered design choices
# ---------------------------------------------------------------------------
# PRIMARY metric: conversion (did the customer purchase within two weeks?).
# Why conversion and not spend? Spend is noisier (zero-inflated, long right
# tail), so it needs far more data for the same power. Why not visit? Visits
# don't pay the bills -- conversion is the closest binary proxy for revenue.
# Spend and visit are SECONDARY metrics: reported, but the ship/no-ship
# decision keys off the primary.

ALPHA = 0.05          # two-sided false-positive budget (industry default)
POWER = 0.80          # 80% chance of detecting a true effect of size MDE
BASELINE_CONVERSION = 0.006   # ~0.6%: typical retail email-window baseline;
                              # the control arm of this test lands near here.

# The smallest lift that would change the business decision. An email blast
# is nearly free to send but has hidden costs (list fatigue, unsubscribes),
# so we set the bar at +0.2pp absolute -- roughly a 33% relative lift on a
# 0.6% baseline. Anything smaller wouldn't justify occupying the send slot.
MDE_ABS = 0.002


def main():
    print("=" * 70)
    print("EXPERIMENT DESIGN -- pre-registered before analysis")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 2. How many customers does this design require?
    # -----------------------------------------------------------------------
    n_needed = required_sample_size(BASELINE_CONVERSION, MDE_ABS,
                                    alpha=ALPHA, power=POWER)
    print(f"""
Primary metric        : conversion (2-week purchase)
Baseline rate         : {BASELINE_CONVERSION:.1%}
Minimum effect (MDE)  : +{MDE_ABS:.1%} absolute """
          f"""({MDE_ABS / BASELINE_CONVERSION:.0%} relative)
Significance (alpha)  : {ALPHA} two-sided
Power                 : {POWER:.0%}

--> Required sample size: {n_needed:,} per arm
""")

    # -----------------------------------------------------------------------
    # 3. What can the experiment we HAVE actually see?
    # -----------------------------------------------------------------------
    # The Hillstrom test has ~21,300 customers per arm. Feasible sample size
    # is usually fixed by list size and send cadence, so the honest question
    # becomes: what is the smallest effect THIS test can reliably detect?
    n_actual = 21_300
    mde_actual = minimum_detectable_effect(BASELINE_CONVERSION, n_actual,
                                           alpha=ALPHA, power=POWER)
    pw_at_mde = achieved_power(BASELINE_CONVERSION, MDE_ABS, n_actual,
                               alpha=ALPHA)
    print(f"""Actual sample size    : {n_actual:,} per arm
--> Smallest detectable lift at 80% power: +{mde_actual:.2%} absolute
--> Power to detect the +{MDE_ABS:.1%} MDE with this n: {pw_at_mde:.0%}
""")

    # -----------------------------------------------------------------------
    # 4. The quadratic cost of precision (why small tests are false economy)
    # -----------------------------------------------------------------------
    # Halve the effect you want to detect and the required n roughly
    # quadruples: noise shrinks with sqrt(n). Printing the table makes the
    # trade-off concrete for stakeholders asking "can't we test on 5k users?"
    print("Required n per arm vs. effect size (alpha=0.05, power=80%):")
    print(f"  {'MDE (abs)':>10} | {'n per arm':>12}")
    print("  " + "-" * 26)
    for mde in [0.001, 0.002, 0.004, 0.008]:
        n = required_sample_size(BASELINE_CONVERSION, mde)
        print(f"  {mde:>10.1%} | {n:>12,}")

    print("""
Decision rule (committed in advance):
  SHIP an email if its Holm-adjusted p-value < 0.05 on conversion AND the
  95% CI on incremental spend excludes zero at the portfolio level.
  Otherwise: do not ship; report the CI so the 'no' is quantified.
""")


if __name__ == "__main__":
    sys.exit(main())
