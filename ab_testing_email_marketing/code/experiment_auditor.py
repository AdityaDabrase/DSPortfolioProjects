"""
Experiment Auditor: grade any A/B test CSV before trusting its p-values.

The point of this tool: most A/B test write-ups jump straight to "is the
lift significant?". But a p-value is only meaningful if the EXPERIMENT
DESIGN was sound. This auditor runs the checks a mature experimentation
platform (Microsoft ExP, Airbnb ERF, Booking) runs automatically, and
issues a plain-English verdict.

Checks:
  1. Sample Ratio Mismatch  -- did the randomizer deliver the intended split?
  2. Covariate balance      -- do the arms look like the same population?
  3. Power                  -- was the test big enough to see the effect it
                               claims to care about?
  4. Peeking simulation     -- educational A/A demo of why early stopping lies
  5. Verdict                -- effect size, CI, p-value, ship / no-ship / invalid

Usage:
    python experiment_auditor.py <csv> --group-col SEGMENT --outcome-col CONV
        [--control-label NO_EMAIL] [--covariates col1 col2 ...]
        [--expected-ratios 0.5 0.5] [--mde 0.002] [--peek-demo out.png]

Examples (from the code/ directory):
    # The Hillstrom email experiment: clean three-arm design
    python experiment_auditor.py ../data/raw/hillstrom.csv \
        --group-col segment --outcome-col conversion \
        --control-label "No E-Mail" \
        --covariates recency history mens womens newbie channel zip_code

    # The Kaggle ad test: watch the auditor catch the 96/4 split
    python experiment_auditor.py ../data/raw/marketing_ab.csv \
        --group-col "test group" --outcome-col converted \
        --control-label psa
"""

import argparse
import sys

import numpy as np
import pandas as pd

from utils import (achieved_power, balance_check, holm_correction,
                   minimum_detectable_effect, srm_check,
                   two_proportion_ztest)

# Report formatting helpers -- pure cosmetics, no statistics here.
WIDTH = 74
PASS, WARN, FAIL = "[PASS]", "[WARN]", "[FAIL]"


def header(title):
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


# ---------------------------------------------------------------------------
# Check 1: Sample Ratio Mismatch
# ---------------------------------------------------------------------------

def check_srm(df, group_col, expected_ratios):
    """
    Compare observed group sizes to the intended split.

    Why this is check #1: an SRM means users were LOST or MISROUTED in a way
    that correlates with assignment -- e.g. the treatment page crashed for
    some browsers, or bot filtering hit one arm harder. The surviving users
    in each arm are then no longer comparable populations, and no amount of
    downstream statistics can repair that. Microsoft reports ~6% of its
    experiments trip this check.
    """
    header("CHECK 1: SAMPLE RATIO MISMATCH (SRM)")
    counts = df[group_col].value_counts().sort_index()
    result = srm_check(counts.values,
                       expected_ratios if expected_ratios else None)

    ratios_txt = ("equal split assumed" if not expected_ratios
                  else f"expected ratios {expected_ratios}")
    print(f"Groups ({ratios_txt}):")
    for name, obs, exp in zip(counts.index, result["observed"],
                              result["expected"]):
        share = obs / result["observed"].sum()
        print(f"  {name:<20} observed {obs:>9,} ({share:6.1%})   "
              f"expected {exp:>11,.0f}")

    # p < 0.001 is the industry alarm threshold (see utils.srm_check).
    print(f"\nChi-square = {result['chi2']:.2f}, p = {result['p_value']:.3g}")
    if result["pass"]:
        print(f"{PASS} Group sizes are consistent with the intended split.")
    else:
        print(f"{FAIL} Sample Ratio Mismatch detected. The assignment")
        print("       mechanism itself is suspect; downstream p-values")
        print("       cannot be trusted until the cause is found.")
    return result["pass"]


# ---------------------------------------------------------------------------
# Check 2: covariate balance
# ---------------------------------------------------------------------------

def check_balance(df, group_col, covariates):
    """
    Pre-treatment variables must be indistinguishable across arms: they were
    fixed before the coin flip, so any systematic difference means the coin
    flip wasn't fair. This is the randomization audit.
    """
    header("CHECK 2: COVARIATE BALANCE (did randomization work?)")
    if not covariates:
        print(f"{WARN} No pre-treatment covariates supplied -- skipping.")
        print("       (The Kaggle ad dataset ships none: 'total ads' etc.")
        print("       are measured AFTER assignment, so they can't be used")
        print("       to audit the randomization.)")
        return None

    table = balance_check(df, group_col, covariates)
    for _, row in table.iterrows():
        flag = PASS if row["balanced"] else FAIL
        print(f"  {flag} {row['covariate']:<16} {row['test']:<11} "
              f"stat={row['statistic']:>9.3f}  p={row['p_value']:.3f}")

    all_ok = bool(table["balanced"].all())
    if all_ok:
        print(f"\n{PASS} Arms are statistically identical on all "
              "pre-treatment traits.")
    else:
        print(f"\n{FAIL} At least one covariate differs across arms beyond "
              "chance.")
    return all_ok


# ---------------------------------------------------------------------------
# Check 3: power
# ---------------------------------------------------------------------------

def check_power(df, group_col, outcome_col, control_label, mde):
    """
    An underpowered test can't distinguish 'no effect' from 'effect too
    small for this sample'. We evaluate power at a HYPOTHESIZED minimum
    effect of interest (never the observed effect -- post-hoc power computed
    from observed data is circular and meaningless).
    """
    header(f"CHECK 3: POWER (at a hypothesized MDE of +{mde:.2%} absolute)")
    ctrl = df[df[group_col] == control_label]
    p_base = ctrl[outcome_col].mean()
    n_smallest = df[group_col].value_counts().min()

    power = achieved_power(p_base, mde, n_smallest)
    mde_80 = minimum_detectable_effect(p_base, n_smallest)

    print(f"Baseline ({control_label}) rate : {p_base:.3%}")
    print(f"Smallest arm size          : {n_smallest:,}")
    print(f"Power to detect +{mde:.2%}    : {power:.0%}")
    print(f"Smallest lift visible at 80% power: +{mde_80:.3%} absolute")

    ok = power >= 0.80
    if ok:
        print(f"{PASS} Adequately powered for the effect size of interest.")
    elif power >= 0.60:
        print(f"{WARN} Marginally powered ({power:.0%}). A null result here "
              "is weak evidence of no effect.")
    else:
        print(f"{FAIL} Underpowered ({power:.0%}). A 'not significant' "
              "outcome from this test is uninformative.")
    return ok


# ---------------------------------------------------------------------------
# Check 4: the peeking demonstration (educational, simulation-based)
# ---------------------------------------------------------------------------

def peeking_demo(save_path=None, seed=7):
    """
    Simulate an A/A test: two arms drawn from the SAME distribution, so any
    'significant' result is by construction a false positive. We then compute
    the p-value repeatedly as data accumulates -- exactly what a stakeholder
    does when they refresh the dashboard every day and ask to stop the test
    the moment the p-value dips under 0.05.

    The punchline: under continuous monitoring, the p-value is nearly
    guaranteed to cross 0.05 at SOME point even with zero true effect.
    'Stop when significant' converts a 5% error rate into a ~30-50% one.
    """
    header("CHECK 4: PEEKING DEMO (why early stopping lies)")
    rng = np.random.default_rng(seed)
    n_total, p_true = 40_000, 0.006  # same baseline as our email test
    checks = np.arange(2_000, n_total + 1, 1_000)

    a = rng.random(n_total) < p_true   # arm A: conversion coin-flips
    b = rng.random(n_total) < p_true   # arm B: IDENTICAL distribution

    p_over_time, crossed = [], []
    for n in checks:
        res = two_proportion_ztest(a[:n].sum(), n, b[:n].sum(), n)
        p_over_time.append(res.p_value)
        if res.p_value < 0.05:
            crossed.append(n)

    if crossed:
        print(f"This A/A test (NO true difference) showed p < 0.05 at "
              f"{len(crossed)} of {len(checks)} looks,")
        print(f"first at n={crossed[0]:,}. Anyone who stopped there would "
              "have shipped a mirage.")
    else:
        print(f"This particular seed never crossed 0.05 across {len(checks)}"
              " looks -- rerun with more seeds and ~30-50% of A/A tests do.")
    print("Rule: fix the sample size in advance (or use sequential methods "
          "designed for peeking).")

    if save_path:
        import matplotlib
        matplotlib.use("Agg")  # no display needed; we only save a file
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(checks, p_over_time, lw=1.8)
        ax.axhline(0.05, color="crimson", ls="--", lw=1.2,
                   label="p = 0.05 threshold")
        if crossed:
            ax.scatter([crossed[0]],
                       [p_over_time[list(checks).index(crossed[0])]],
                       color="crimson", zorder=5,
                       label=f"first false 'winner' (n={crossed[0]:,})")
        ax.set_xlabel("Sample size so far (per arm)")
        ax.set_ylabel("p-value at that moment")
        ax.set_title("A/A test (no true effect): p-value under continuous "
                     "peeking")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"Chart saved to {save_path}")


# ---------------------------------------------------------------------------
# Check 5: verdict
# ---------------------------------------------------------------------------

def verdict(df, group_col, outcome_col, control_label, design_ok):
    """
    Only now do we look at the outcome -- and every treatment-vs-control
    p-value is Holm-corrected, because testing multiple arms against one
    control multiplies the false-positive budget.
    """
    header("CHECK 5: VERDICT")
    treatments = [g for g in df[group_col].unique() if g != control_label]
    ctrl = df[df[group_col] == control_label]

    results = []
    for t in treatments:
        arm = df[df[group_col] == t]
        res = two_proportion_ztest(
            int(arm[outcome_col].sum()), len(arm),
            int(ctrl[outcome_col].sum()), len(ctrl))
        results.append((t, res))

    adjusted, reject = holm_correction([r.p_value for _, r in results])

    for (name, res), p_adj, sig in zip(results, adjusted, reject):
        print(f"\n  {name}  vs  {control_label}")
        print(f"    rates      : {res.p_treat:.3%} vs {res.p_ctrl:.3%}")
        print(f"    lift       : {res.abs_lift:+.3%} absolute "
              f"({res.rel_lift:+.0%} relative)")
        print(f"    95% CI     : [{res.ci_low:+.3%}, {res.ci_high:+.3%}]")
        print(f"    p-value    : {res.p_value:.2e}  "
              f"(Holm-adjusted: {p_adj:.2e})")
        print(f"    conclusion : "
              + ("statistically significant" if sig else "not significant"))

    print("\n" + "-" * WIDTH)
    if not design_ok:
        print(f"{FAIL} OVERALL: TEST INVALID OR COMPROMISED.")
        print("   Design checks failed. The numbers above are reported for")
        print("   completeness but should NOT drive a shipping decision.")
    elif any(reject):
        winners = [name for (name, _), sig in zip(results, reject) if sig]
        print(f"{PASS} OVERALL: SHIP -- significant winner(s): "
              f"{', '.join(winners)}.")
    else:
        print(f"{WARN} OVERALL: DO NOT SHIP -- no arm beat control beyond "
              "chance at this sample size.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("csv")
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--outcome-col", required=True)
    ap.add_argument("--control-label", required=True)
    ap.add_argument("--covariates", nargs="*", default=[])
    ap.add_argument("--expected-ratios", nargs="*", type=float, default=[],
                    help="Intended split, e.g. 0.5 0.5. Default: equal.")
    ap.add_argument("--mde", type=float, default=0.002,
                    help="Hypothesized minimum effect for the power check.")
    ap.add_argument("--peek-demo", default=None, metavar="PNG",
                    help="Also run the peeking simulation, save chart here.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Booleans (Kaggle's 'converted' column) -> 0/1 so the math is uniform.
    if df[args.outcome_col].dtype == bool:
        df[args.outcome_col] = df[args.outcome_col].astype(int)

    print("#" * WIDTH)
    print(f"# EXPERIMENT AUDIT: {args.csv}")
    print(f"# groups='{args.group_col}'  outcome='{args.outcome_col}'  "
          f"control='{args.control_label}'")
    print("#" * WIDTH)

    srm_ok = check_srm(df, args.group_col,
                       args.expected_ratios or None)
    bal_ok = check_balance(df, args.group_col, args.covariates)
    pow_ok = check_power(df, args.group_col, args.outcome_col,
                         args.control_label, args.mde)
    if args.peek_demo:
        peeking_demo(save_path=args.peek_demo)

    # Balance "None" (not checkable) doesn't invalidate; SRM failure does.
    design_ok = srm_ok and (bal_ok is not False)
    verdict(df, args.group_col, args.outcome_col, args.control_label,
            design_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
