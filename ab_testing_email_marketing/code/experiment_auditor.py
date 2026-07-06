"""Experiment Auditor: automated health report for any A/B test CSV.

Point it at a CSV, name the group column and the outcome column, and it
grades the experiment the way a good data scientist would before trusting
its result:

  1. Sample Ratio Mismatch  - did the traffic split match the design?
  2. Covariate balance      - did randomization actually mix the groups?
  3. Statistical power      - was the test big enough to see anything?
  4. Peeking simulation     - what early stopping would have done here
  5. Verdict                - effect size, CI, p-value, plain-English call

Usage (from the project root):
    python code/experiment_auditor.py data/raw/hillstrom.csv \
        --group segment --outcome conversion --control "No E-Mail"

    python code/experiment_auditor.py data/raw/marketing_ab.csv \
        --group "test group" --outcome converted --control psa \
        --expected-share ad=0.96 --expected-share psa=0.04

The tool is industry-agnostic on purpose: nothing in it knows about
email or ads. Any randomized test with a group column and a binary
outcome column can be audited.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Allow running as a script from anywhere: put this file's folder on the
# path so `import utils` works without packaging ceremony.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_design import mde_two_proportions, power_achieved
from utils import srm_check, standardized_mean_difference, two_proportion_ztest

SEPARATOR = "-" * 68


# ---------------------------------------------------------------------------
# Check 1: Sample Ratio Mismatch
# ---------------------------------------------------------------------------

def check_srm(df: pd.DataFrame, group_col: str, expected_shares: dict[str, float] | None):
    """SRM is the single highest-value automated check in experimentation.

    If users were supposed to be split 50/50 (or 96/4) and the observed
    counts are statistically incompatible with that split, something in
    the assignment pipeline is broken and EVERY downstream number is
    suspect. Microsoft/LinkedIn/Airbnb all run this check automatically.
    """
    counts = df[group_col].value_counts().to_dict()
    chi2, p, verdict = srm_check(counts, expected_shares)

    print(f"[1] SAMPLE RATIO MISMATCH CHECK  ->  {verdict.split(' ')[0]}")
    for grp, n in counts.items():
        share = n / len(df)
        expected = expected_shares.get(grp, 1 / len(counts)) if expected_shares else 1 / len(counts)
        print(f"    {grp:<20} n={n:>8,}  observed {share:6.2%}  vs designed {expected:6.2%}")
    print(f"    chi-square p-value = {p:.3g}  (alarm threshold: p < 0.001)")
    if "FAIL" in verdict:
        print("    !! Split is incompatible with the design. Do not trust this")
        print("    !! experiment until the assignment mechanism is explained.")
    else:
        print("    Split is consistent with the designed allocation.")
    print(SEPARATOR)
    return "FAIL" not in verdict


# ---------------------------------------------------------------------------
# Check 2: Covariate balance
# ---------------------------------------------------------------------------

def check_balance(df: pd.DataFrame, group_col: str, outcome_col: str,
                  post_treatment_cols: list[str] | None = None):
    """Randomization's whole job is making groups comparable BEFORE treatment.

    We compare every pre-treatment numeric column across groups using the
    standardized mean difference. |SMD| < 0.1 is the standard threshold:
    below it, differences are too small to meaningfully confound results.
    (We use SMD, not t-tests, because at large n a t-test flags trivial
    differences - see utils.standardized_mean_difference.)

    IMPORTANT: only PRE-treatment variables belong here. Outcome-like
    columns (visits, spend, ...) measured after treatment SHOULD differ
    between arms if the treatment works - flagging them as 'imbalance'
    would be a false alarm. That is why the caller must declare them via
    `post_treatment_cols`.
    """
    excluded = set(post_treatment_cols or []) | {outcome_col}
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in excluded and df[c].nunique() > 1
    ]
    groups = df[group_col].unique()
    print("[2] COVARIATE BALANCE CHECK (pre-treatment variables)")

    if not numeric_cols:
        print("    No pre-treatment numeric covariates found - skipping.")
        print("    (Balance can only be assessed on variables measured before treatment.)")
        print(SEPARATOR)
        return True

    all_ok = True
    # Compare every pair of groups on every covariate; report the worst SMD.
    for col in numeric_cols:
        worst = 0.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a = df.loc[df[group_col] == groups[i], col].to_numpy(dtype=float)
                b = df.loc[df[group_col] == groups[j], col].to_numpy(dtype=float)
                worst = max(worst, abs(standardized_mean_difference(a, b)))
        status = "balanced" if worst < 0.1 else "IMBALANCED"
        if worst >= 0.1:
            all_ok = False
        print(f"    {col:<20} worst |SMD| = {worst:.4f}  [{status}]")
    print("    Threshold: |SMD| < 0.10 (Austin 2009, standard in causal inference).")
    print(SEPARATOR)
    return all_ok


# ---------------------------------------------------------------------------
# Check 3: Power
# ---------------------------------------------------------------------------

def check_power(df: pd.DataFrame, group_col: str, outcome_col: str, control_label: str):
    """Was this experiment big enough for its own conclusion?

    We compute the minimum detectable effect at the smallest arm's size.
    If the observed effect is well below the MDE and 'not significant',
    the test never had a chance - that's an underpowered test, not
    evidence of no effect.
    """
    control = df[df[group_col] == control_label]
    baseline = control[outcome_col].mean()
    n_smallest = df[group_col].value_counts().min()

    mde = mde_two_proportions(baseline, int(n_smallest))
    print("[3] POWER CHECK")
    print(f"    Baseline (control) rate      : {baseline:.3%}")
    print(f"    Smallest arm                 : n={n_smallest:,}")
    print(f"    Minimum detectable effect    : {mde*100:.3f}pp absolute "
          f"({mde/baseline:.0%} relative) at 80% power")

    # Compare against what was actually observed in the best arm.
    rates = df.groupby(group_col)[outcome_col].mean()
    biggest_lift = (rates - baseline).drop(index=control_label, errors="ignore").abs().max()
    pw = power_achieved(baseline, float(biggest_lift), int(n_smallest))
    print(f"    Largest observed lift        : {biggest_lift*100:.3f}pp "
          f"-> power to detect it = {pw:.0%}")
    ok = pw >= 0.5
    if not ok:
        print("    !! Underpowered for effects of the observed size: a null result")
        print("    !! here would be uninformative, not reassuring.")
    print(SEPARATOR)
    return ok


# ---------------------------------------------------------------------------
# Check 4: Peeking simulation
# ---------------------------------------------------------------------------

def check_peeking(df: pd.DataFrame, group_col: str, outcome_col: str,
                  control_label: str, treatment_label: str,
                  n_checkpoints: int = 100, seed: int = 7,
                  save_plot: str | None = None):
    """Replay THIS experiment's data arriving over time and record the
    p-value at each interim look.

    The point: if a team 'peeks' and stops the moment p < 0.05, they are
    running many tests, not one, and the real false-positive rate can
    triple or worse. We report how many interim looks dipped below 0.05 -
    on a healthy experiment with a real effect the p-value dives and stays
    down; on a null experiment it wanders across the line by luck.
    """
    rng = np.random.default_rng(seed)
    sub = df[df[group_col].isin([control_label, treatment_label])]
    # Shuffle to simulate random arrival order (the CSV has no timestamps).
    sub = sub.sample(frac=1, random_state=rng.integers(1e9)).reset_index(drop=True)

    is_treat = (sub[group_col] == treatment_label).to_numpy()
    outcome = sub[outcome_col].to_numpy(dtype=float)

    checkpoints = np.linspace(len(sub) * 0.02, len(sub), n_checkpoints, dtype=int)
    p_trajectory = []
    for n in checkpoints:
        t_mask = is_treat[:n]
        s_t, n_t = outcome[:n][t_mask].sum(), t_mask.sum()
        s_c, n_c = outcome[:n][~t_mask].sum(), (~t_mask).sum()
        if min(n_t, n_c) < 30 or s_t + s_c == 0:  # too early to test meaningfully
            p_trajectory.append(1.0)
            continue
        res = two_proportion_ztest(int(s_t), int(n_t), int(s_c), int(n_c))
        p_trajectory.append(res.p_value)

    p_trajectory = np.array(p_trajectory)
    dips = int((p_trajectory < 0.05).sum())
    final_p = p_trajectory[-1]

    print(f"[4] PEEKING SIMULATION ({treatment_label} vs {control_label})")
    print(f"    Interim looks below p<0.05   : {dips}/{n_checkpoints}")
    print(f"    Final p-value                : {final_p:.3g}")
    if final_p < 0.05 and dips > n_checkpoints * 0.5:
        print("    Signal is stable: significant early and stayed significant.")
    elif final_p >= 0.05 and dips > 0:
        print("    !! A peeking analyst could have declared a false winner at an")
        print("    !! interim look, even though the final result is not significant.")
    else:
        print("    No early-stopping hazard evident in this replay.")

    if save_plot:
        # Log scale because interesting p-values span many orders of magnitude;
        # the red line is the naive stopping rule a "peeking" team would use.
        import matplotlib
        matplotlib.use("Agg")  # no display needed when run from the CLI
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(checkpoints, np.maximum(p_trajectory, 1e-16), lw=1.8, color="steelblue")
        ax.axhline(0.05, color="crimson", ls="--", label="p = 0.05 (naive stop rule)")
        ax.set_yscale("log")
        ax.set_xlabel("customers observed so far")
        ax.set_ylabel("p-value at interim look (log scale)")
        ax.set_title(f"Peeking replay: {treatment_label} vs {control_label} "
                     f"({dips}/{n_checkpoints} looks below 0.05)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_plot, dpi=150)
        plt.close(fig)
        print(f"    Plot saved to {save_plot}")
    print(SEPARATOR)
    return p_trajectory, checkpoints


# ---------------------------------------------------------------------------
# Check 5: Verdict
# ---------------------------------------------------------------------------

def check_verdict(df: pd.DataFrame, group_col: str, outcome_col: str,
                  control_label: str, design_ok: bool):
    """The actual answer, gated by the design checks above.

    Order matters: effect sizes are only meaningful if checks 1-2 passed.
    A beautiful p-value on an SRM'd experiment is a beautifully precise
    measurement of a broken pipeline.
    """
    control = df[df[group_col] == control_label]
    s_c, n_c = int(control[outcome_col].sum()), len(control)

    print("[5] VERDICT")
    results = []
    for label in [g for g in df[group_col].unique() if g != control_label]:
        arm = df[df[group_col] == label]
        res = two_proportion_ztest(
            int(arm[outcome_col].sum()), len(arm), s_c, n_c,
            metric=outcome_col, comparison=f"{label} vs {control_label}",
        )
        results.append(res)
        sig = "significant" if res.p_value < 0.05 else "not significant"
        print(f"    {res.comparison}")
        print(f"      lift {res.effect*100:+.3f}pp ({res.relative_lift:+.1%} relative), "
              f"95% CI [{res.ci_low*100:+.3f}, {res.ci_high*100:+.3f}]pp, "
              f"p={res.p_value:.2e} ({sig})")

    print()
    if not design_ok:
        print("    RECOMMENDATION: TEST INVALID. Design checks failed - fix the")
        print("    assignment pipeline and rerun before making any decision.")
    else:
        winners = [r for r in results if r.p_value < 0.05 and r.effect > 0]
        if winners:
            best = max(winners, key=lambda r: r.effect)
            print(f"    RECOMMENDATION: SHIP. '{best.comparison.split(' vs ')[0]}' shows a")
            print(f"    significant lift on a validly designed experiment.")
        else:
            print("    RECOMMENDATION: DO NOT SHIP. No arm shows a significant")
            print("    improvement (check the power section before calling it a null).")
    print(SEPARATOR)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def audit(csv_path: str, group_col: str, outcome_col: str, control_label: str,
          expected_shares: dict[str, float] | None = None,
          post_treatment_cols: list[str] | None = None,
          ignore_cols: list[str] | None = None,
          peeking_plot: str | None = None):
    df = pd.read_csv(csv_path)
    # Drop identifier columns and pandas' "Unnamed: 0" index artifact:
    # IDs are numeric but carry no covariate meaning, so leaving them in
    # would add noise rows to the balance check.
    drop = [c for c in df.columns if c.startswith("Unnamed")] + list(ignore_cols or [])
    df = df.drop(columns=[c for c in drop if c in df.columns])
    # Normalize boolean-ish outcomes (True/False strings, bools) to 0/1 so
    # the same code paths work on any dataset.
    if df[outcome_col].dtype == bool or df[outcome_col].dtype == object:
        df[outcome_col] = (
            df[outcome_col].astype(str).str.lower().map({"true": 1, "false": 0})
            .fillna(pd.to_numeric(df[outcome_col], errors="coerce"))
        )

    print("=" * 68)
    print(f"EXPERIMENT AUDIT: {Path(csv_path).name}")
    print(f"groups='{group_col}'  outcome='{outcome_col}'  control='{control_label}'")
    print("=" * 68)

    srm_ok = check_srm(df, group_col, expected_shares)
    balance_ok = check_balance(df, group_col, outcome_col, post_treatment_cols)
    check_power(df, group_col, outcome_col, control_label)

    # Peek-replay the largest treatment arm against control.
    treat_labels = [g for g in df[group_col].unique() if g != control_label]
    biggest = df[df[group_col].isin(treat_labels)][group_col].value_counts().idxmax()
    check_peeking(df, group_col, outcome_col, control_label, biggest,
                  save_plot=peeking_plot)

    check_verdict(df, group_col, outcome_col, control_label, srm_ok and balance_ok)


def parse_expected_share(pairs: list[str]) -> dict[str, float] | None:
    if not pairs:
        return None
    out = {}
    for pair in pairs:
        key, val = pair.rsplit("=", 1)
        out[key] = float(val)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit an A/B test CSV.")
    parser.add_argument("csv", help="path to the experiment CSV")
    parser.add_argument("--group", required=True, help="column with the arm assignment")
    parser.add_argument("--outcome", required=True, help="binary outcome column")
    parser.add_argument("--control", required=True, help="label of the control arm")
    parser.add_argument(
        "--expected-share", action="append", default=[],
        metavar="ARM=SHARE",
        help="designed traffic share per arm, e.g. --expected-share ad=0.96 "
             "(omit for an equal split design)",
    )
    parser.add_argument(
        "--post-treatment", action="append", default=[],
        metavar="COL",
        help="columns measured AFTER treatment (other outcomes); excluded "
             "from the balance check since they legitimately differ by arm",
    )
    parser.add_argument(
        "--ignore", action="append", default=[],
        metavar="COL",
        help="identifier columns to drop entirely (e.g. user id)",
    )
    parser.add_argument(
        "--peeking-plot", default=None, metavar="PATH",
        help="save the peeking-simulation chart to this path",
    )
    args = parser.parse_args()
    audit(args.csv, args.group, args.outcome, args.control,
          parse_expected_share(args.expected_share),
          post_treatment_cols=args.post_treatment,
          ignore_cols=args.ignore,
          peeking_plot=args.peeking_plot)
