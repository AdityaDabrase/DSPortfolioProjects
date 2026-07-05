"""
Shared statistical helpers for the A/B testing project.

Every function here returns plain floats/dicts rather than printing, so the
notebooks can format results however they like. Each function's docstring
explains WHY the method is appropriate, not just what it computes -- the goal
is that you could defend any of these choices in an interview.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Two-proportion z-test (for binary outcomes: visit, conversion)
# ---------------------------------------------------------------------------

@dataclass
class ProportionTestResult:
    """Everything a stakeholder-facing report needs about one comparison."""

    p_treat: float          # observed rate in treatment
    p_ctrl: float           # observed rate in control
    abs_lift: float         # p_treat - p_ctrl (percentage points / 100)
    rel_lift: float         # abs_lift / p_ctrl ("we grew conversion by X%")
    z_stat: float
    p_value: float          # two-sided
    ci_low: float           # 95% CI on the absolute lift
    ci_high: float
    n_treat: int
    n_ctrl: int


def two_proportion_ztest(success_treat, n_treat, success_ctrl, n_ctrl,
                         alpha=0.05):
    """
    Compare two conversion rates.

    Why a z-test? With n in the tens of thousands and outcomes that are 0/1,
    the sampling distribution of the difference in proportions is extremely
    well approximated by a normal distribution (Central Limit Theorem). The
    common rule of thumb -- at least ~10 successes and 10 failures per arm --
    is exceeded by orders of magnitude here.

    Two details that are easy to get wrong:

    1.  The HYPOTHESIS TEST uses a *pooled* standard error. Under the null
        hypothesis the two groups share one true rate, so we estimate that
        single rate from the combined data. Using an unpooled SE for the test
        slightly mis-calibrates the Type I error.

    2.  The CONFIDENCE INTERVAL uses an *unpooled* standard error. The CI
        describes the difference as it actually is (two distinct rates), not
        under the null. Mixing these up is a classic mistake.
    """
    p1 = success_treat / n_treat
    p0 = success_ctrl / n_ctrl
    diff = p1 - p0

    # Pooled SE: assumes one common rate, as the null hypothesis states.
    p_pool = (success_treat + success_ctrl) / (n_treat + n_ctrl)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_treat + 1 / n_ctrl))
    z = diff / se_pooled
    p_value = 2 * stats.norm.sf(abs(z))  # two-sided: effect could go either way

    # Unpooled SE for the CI: each group keeps its own variance.
    se_unpooled = np.sqrt(p1 * (1 - p1) / n_treat + p0 * (1 - p0) / n_ctrl)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return ProportionTestResult(
        p_treat=p1,
        p_ctrl=p0,
        abs_lift=diff,
        rel_lift=diff / p0 if p0 > 0 else np.nan,
        z_stat=z,
        p_value=p_value,
        ci_low=diff - z_crit * se_unpooled,
        ci_high=diff + z_crit * se_unpooled,
        n_treat=n_treat,
        n_ctrl=n_ctrl,
    )


# ---------------------------------------------------------------------------
# Welch's t-test (for continuous outcomes: spend)
# ---------------------------------------------------------------------------

def welch_ttest(x_treat, x_ctrl, alpha=0.05):
    """
    Compare mean spend between two groups.

    Why Welch and not Student's t? Student's t assumes equal variances across
    groups. Revenue data never satisfies that: the treated group has both more
    buyers AND a different spend distribution among buyers. Welch's version
    estimates each group's variance separately and adjusts the degrees of
    freedom, so it stays valid under unequal variances. There is essentially
    no downside to defaulting to Welch.

    Why is a t-test valid at all on zero-inflated, skewed revenue? Because we
    are comparing MEANS of large samples: the CLT applies to the sampling
    distribution of the mean even when the underlying data is wildly
    non-normal. With n > 21,000 per arm we are comfortably in CLT territory.
    (We still cross-check with a bootstrap in the analysis notebook, which
    makes no distributional assumptions at all.)
    """
    x_treat = np.asarray(x_treat, dtype=float)
    x_ctrl = np.asarray(x_ctrl, dtype=float)

    t_stat, p_value = stats.ttest_ind(x_treat, x_ctrl, equal_var=False)

    diff = x_treat.mean() - x_ctrl.mean()
    se = np.sqrt(x_treat.var(ddof=1) / len(x_treat)
                 + x_ctrl.var(ddof=1) / len(x_ctrl))

    # Welch-Satterthwaite degrees of freedom -- with samples this large the
    # t distribution is indistinguishable from normal, but we do it properly.
    v1 = x_treat.var(ddof=1) / len(x_treat)
    v0 = x_ctrl.var(ddof=1) / len(x_ctrl)
    dof = (v1 + v0) ** 2 / (v1 ** 2 / (len(x_treat) - 1)
                            + v0 ** 2 / (len(x_ctrl) - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    return {
        "mean_treat": x_treat.mean(),
        "mean_ctrl": x_ctrl.mean(),
        "diff": diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
        "dof": dof,
    }


def bootstrap_diff_ci(x_treat, x_ctrl, n_boot=10_000, alpha=0.05, seed=42):
    """
    Bootstrap CI for the difference in means -- our assumption-free cross-check.

    The idea: if we could rerun the experiment thousands of times, how much
    would the measured difference wobble? We can't rerun it, but resampling
    each group WITH REPLACEMENT simulates exactly that. No normality
    assumption, no variance formula -- just brute-force resampling.

    If the bootstrap CI and the Welch CI agree (they will here), that is
    strong evidence the parametric shortcut was safe. If they disagreed, we
    would trust the bootstrap.
    """
    rng = np.random.default_rng(seed)
    x_treat = np.asarray(x_treat, dtype=float)
    x_ctrl = np.asarray(x_ctrl, dtype=float)

    # Vectorized resampling: one (n_boot x n) index matrix per group.
    boot_treat = rng.choice(x_treat, size=(n_boot, len(x_treat)),
                            replace=True).mean(axis=1)
    boot_ctrl = rng.choice(x_ctrl, size=(n_boot, len(x_ctrl)),
                           replace=True).mean(axis=1)
    boot_diffs = boot_treat - boot_ctrl

    lo, hi = np.percentile(boot_diffs, [100 * alpha / 2,
                                        100 * (1 - alpha / 2)])
    return {"ci_low": lo, "ci_high": hi, "boot_diffs": boot_diffs}


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def holm_correction(p_values, alpha=0.05):
    """
    Holm-Bonferroni step-down correction.

    Why we need this: we test TWO treatments against control across THREE
    metrics. Each test carries its own 5% false-positive risk; run six tests
    and the chance that at least one is a fluke is ~26%. Corrections shrink
    that family-wise error back to 5%.

    Why Holm over plain Bonferroni: Holm is uniformly more powerful (it
    rejects everything Bonferroni rejects, and sometimes more) while giving
    the exact same family-wise error guarantee. There is no statistical
    reason to prefer plain Bonferroni.

    Mechanics: sort p-values ascending; compare the smallest to alpha/m, the
    next to alpha/(m-1), and so on; stop at the first failure.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)

    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        # Each p-value is scaled by the number of hypotheses still "alive".
        adj = min((m - rank) * p[idx], 1.0)
        # Enforce monotonicity: an adjusted p can't be smaller than the one
        # before it, otherwise rejection decisions would be inconsistent.
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    return adjusted, adjusted < alpha


# ---------------------------------------------------------------------------
# Design: power / sample size / MDE (used by 02_experiment_design.py)
# ---------------------------------------------------------------------------

def required_sample_size(p_base, mde_abs, alpha=0.05, power=0.80):
    """
    Sample size PER ARM to detect an absolute lift of `mde_abs` over a
    baseline rate `p_base`, with two-sided significance `alpha` and the
    requested power.

    Standard normal-approximation formula:

        n = (z_{1-a/2} * sqrt(2 p̄ q̄) + z_{power} * sqrt(p0 q0 + p1 q1))^2
            -----------------------------------------------------------
                                (p1 - p0)^2

    Intuition for the shape of the formula: halving the effect you want to
    detect QUADRUPLES the required sample -- noise shrinks with sqrt(n), so
    resolution improves slowly. This is why "just run a smaller test" is
    usually a false economy.
    """
    p0 = p_base
    p1 = p_base + mde_abs
    p_bar = (p0 + p1) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    numerator = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
                 + z_power * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(np.ceil(numerator / (p1 - p0) ** 2))


def minimum_detectable_effect(p_base, n_per_arm, alpha=0.05, power=0.80):
    """
    Invert the sample-size relationship: given the n we actually have, what
    is the smallest absolute lift we could reliably detect?

    This is the honest way to interpret a "null" result. If the MDE at our
    sample size is 0.4pp and we measured +0.2pp (not significant), the right
    conclusion is "the test cannot see effects this small", NOT "there is no
    effect". Solved numerically because the MDE appears on both sides of the
    power equation.
    """
    from scipy.optimize import brentq

    def power_gap(mde):
        return achieved_power(p_base, mde, n_per_arm, alpha) - power

    # Bracket: effects between 0.001pp and 50pp cover any realistic case.
    return brentq(power_gap, 1e-5, 0.5)


def achieved_power(p_base, effect_abs, n_per_arm, alpha=0.05):
    """
    Probability that a test of this size detects a true effect of
    `effect_abs`. Used to answer "was this experiment big enough?" after
    the fact (with a HYPOTHESIZED effect, never the observed one -- post-hoc
    power computed from the observed effect is a well-known statistical sin:
    it is just a transformation of the p-value and adds no information).
    """
    p0 = p_base
    p1 = p_base + effect_abs
    p_bar = (p0 + p1) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
    se_alt = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n_per_arm)

    # Power = P(|Z| > z_crit | true effect). The second term is negligible
    # for positive effects but included for correctness.
    z_shift = (abs(p1 - p0) - z_alpha * se_null) / se_alt
    z_shift_other = (-abs(p1 - p0) - z_alpha * se_null) / se_alt
    return stats.norm.cdf(z_shift) + stats.norm.cdf(z_shift_other)


# ---------------------------------------------------------------------------
# Experiment-validity checks (shared with the auditor)
# ---------------------------------------------------------------------------

def srm_check(observed_counts, expected_ratios=None):
    """
    Sample Ratio Mismatch check.

    If the randomizer intended a 50/50 (or here 1/3-1/3-1/3) split, the
    observed group sizes should differ from that only by chance. A chi-square
    goodness-of-fit test quantifies "only by chance". A tiny p-value means
    the assignment mechanism itself is broken (bugs in bucketing, bot
    filtering applied to one arm, redirect losses...) -- and if assignment is
    broken, every downstream p-value is meaningless.

    Industry convention (Microsoft, Airbnb, Booking) is to alarm at
    p < 0.001 rather than 0.05: group sizes are checked constantly, so a
    stricter threshold avoids false alarms while still catching real bugs.
    """
    observed = np.asarray(observed_counts, dtype=float)
    n = observed.sum()
    k = len(observed)

    if expected_ratios is None:
        expected_ratios = np.ones(k) / k  # default: equal split intended
    expected = n * np.asarray(expected_ratios, dtype=float)

    chi2, p_value = stats.chisquare(observed, expected)
    return {"chi2": chi2, "p_value": p_value,
            "observed": observed.astype(int), "expected": expected,
            "pass": p_value >= 0.001}


def balance_check(df, group_col, covariates, alpha=0.001):
    """
    Covariate balance: pre-treatment variables should be statistically
    indistinguishable across arms, because assignment happened AFTER those
    variables were fixed. Any imbalance beyond chance means the "random"
    assignment correlated with customer traits, i.e. the comparison is
    confounded.

    Numeric covariates -> one-way ANOVA across arms.
    Categorical covariates -> chi-square test of independence.

    Same strict alpha as SRM, and for the same reason: with many covariates
    a 0.05 threshold would flag one of twenty by pure chance.
    """
    rows = []
    groups = df[group_col].unique()

    for cov in covariates:
        if pd.api.types.is_numeric_dtype(df[cov]):
            # Cast to float64: f_oneway's sum-of-squares can overflow on
            # large int64 columns, producing spurious RuntimeWarnings.
            samples = [df.loc[df[group_col] == g, cov].dropna()
                         .astype(float) for g in groups]
            stat, p = stats.f_oneway(*samples)
            test_name = "ANOVA F"
        else:
            contingency = pd.crosstab(df[cov], df[group_col])
            stat, p, _, _ = stats.chi2_contingency(contingency)
            test_name = "chi-square"
        rows.append({"covariate": cov, "test": test_name,
                     "statistic": stat, "p_value": p,
                     "balanced": p >= alpha})

    return pd.DataFrame(rows)
