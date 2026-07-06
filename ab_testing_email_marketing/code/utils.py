"""Shared statistics helpers for the A/B testing project.

Everything in this file is deliberately written from first principles
(rather than calling one black-box library function) so the math is
visible and defensible in front of a technical reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# The two-week outcome window and three arms of the Hillstrom test.
CONTROL_LABEL = "No E-Mail"
TREATMENT_LABELS = ["Mens E-Mail", "Womens E-Mail"]


# ---------------------------------------------------------------------------
# Core hypothesis tests
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Container so every test in the project reports the same fields.

    Reporting an effect size + confidence interval (not just a p-value)
    is a deliberate choice: stakeholders make decisions on magnitudes,
    and a p-value alone says nothing about how big the effect is.
    """

    metric: str
    comparison: str
    effect: float          # absolute difference (treatment - control)
    relative_lift: float   # effect / control mean, for "X% lift" statements
    ci_low: float
    ci_high: float
    p_value: float
    control_mean: float
    treatment_mean: float
    n_control: int
    n_treatment: int

    def summary(self) -> str:
        return (
            f"{self.metric} | {self.comparison}: "
            f"{self.treatment_mean:.4f} vs {self.control_mean:.4f} "
            f"(diff {self.effect:+.4f}, {self.relative_lift:+.1%} relative), "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"p={self.p_value:.2e}"
        )


def two_proportion_ztest(
    successes_t: int, n_t: int, successes_c: int, n_c: int,
    metric: str = "", comparison: str = "",
) -> TestResult:
    """Two-proportion z-test for binary outcomes (visit, conversion).

    Why a z-test: with tens of thousands of observations per arm the
    sampling distribution of a proportion is extremely well approximated
    by a normal (CLT), so the z-test is both exact enough and easy to
    explain. For tiny samples we'd switch to Fisher's exact test.
    """
    p_t = successes_t / n_t
    p_c = successes_c / n_c

    # Pooled proportion under H0 (no difference). The null hypothesis
    # assumes both arms share one true rate, so the standard error for
    # the *test* uses the pooled estimate.
    p_pool = (successes_t + successes_c) / (n_t + n_c)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = (p_t - p_c) / se_pooled
    # Two-sided: we'd want to know about a *negative* effect just as much
    # as a positive one (an email that hurts sales is a business problem).
    p_value = 2 * stats.norm.sf(abs(z))

    # The confidence interval uses the UNpooled standard error because the
    # CI describes the difference itself, not a world where H0 is true.
    se_unpooled = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z_crit = stats.norm.ppf(0.975)
    diff = p_t - p_c

    return TestResult(
        metric=metric, comparison=comparison,
        effect=diff,
        relative_lift=diff / p_c if p_c > 0 else np.nan,
        ci_low=diff - z_crit * se_unpooled,
        ci_high=diff + z_crit * se_unpooled,
        p_value=p_value,
        control_mean=p_c, treatment_mean=p_t,
        n_control=n_c, n_treatment=n_t,
    )


def welch_ttest(
    treatment: np.ndarray, control: np.ndarray,
    metric: str = "", comparison: str = "",
) -> TestResult:
    """Welch's t-test for continuous outcomes (spend).

    Why Welch and not Student's t: Welch does not assume equal variances
    between arms. Spend data is zero-inflated (most people buy nothing),
    so variances can differ a lot between arms; Welch costs nothing and
    is robust to that. This is the default recommendation in modern
    experimentation platforms.
    """
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    mean_t, mean_c = treatment.mean(), control.mean()
    diff = mean_t - mean_c
    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    # Welch-Satterthwaite degrees of freedom (what makes Welch "Welch").
    df = se**4 / (
        (treatment.var(ddof=1) / len(treatment)) ** 2 / (len(treatment) - 1)
        + (control.var(ddof=1) / len(control)) ** 2 / (len(control) - 1)
    )
    t_crit = stats.t.ppf(0.975, df)

    return TestResult(
        metric=metric, comparison=comparison,
        effect=diff,
        relative_lift=diff / mean_c if mean_c > 0 else np.nan,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        p_value=p_value,
        control_mean=mean_c, treatment_mean=mean_t,
        n_control=len(control), n_treatment=len(treatment),
    )


def bootstrap_diff_means(
    treatment: np.ndarray, control: np.ndarray,
    n_boot: int = 10_000, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for a difference in means. Returns (diff, ci_low, ci_high).

    Why bootstrap alongside the t-test: spend is heavily skewed (a few
    customers spend $400+, most spend $0), and skew is the classic case
    where people distrust normal-theory intervals. The bootstrap makes no
    distributional assumption - we resample the data we actually observed.
    If bootstrap and Welch agree (they do here, thanks to n=21k+ per arm),
    that agreement itself is evidence the parametric result is trustworthy.
    """
    rng = np.random.default_rng(seed)
    # Resample each arm independently WITH replacement - this simulates
    # "what other datasets could this experiment have produced?"
    boot_t = rng.choice(treatment, size=(n_boot, len(treatment)), replace=True).mean(axis=1)
    boot_c = rng.choice(control, size=(n_boot, len(control)), replace=True).mean(axis=1)
    diffs = boot_t - boot_c
    return (
        float(treatment.mean() - control.mean()),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values.

    Why we need this: with 3 arms x 3 metrics we run many tests. Each test
    carries a 5% false-positive risk, so running nine of them pushes the
    chance of at least one false "winner" toward ~37%. Holm controls the
    family-wise error rate back to 5% and is uniformly more powerful than
    plain Bonferroni (it never rejects less), so there is no reason to
    prefer Bonferroni.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        # Holm multiplies the smallest p by m, the next by m-1, etc.,
        # and enforces monotonicity so adjusted p-values never decrease.
        adj = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Experiment validity checks (used by the EDA notebook and the auditor)
# ---------------------------------------------------------------------------

def srm_check(observed_counts: dict[str, int], expected_shares: dict[str, float] | None = None):
    """Sample Ratio Mismatch check via chi-square goodness of fit.

    An SRM means the traffic split you got differs from the split you
    designed - almost always a symptom of a broken assignment pipeline
    (bot filtering one arm, redirects dropping users, logging bugs).
    Microsoft reports ~6% of their experiments hit SRM; an SRM'd test
    should be thrown away, not analyzed. This is check #1 for a reason.

    If expected_shares is None we assume the arms were meant to be equal.
    Returns (chi2, p_value, verdict_string).
    """
    labels = list(observed_counts.keys())
    observed = np.array([observed_counts[label] for label in labels], dtype=float)
    total = observed.sum()
    if expected_shares is None:
        expected = np.full(len(labels), total / len(labels))
    else:
        expected = np.array([expected_shares[label] * total for label in labels])

    chi2, p = stats.chisquare(observed, expected)
    # p < 0.001 is the conventional SRM alarm threshold: we deliberately use
    # a stricter cutoff than 0.05 because with big samples the check is very
    # sensitive, and a false SRM alarm kills a healthy experiment.
    verdict = "PASS" if p >= 0.001 else "FAIL - investigate assignment pipeline before analyzing"
    return chi2, p, verdict


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """SMD between two groups for a numeric covariate.

    Why SMD instead of a t-test for balance checks: with 20k+ per arm a
    t-test flags differences that are statistically detectable but far too
    small to matter. SMD measures the *size* of the imbalance in units of
    standard deviations; |SMD| < 0.1 is the standard 'balanced' threshold
    from the causal-inference literature.
    """
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0
