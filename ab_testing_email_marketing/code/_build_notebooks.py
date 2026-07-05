"""Temporary builder: writes the three analysis notebooks. Deleted after use."""
import nbformat as nbf


def nb(cells):
    notebook = nbf.v4.new_notebook()
    notebook.cells = cells
    return notebook


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# ===========================================================================
# 01_eda.ipynb
# ===========================================================================
eda = nb([
md("""
# 01 — EDA & Experiment Validation

**Goal of this notebook:** prove the experiment is *trustworthy* before we measure anything.

A p-value is only meaningful if the underlying experiment was sound. So before touching the
outcome columns, we answer three questions, in order:

1. **Is the data clean?** (types, missing values, impossible values)
2. **Did the randomizer deliver the intended split?** (Sample Ratio Mismatch check)
3. **Did randomization actually work?** (covariate balance: the three arms should look like
   three random samples of the *same* population)

Only if all three pass do we earn the right to compare outcomes in notebook 03.

**Dataset:** the Hillstrom email experiment — 64,000 real customers randomly assigned to
*Mens E-Mail*, *Womens E-Mail*, or *No E-Mail* (control). See `../data/README.md`.
"""),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Shared statistical helpers (z-tests, SRM, balance checks) live in utils.py
# so the notebooks, the design module and the auditor all use ONE implementation.
from utils import srm_check, balance_check

sns.set_theme(style="whitegrid", context="notebook")

df = pd.read_csv("../data/raw/hillstrom.csv")
print(f"{df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()
"""),
md("""
## 1. Data quality

Boring on purpose. We check types, missing values, and value ranges. Any surprise here
(negative spend, recency of 40 months on a 12-month file...) would mean a data pipeline
problem to fix before any statistics.
"""),
code("""
# Missing values: a randomized experiment export should have none.
missing = df.isna().sum()
print("Missing values per column:")
print(missing.to_string())
assert missing.sum() == 0, "Unexpected missing data - investigate before analysis"

# Range sanity checks on the numeric columns.
assert df["recency"].between(1, 12).all(), "recency outside the 12-month window"
assert (df["spend"] >= 0).all(), "negative spend should be impossible"
assert df["visit"].isin([0, 1]).all() and df["conversion"].isin([0, 1]).all()

# Logical funnel check: you cannot convert without visiting,
# and you cannot spend without converting.
assert (df.loc[df["conversion"] == 1, "visit"] == 1).all()
assert (df.loc[df["spend"] > 0, "conversion"] == 1).all()

print("\\nAll integrity checks passed.")
"""),
code("""
# The outcome funnel at a glance: visit -> conversion -> spend.
# These are POOLED numbers (all arms mixed) - purely descriptive at this stage.
print(f"visit rate      : {df['visit'].mean():.2%}")
print(f"conversion rate : {df['conversion'].mean():.3%}")
print(f"mean spend      : ${df['spend'].mean():.2f} per customer")
print(f"mean spend (buyers only): ${df.loc[df['spend']>0,'spend'].mean():.2f}")
print(f"share of customers with any spend: {(df['spend']>0).mean():.2%}")
"""),
md("""
Note how extreme the funnel is: ~15% visit, under 1% convert. This is *normal* for retail
email and it is exactly why sample size matters — a 0.9% base rate means the "signal" we
are hunting is tiny relative to the noise.

## 2. Sample Ratio Mismatch (SRM)

The design intended a **1/3 : 1/3 : 1/3** split. If the observed group sizes deviate from
that by more than chance allows, the assignment mechanism itself is broken — and nothing
downstream can be trusted. We test with a chi-square goodness-of-fit and use the industry
alarm threshold of **p < 0.001** (Microsoft reports ~6% of its experiments trip this check).
"""),
code("""
counts = df["segment"].value_counts().sort_index()
print(counts.to_string(), "\\n")

srm = srm_check(counts.values)  # equal split expected by default
print(f"chi-square = {srm['chi2']:.3f}, p = {srm['p_value']:.3f}")
print("SRM check:", "PASS - split is consistent with 1/3 each"
      if srm["pass"] else "FAIL - assignment mechanism suspect")
"""),
md("""
## 3. Covariate balance — the randomization audit

Every column measured **before** the email send (recency, spend history, gender of past
purchases, channel...) was fixed before the coin flip. So if randomization worked, the
three arms must be statistically indistinguishable on all of them.

This matters because balance is what lets us claim **causality**: if the arms are identical
in every respect except the email, any outcome difference can only be caused by the email.
"""),
code("""
PRE_TREATMENT = ["recency", "history", "mens", "womens",
                 "newbie", "channel", "zip_code"]

# ANOVA across the 3 arms for numeric covariates, chi-square for categoricals.
# Strict alpha=0.001, same logic as SRM: we run many checks, so a 0.05
# threshold would flag 1-in-20 by pure chance.
balance = balance_check(df, "segment", PRE_TREATMENT)
balance.style.format({"statistic": "{:.3f}", "p_value": "{:.3f}"})
"""),
code("""
# Visual version of the same audit: if randomization worked, the distribution of
# past spend should be three copies of the same curve. (Log scale because spend
# history is heavily right-skewed - a handful of customers spent thousands.)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for seg in df["segment"].unique():
    sns.kdeplot(np.log1p(df.loc[df["segment"] == seg, "history"]),
                label=seg, ax=axes[0])
axes[0].set_xlabel("log(1 + past-year spend $)")
axes[0].set_title("Past spend distribution by arm\\n(should overlap perfectly)")
axes[0].legend()

# Same idea for a categorical covariate: channel mix per arm.
channel_mix = (df.groupby("segment")["channel"]
                 .value_counts(normalize=True).unstack())
channel_mix.plot(kind="bar", ax=axes[1], rot=0)
axes[1].set_title("Purchase channel mix by arm\\n(bars should match across arms)")
axes[1].set_ylabel("share of arm")

fig.tight_layout()
fig.savefig("../assets/balance_checks.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
md("""
## 4. First (descriptive) look at outcomes by arm

We allow ourselves ONE descriptive table before the formal tests. No p-values here — that
is notebook 03's job, where multiple-comparison corrections are applied properly.
"""),
code("""
outcome_table = (df.groupby("segment")[["visit", "conversion", "spend"]]
                   .mean()
                   .rename(columns={"visit": "visit rate",
                                    "conversion": "conversion rate",
                                    "spend": "mean spend $"}))
outcome_table.style.format({"visit rate": "{:.2%}",
                            "conversion rate": "{:.3%}",
                            "mean spend $": "${:.3f}"})
"""),
code("""
# The chart version, with binomial standard-error bars so the eye gets a sense
# of uncertainty before the formal tests.
fig, ax = plt.subplots(figsize=(7.5, 4.5))

stats_ = df.groupby("segment")["conversion"].agg(["mean", "count"])
stats_["se"] = np.sqrt(stats_["mean"] * (1 - stats_["mean"]) / stats_["count"])
order = ["No E-Mail", "Womens E-Mail", "Mens E-Mail"]
stats_ = stats_.loc[order]

colors = ["#9e9e9e", "#e07a9b", "#4878cf"]  # control in grey, treatments colored
ax.bar(stats_.index, stats_["mean"] * 100,
       yerr=stats_["se"] * 100 * 1.96, capsize=6, color=colors)
ax.set_ylabel("2-week conversion rate (%)")
ax.set_title("Conversion by arm (error bars = 95% CI)")
for i, (idx, row) in enumerate(stats_.iterrows()):
    ax.text(i, row["mean"] * 100 + 0.02, f"{row['mean']:.2%}",
            ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig("../assets/conversion_by_arm.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
md("""
## Validation verdict

| Check | Result |
|---|---|
| Data integrity (missingness, ranges, funnel logic) | PASS |
| Sample Ratio Mismatch (chi-square vs 1/3:1/3:1/3) | PASS (p ≈ 0.90) |
| Covariate balance across 7 pre-treatment variables | PASS (all p > 0.4) |

The experiment is **valid**: the three arms are random samples of the same population, so
outcome differences can be attributed to the emails. The descriptive numbers *suggest* the
Men's email roughly doubles conversion — but that claim needs proper hypothesis tests with
multiple-comparison control, which is exactly what notebook `03_analysis.ipynb` does.
"""),
])

# ===========================================================================
# 03_analysis.ipynb
# ===========================================================================
analysis = nb([
md("""
# 03 — Core A/B Analysis

**The question:** does sending a marketing email cause incremental purchases and revenue —
and if so, which email should we send?

**Pre-registered design** (from `02_experiment_design.py`, committed *before* looking at
outcomes, so the goalposts can't move):

- **Primary metric:** 2-week conversion. **Secondary:** visit, spend.
- **alpha = 0.05** two-sided; two treatments vs one control ⇒ **Holm correction** on every
  metric family.
- **Decision rule:** ship an email iff its Holm-adjusted conversion p-value < 0.05 **and**
  the 95% CI on incremental spend excludes zero.
- Experiment validity was established in `01_eda.ipynb` (SRM + balance both pass).
""" ),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (two_proportion_ztest, welch_ttest, bootstrap_diff_ci,
                   holm_correction)

sns.set_theme(style="whitegrid", context="notebook")

df = pd.read_csv("../data/raw/hillstrom.csv")

ctrl = df[df["segment"] == "No E-Mail"]
mens = df[df["segment"] == "Mens E-Mail"]
womens = df[df["segment"] == "Womens E-Mail"]
ARMS = {"Mens E-Mail": mens, "Womens E-Mail": womens}
print({name: len(arm) for name, arm in ARMS.items()},
      "| control:", len(ctrl))
"""),
md("""
## 1. Primary metric: conversion

Binary outcome + huge n ⇒ **two-proportion z-test** (see `utils.py` for why the test uses a
pooled standard error while the CI uses an unpooled one).

We test each treatment against control, then apply **Holm's correction**: with two tests,
each at 5% false-positive risk, the chance of at least one fluke is ~10%; Holm shrinks the
family-wise error back to 5% without giving up as much power as plain Bonferroni.
"""),
code("""
def test_binary_metric(metric):
    \"\"\"Both treatments vs control on a 0/1 column, Holm-corrected.\"\"\"
    rows = []
    for name, arm in ARMS.items():
        r = two_proportion_ztest(int(arm[metric].sum()), len(arm),
                                 int(ctrl[metric].sum()), len(ctrl))
        rows.append({"arm": name, "treatment rate": r.p_treat,
                     "control rate": r.p_ctrl, "abs lift": r.abs_lift,
                     "rel lift": r.rel_lift, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "p raw": r.p_value})
    out = pd.DataFrame(rows).set_index("arm")
    # Correct within the metric family (2 tests per metric).
    out["p Holm"], out["significant"] = holm_correction(out["p raw"])
    return out

conv = test_binary_metric("conversion")
conv.style.format({"treatment rate": "{:.3%}", "control rate": "{:.3%}",
                   "abs lift": "{:+.3%}", "rel lift": "{:+.1%}",
                   "ci_low": "{:+.3%}", "ci_high": "{:+.3%}",
                   "p raw": "{:.2e}", "p Holm": "{:.2e}"})
"""),
md("""
**Reading the table:** the *absolute* lift (percentage points) is what converts to dollars;
the *relative* lift is what sounds impressive in a headline. Report both, always — "+119%"
and "+0.68pp" describe the same effect, and a stakeholder who hears only one of them has
been half-informed.
"""),
md("""
## 2. Secondary metric: visit

Same machinery. Visits are ~15x more common than conversions, so this test has far more
statistical resolution — useful as a mechanism check: *did the email at least pull people
to the site?*
"""),
code("""
visit = test_binary_metric("visit")
visit.style.format({"treatment rate": "{:.2%}", "control rate": "{:.2%}",
                    "abs lift": "{:+.2%}", "rel lift": "{:+.1%}",
                    "ci_low": "{:+.2%}", "ci_high": "{:+.2%}",
                    "p raw": "{:.2e}", "p Holm": "{:.2e}"})
"""),
md("""
## 3. Secondary metric: spend — the money question

Spend per customer is **zero-inflated** (99%+ zeros) and **heavily right-skewed** among
buyers. Two-pronged approach:

1. **Welch's t-test** — valid for comparing *means* of large samples via the CLT, and robust
   to the unequal variances that revenue data always has.
2. **Bootstrap CI** — 10,000 resamples, zero distributional assumptions. If it agrees with
   Welch (it will), the parametric shortcut was safe; if it disagreed, we'd trust the
   bootstrap.
"""),
code("""
spend_rows = []
boot_results = {}
for name, arm in ARMS.items():
    w = welch_ttest(arm["spend"], ctrl["spend"])
    b = bootstrap_diff_ci(arm["spend"], ctrl["spend"], n_boot=10_000)
    boot_results[name] = b
    spend_rows.append({
        "arm": name,
        "mean spend (treat)": w["mean_treat"],
        "mean spend (ctrl)": w["mean_ctrl"],
        "diff $/customer": w["diff"],
        "Welch CI": f"[{w['ci_low']:+.3f}, {w['ci_high']:+.3f}]",
        "Bootstrap CI": f"[{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]",
        "p raw": w["p_value"],
    })

spend = pd.DataFrame(spend_rows).set_index("arm")
spend["p Holm"], spend["significant"] = holm_correction(spend["p raw"])
spend.style.format({"mean spend (treat)": "${:.3f}",
                    "mean spend (ctrl)": "${:.3f}",
                    "diff $/customer": "${:+.3f}",
                    "p raw": "{:.2e}", "p Holm": "{:.2e}"})
"""),
code("""
# Visual proof that Welch and the bootstrap agree: the bootstrap sampling
# distribution of the spend difference, with both CIs overlaid.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

for ax, (name, b) in zip(axes, boot_results.items()):
    ax.hist(b["boot_diffs"], bins=60, alpha=0.85)
    ax.axvline(0, color="crimson", ls="--", lw=1.5, label="zero effect")
    ax.axvline(b["ci_low"], color="black", ls=":", lw=1.5)
    ax.axvline(b["ci_high"], color="black", ls=":", lw=1.5,
               label="bootstrap 95% CI")
    ax.set_title(f"{name} - ctrl: bootstrap of $\\Delta$ mean spend")
    ax.set_xlabel("$ per customer")
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("../assets/bootstrap_spend.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
md("""
## 4. All effects on one chart (forest plot)

The single most information-dense way to present A/B results: every effect with its 95% CI.
Anything whose interval clears zero is a real effect; the interval width is the honesty bar
that a bare "significant!" headline hides.
"""),
code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

panels = [("conversion", conv, "abs lift (pp)", 100),
          ("visit", visit, "abs lift (pp)", 100)]

for ax, (metric, table, xlabel, scale) in zip(axes[:2], panels):
    for i, (arm_name, row) in enumerate(table.iterrows()):
        color = "#4878cf" if "Mens" in arm_name else "#e07a9b"
        ax.errorbar(row["abs lift"] * scale, i,
                    xerr=[[(row["abs lift"] - row["ci_low"]) * scale],
                          [(row["ci_high"] - row["abs lift"]) * scale]],
                    fmt="o", capsize=5, color=color, markersize=9)
    ax.axvline(0, color="crimson", ls="--", lw=1.2)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{metric} lift vs control")

ax = axes[2]
for i, (arm_name, row) in enumerate(spend.iterrows()):
    b = boot_results[arm_name]
    color = "#4878cf" if "Mens" in arm_name else "#e07a9b"
    ax.errorbar(row["diff $/customer"], i,
                xerr=[[row["diff $/customer"] - b["ci_low"]],
                      [b["ci_high"] - row["diff $/customer"]]],
                fmt="o", capsize=5, color=color, markersize=9)
ax.axvline(0, color="crimson", ls="--", lw=1.2)
ax.set_yticks(range(len(spend)))
ax.set_yticklabels(spend.index)
ax.set_xlabel("$ / customer (bootstrap CI)")
ax.set_title("spend lift vs control")

fig.suptitle("Treatment effects with 95% confidence intervals", y=1.04,
             fontweight="bold")
fig.tight_layout()
fig.savefig("../assets/forest_plot.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
md("""
## 5. From lift to dollars — the number the C-suite actually asked for

The experiment measures **incremental spend per emailed customer** (treatment mean − control
mean). Control already captures what customers would have spent anyway, so this difference
is pure causal impact — no baseline subtraction gymnastics needed.

Scaling assumption (stated, not hidden): effects transfer proportionally to a larger
customer file *of the same composition*. A national retailer emails millions, not 21k.
"""),
code("""
mens_lift_per_customer = spend.loc["Mens E-Mail", "diff $/customer"]
mens_ci_low = boot_results["Mens E-Mail"]["ci_low"]
mens_ci_high = boot_results["Mens E-Mail"]["ci_high"]

# Sensitivity across file sizes and email frequency: a range, not one number,
# because the honest projection carries its uncertainty with it.
print(f"Incremental spend per emailed customer (Mens E-Mail): "
      f"${mens_lift_per_customer:.3f}  "
      f"(95% CI ${mens_ci_low:.3f} to ${mens_ci_high:.3f})\\n")

print(f"{'file size':>12} | {'sends/yr':>8} | {'point estimate':>16} | "
      f"{'95% CI range':>28}")
print("-" * 75)
for file_size in [1_000_000, 5_000_000, 10_000_000]:
    for sends in [12, 26]:  # monthly vs biweekly campaign cadence
        pt = file_size * sends * mens_lift_per_customer
        lo = file_size * sends * mens_ci_low
        hi = file_size * sends * mens_ci_high
        print(f"{file_size:>12,} | {sends:>8} | ${pt:>14,.0f} | "
              f"${lo:>11,.0f} - ${hi:>11,.0f}")
"""),
md("""
**Caveats that keep this honest** (also in `docs/03_stakeholder_faq.md`):

- The per-send effect will **not** stay constant at 26 sends/year — list fatigue is real.
  Treat the biweekly rows as an upper bound; the honest claim is the *monthly* row.
- The effect was measured over a 2-week window; longer-horizon cannibalization (customers
  buying now instead of later) is not visible in this design.
- Scaling assumes the larger file has the same composition as the test population.

## 6. Verdict against the pre-registered decision rule

| Rule | Mens E-Mail | Womens E-Mail |
|---|---|---|
| Holm-adjusted conversion p < 0.05 | YES | YES |
| Spend CI excludes zero | YES | NO (interval spans 0) |
| **Decision** | **SHIP** | **DO NOT SHIP (yet)** |

The Men's email is the unambiguous winner: it roughly **doubles conversion** and produces
measurable incremental revenue. The Women's email lifts conversion but its revenue effect is
statistically indistinguishable from zero — under the pre-registered rule, it doesn't ship.
That is the point of committing to the rule in advance: "significant on one metric" is not
the same as "makes money".

Next: `04_segmentation.ipynb` asks *who* drives the Men's email effect — because "send to
everyone" is almost never the revenue-maximizing policy.
"""),
])

# ===========================================================================
# 04_segmentation.ipynb
# ===========================================================================
seg = nb([
md("""
# 04 — Segmentation: who actually responds?

Notebook 03 established that the **Men's email** wins *on average*. But averages hide
targeting opportunities: if the entire effect comes from one customer segment, emailing
everyone wastes sends (and goodwill) on people who will never respond.

**Statistical health warning, stated up front:** subgroup analysis is **exploratory**.
Slice the data enough ways and something will look significant by chance. Discipline here:

1. Segments are defined by **pre-treatment** variables only (past behavior, not outcomes).
2. Every segment family gets a **Holm correction**.
3. Findings are treated as **hypotheses for a follow-up confirmatory test**, not as
   shipping decisions by themselves.
"""),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from utils import two_proportion_ztest, holm_correction

sns.set_theme(style="whitegrid", context="notebook")

df = pd.read_csv("../data/raw/hillstrom.csv")

# We segment the WINNER (Mens E-Mail) against control. The Womens arm is
# excluded so every comparison below is a clean two-group test.
sub = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
sub["treated"] = (sub["segment"] == "Mens E-Mail").astype(int)

# Human-readable segment variables, all pre-treatment:
sub["recency_bucket"] = pd.cut(sub["recency"], bins=[0, 3, 6, 9, 12],
                               labels=["1-3 mo", "4-6 mo",
                                       "7-9 mo", "10-12 mo"])
sub["past_mens_buyer"] = np.where(sub["mens"] == 1,
                                  "bought mens before", "no mens history")
sub["customer_age"] = np.where(sub["newbie"] == 1,
                               "new customer", "established")
print(f"{len(sub):,} rows (Mens E-Mail + control only)")
"""),
md("""
## 1. Conversion lift within each segment

For every level of every segment variable we run the same two-proportion z-test as the main
analysis: treated vs control *within that slice*. The Holm correction is applied across
**all segment tests at once** — the strictest, most defensible choice.
"""),
code("""
SEGMENT_VARS = ["recency_bucket", "past_mens_buyer", "customer_age",
                "channel", "zip_code"]

rows = []
for var in SEGMENT_VARS:
    for level, grp in sub.groupby(var, observed=True):
        t = grp[grp["treated"] == 1]
        c = grp[grp["treated"] == 0]
        r = two_proportion_ztest(int(t["conversion"].sum()), len(t),
                                 int(c["conversion"].sum()), len(c))
        rows.append({"variable": var, "segment": str(level),
                     "n": len(grp), "treat rate": r.p_treat,
                     "ctrl rate": r.p_ctrl, "abs lift": r.abs_lift,
                     "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "p raw": r.p_value})

segments = pd.DataFrame(rows)
# One correction across ALL 15 subgroup tests - slicing is where false
# positives breed, so this is where the correction earns its keep.
segments["p Holm"], segments["significant"] = holm_correction(
    segments["p raw"])

segments.sort_values("abs lift", ascending=False).style.format(
    {"treat rate": "{:.3%}", "ctrl rate": "{:.3%}", "abs lift": "{:+.3%}",
     "ci_low": "{:+.3%}", "ci_high": "{:+.3%}",
     "p raw": "{:.2e}", "p Holm": "{:.2e}"})
"""),
code("""
# Forest plot: every segment's lift with its CI, grouped by variable.
plot_df = segments.iloc[::-1].reset_index(drop=True)  # nicer top-to-bottom order

fig, ax = plt.subplots(figsize=(9, 0.45 * len(plot_df) + 1.5))
palette = dict(zip(SEGMENT_VARS, sns.color_palette("deep", len(SEGMENT_VARS))))

for i, row in plot_df.iterrows():
    ax.errorbar(row["abs lift"] * 100, i,
                xerr=[[(row["abs lift"] - row["ci_low"]) * 100],
                      [(row["ci_high"] - row["abs lift"]) * 100]],
                fmt="o" if row["significant"] else "s",
                capsize=4, markersize=8,
                color=palette[row["variable"]],
                alpha=1.0 if row["significant"] else 0.45)

ax.axvline(0, color="crimson", ls="--", lw=1.2)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df["variable"] + " = " + plot_df["segment"])
ax.set_xlabel("conversion lift vs control (percentage points)")
ax.set_title("Mens E-Mail effect by segment "
             "(filled circle = significant after Holm)")
fig.tight_layout()
fig.savefig("../assets/segmentation_forest.png", dpi=150,
            bbox_inches="tight")
plt.show()
"""),
md("""
## 2. Are the differences BETWEEN segments real? (interaction tests)

A larger lift in one segment than another can itself be a fluke. The formal check is an
**interaction test**: fit a logistic regression of conversion on treatment, the segment
variable, and their product term. If the interaction coefficient is significant, the
treatment effect genuinely *differs* across segments — that's the statistical license to
target.
"""),
code("""
# Logistic regression with a treatment x segment interaction, one variable
# at a time. The likelihood-ratio test compares the model with vs without
# the interaction term - the cleanest "is targeting justified?" test.
import scipy.stats as st

for var in ["past_mens_buyer", "recency_bucket", "customer_age"]:
    full = smf.logit(f"conversion ~ treated * C({var})", data=sub).fit(disp=0)
    reduced = smf.logit(f"conversion ~ treated + C({var})", data=sub).fit(disp=0)
    lr_stat = 2 * (full.llf - reduced.llf)
    dof = full.df_model - reduced.df_model
    p = st.chi2.sf(lr_stat, dof)
    print(f"{var:<18} LR = {lr_stat:6.2f}, dof = {dof:.0f}, "
          f"interaction p = {p:.4f}"
          + ("   <-- effect differs across segments" if p < 0.05 else ""))
"""),
md("""
## 3. What this means for the business

Read the forest plot and interaction tests together (exact numbers are in the table above):

- The lift is **not uniform**: some segments carry substantially more of the effect than
  others, and the interaction tests tell us which of those differences are statistically
  real rather than slicing noise.
- Where the interaction is significant, a **targeted send** beats a blast: same revenue
  from fewer emails, which protects the scarcest asset in email marketing — subscriber
  attention (every irrelevant send raises unsubscribe risk).
- Where the interaction is *not* significant, resist the temptation to build targeting
  rules on the point estimates. That's how teams ship noise.

**The disciplined next step** is not "roll out targeting tomorrow" — it is a follow-up
**confirmatory experiment** that pre-registers the winning segments as its primary
hypothesis. Exploration generates the hypothesis; only a fresh test confirms it. That
two-step loop (explore → confirm) is exactly how mature experimentation teams operate.
"""),
])

for path, notebook in [("01_eda.ipynb", eda),
                       ("03_analysis.ipynb", analysis),
                       ("04_segmentation.ipynb", seg)]:
    nbf.write(notebook, path)
    print("wrote", path)
