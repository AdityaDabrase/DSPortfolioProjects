# Technical Deep Dive: The Math Behind the Test

*Companion to `01_theory.md`. This is the derivation-level material — what you'd want in your head before a technical interview or a methods review. Every formula here is implemented from scratch in [`code/utils.py`](../code/utils.py) and [`code/experiment_design.py`](../code/experiment_design.py), and every numeric example uses this project's actual results.*

---

## 1. The two-proportion z-test, from first principles

### Setup

Each customer in arm \(i\) either converts (1) or doesn't (0) — a Bernoulli trial with unknown true rate \(p_i\). With \(n\) customers, the observed rate \(\hat{p} = X/n\) has:

\[
E[\hat{p}] = p, \qquad \operatorname{Var}(\hat{p}) = \frac{p(1-p)}{n}
\]

By the Central Limit Theorem, \(\hat{p}\) is approximately normal for large \(n\). Our smallest cell (conversions in control: 122 events out of 21,306) is far above the usual \(np \ge 10\) rule of thumb, so the approximation is excellent.

### The test statistic

We want the distribution of the *difference* \(\hat{p}_T - \hat{p}_C\) under the null hypothesis \(H_0: p_T = p_C = p\). Independent arms mean variances add:

\[
\operatorname{Var}(\hat{p}_T - \hat{p}_C) = p(1-p)\left(\frac{1}{n_T} + \frac{1}{n_C}\right)
\]

We don't know \(p\), so under \(H_0\) we estimate it by pooling both arms: \(\hat{p}_{pool} = (X_T + X_C)/(n_T + n_C)\). The z-statistic is:

\[
z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}_{pool}(1-\hat{p}_{pool})\left(\frac{1}{n_T} + \frac{1}{n_C}\right)}}
\]

and the two-sided p-value is \(2\Phi(-|z|)\).

**Worked example (men's email, conversion):** \(\hat{p}_T = 0.0125\), \(\hat{p}_C = 0.0057\), \(n_T = 21{,}307\), \(n_C = 21{,}306\). Pooled rate ≈ 0.0091, SE ≈ 0.00092, \(z ≈ 7.4\), \(p ≈ 1.5 \times 10^{-13}\).

### Why the CI uses a different standard error

The confidence interval describes the difference itself — no null hypothesis imposed — so each arm keeps its own variance estimate:

\[
(\hat{p}_T - \hat{p}_C) \pm z_{0.975}\sqrt{\frac{\hat{p}_T(1-\hat{p}_T)}{n_T} + \frac{\hat{p}_C(1-\hat{p}_C)}{n_C}}
\]

This pooled-vs-unpooled distinction is a favorite interview probe: **pool under the null (test), don't pool for estimation (CI).**

## 2. Welch's t-test and why not Student's

For spend (continuous), the statistic is:

\[
t = \frac{\bar{x}_T - \bar{x}_C}{\sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}}
\]

Student's t-test assumes equal variances across arms and pools them; Welch's does not, and adjusts the degrees of freedom via Welch–Satterthwaite:

\[
\nu \approx \frac{\left(\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}\right)^2}{\frac{(s_T^2/n_T)^2}{n_T - 1} + \frac{(s_C^2/n_C)^2}{n_C - 1}}
\]

A treatment that *works* changes the outcome distribution, and usually its variance too (more buyers ⇒ more nonzero spends ⇒ higher variance). So the equal-variance assumption is most wrong exactly when the test matters most. Welch costs essentially nothing when variances happen to be equal — which is why "always Welch" is the modern default.

### The zero-inflation objection, answered with a bootstrap

Spend is ~99% zeros with a long right tail — nothing like a normal distribution. The t-test doesn't need the *data* to be normal, only the *sampling distribution of the mean*, which the CLT guarantees at \(n \approx 21{,}000\). Rather than assert this, notebook 03 demonstrates it: a 10,000-resample bootstrap (resample each arm with replacement, recompute the difference in means, take the 2.5th/97.5th percentiles) gives

| Comparison | Welch 95% CI | Bootstrap 95% CI |
|---|---|---|
| Men's vs control | [+$0.485, +$1.055] | [+$0.487, +$1.059] |
| Women's vs control | [+$0.169, +$0.680] | [+$0.169, +$0.682] |

Agreement to the cent. When a skeptic raises skew, you show this table.

(If the sample had been small, the right tool would have been the bootstrap alone, or a Mann–Whitney U test — remembering that Mann–Whitney tests distributional shift, not the difference in means, and revenue decisions are about means.)

## 3. Power and sample size, derived

Two error types, two knobs:

- **Type I error** \(\alpha\): probability of declaring a winner when the true effect is zero. Set \(\alpha = 0.05\).
- **Type II error** \(\beta\): probability of missing a real effect of size \(\delta\). Power \(= 1 - \beta\), designed at 0.80.

The test rejects when \(|\hat{p}_T - \hat{p}_C| > z_{1-\alpha/2} \cdot SE_0\). For power, we need this to happen with probability \(1-\beta\) when the truth is a difference of \(\delta\). Setting the two conditions against each other and solving for \(n\) per arm:

\[
n = \frac{\left(z_{1-\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{1-\beta}\sqrt{p_1(1-p_1) + p_2(1-p_2)}\right)^2}{\delta^2}
\]

where \(\bar{p} = (p_1 + p_2)/2\). The two standard-error terms differ because the variance under the null (both arms at \(\bar p\)) differs from the variance under the alternative (arms at \(p_1\), \(p_2\)).

**Key structural fact:** \(n \propto 1/\delta^2\). Halving the effect you want to detect quadruples the sample. This single fact drives most real-world experiment sizing conversations.

**This project's numbers** (baseline 0.57%, \(\alpha=0.05\), power 80%):

| Target MDE | Required n per arm | Feasible at 21,306? |
|---|---|---|
| +0.10pp | 97,126 | No |
| +0.20pp | 26,218 | No |
| +0.30pp | 12,512 | Yes |
| MDE at n = 21,306 | — | **0.224pp (39% relative)** |

The observed +0.68pp was ~3× the MDE; post-hoc power to detect it ≈ 100%. Had the result been null, we could honestly have ruled out effects above ~0.22pp — but nothing smaller.

The MDE inversion has no closed form; `experiment_design.py` solves it by bisection on the sample-size formula.

## 4. Multiple comparisons: Holm–Bonferroni mechanics

Run \(m\) independent tests at level \(\alpha\) and the family-wise error rate (FWER — probability of ≥1 false positive) is \(1 - (1-\alpha)^m\). For our \(m = 6\) confirmatory tests: \(1 - 0.95^6 \approx 26\%\).

**Bonferroni** fixes this by testing each at \(\alpha/m\) — valid but conservative. **Holm** is uniformly better with the same guarantee:

1. Sort p-values ascending: \(p_{(1)} \le \dots \le p_{(m)}\).
2. Compare \(p_{(k)}\) against \(\alpha / (m - k + 1)\) — the smallest p faces the harshest bar (\(\alpha/m\)), the next faces \(\alpha/(m-1)\), etc.
3. Stop at the first non-rejection; everything after also fails.
4. Adjusted p-values: \(\tilde p_{(k)} = \max_{j \le k} \left[(m - j + 1) \, p_{(j)}\right]\), capped at 1 (the max enforces monotonicity).

All six of our tests survive Holm comfortably (largest adjusted p = 0.0011). In the segmentation notebook the family is much larger (~30 subgroup tests), which is exactly where uncorrected analysis starts minting false discoveries — several nominally significant slices there are honest only *because* they still clear the corrected bar.

When the test family is huge (metrics platforms with hundreds of metrics), FWER control becomes too strict and the field switches to controlling the **false discovery rate** (Benjamini–Hochberg): "of the things we call winners, at most 5% are false" — a different, weaker, but more scalable promise.

## 5. Validity checks, formally

### Sample Ratio Mismatch

Chi-square goodness of fit of observed arm counts against designed shares:

\[
\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k-1} \text{ under } H_0
\]

Hillstrom: counts (21,387 / 21,307 / 21,306) vs expected 1/3 each ⇒ p = 0.904, clean pass. The alarm threshold is p < 0.001 rather than 0.05 because assignment is checked on every experiment ever run — at that volume, a 5% false-alarm rate would swamp real failures, and true SRMs tend to produce absurdly small p-values anyway (the Kaggle ad dataset "fails" at p ≈ 0 against a 50/50 assumption; it passes against its actual 96/4 design).

### Covariate balance via standardized mean differences

\[
SMD = \frac{\bar{x}_T - \bar{x}_C}{\sqrt{(s_T^2 + s_C^2)/2}}
\]

Threshold |SMD| < 0.1 (Austin, 2009). The deliberate choice here: **not** t-tests. At \(n = 21k\), a t-test flags a $2 difference in average purchase history as "significant imbalance" when it is utterly incapable of confounding a conversion analysis. SMD measures the imbalance in units that relate to confounding potential. All five pre-treatment covariates in Hillstrom show worst-pairwise |SMD| < 0.009 — an order of magnitude below the threshold.

Also formalized in the auditor: balance is only checkable on **pre-treatment** variables. Post-treatment variables (visits, spend) differ *because the treatment worked*; including them would generate false alarms. (First version of our own auditor made exactly this mistake and flagged `visit` — the fix is preserved in the tool's `--post-treatment` flag, and it's a good war story.)

## 6. Peeking, quantified

Under continuous monitoring with a naive stop-at-p<0.05 rule, the p-value process is a random walk that gets unlimited attempts at crossing the boundary. Armitage et al. (1969) computed the inflation: checking 5 times ≈ 14% real \(\alpha\); 20 times ≈ 22%; continuous ≈ guaranteed eventual crossing as \(n \to \infty\).

The auditor's peeking module replays the actual dataset in random arrival order and computes the p-value at 100 interim checkpoints. On Hillstrom (women's email arm), 72/100 looks were below 0.05 and the final p was 0.00016 — a stable real signal. The interesting contrast is an A/A test (no true effect), where the trajectory routinely dips below 0.05 and comes back — each dip a false winner for a peeking team.

Principled interim looks exist: group-sequential designs (O'Brien–Fleming boundaries spend the \(\alpha\) budget across looks) and always-valid inference (mixture sequential probability ratio tests, used by Optimizely/Eppo). The point is not "never look" — it's "looking must be priced into the math".

## 7. Variance reduction: CUPED in brief

The most important modern technique this dataset can't showcase well. CUPED (Controlled-experiment Using Pre-Experiment Data, Deng et al. 2013) adjusts each outcome by its correlation with a pre-experiment covariate \(X\):

\[
Y^{cuped} = Y - \theta(X - \bar{X}), \qquad \theta = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)}
\]

Randomization makes \(E[X_T] = E[X_C]\), so the adjustment doesn't move the effect estimate — it only removes the part of outcome variance predictable from \(X\), shrinking variance by a factor \((1 - \rho^2)\). With \(\rho = 0.5\) you need 25% less sample for the same power; Netflix and Microsoft report routine gains like this using pre-experiment engagement as \(X\).

On Hillstrom, the available covariates (past spend `history`, etc.) correlate with two-week conversion at \(\rho < 0.08\), so CUPED would cut variance by well under 1% — the honest conclusion is "right technique, wrong covariates". The published replications of this dataset find the same.

## 8. The Bayesian alternative, for contrast

Everything above is frequentist. The Bayesian formulation: put a prior on each arm's rate (e.g. \(\text{Beta}(1,1)\)), update with observed conversions (Beta is conjugate: posterior \(\text{Beta}(1 + X, 1 + n - X)\)), then compute \(P(p_T > p_C \mid \text{data})\) by sampling both posteriors.

Advantages: the output ("99.99% probability the men's email is better, expected lift +0.68pp") is what stakeholders think a p-value means; no explicit multiple-comparison machinery; expected-loss decision rules handle early stopping more gracefully. Costs: priors must be defended, and badly implemented Bayesian dashboards smuggle the same peeking problem back in. On data this decisive, both frameworks scream the same answer — the philosophical difference only bites on marginal calls.

## 9. Causal language, precisely

What this RCT licenses: "sending the men's email **caused** an additional 0.68pp of customers to purchase within two weeks." Randomization severs every backdoor path between assignment and outcome, so the average treatment effect is identified by the simple difference in means:

\[
ATE = E[Y \mid T{=}1] - E[Y \mid T{=}0]
\]

What it does not license: extrapolation beyond the population (customers with a purchase in the last 12 months), beyond the window (two weeks — long-run annoyance/unsubscribe effects are invisible here), or to different creatives. Each of those is a new hypothesis for a new test. Being precise about the boundary of the causal claim is, in a boardroom, the difference between an analyst and a liability.
