# A/B Testing, Explained Properly

*A theory guide written for a smart reader who is not a statistician. Everything here applies to any industry — email, ads, pricing, product features, medicine. The companion doc, `02_technical_deep_dive.md`, has the math.*

---

## 1. Why experiments exist at all

Suppose a retailer sends a promotional email and sales go up 8% that week. Did the email work?

You genuinely cannot tell. Maybe it was payday week. Maybe a competitor had a stockout. Maybe the weather changed. Sales move for a hundred reasons, and the email is only one of them. Comparing "after" to "before" — or this month to last month — confounds the email's effect with everything else that changed at the same time.

The fix is one idea, and it is the whole foundation of A/B testing:

> **Split your audience randomly. Treat one group. Leave the other alone. Compare them at the same moment in time.**

Because the split is *random*, the two groups are statistically identical in every way — same mix of loyal customers, bargain hunters, new signups, big spenders. Payday week hits both groups equally. The competitor's stockout helps both groups equally. The *only* systematic difference between them is the thing you changed. So any difference in outcomes has exactly two possible explanations: your change, or random luck. And statistics exists precisely to measure how much luck could plausibly explain.

That is the entire trick. Everything else in this document is the machinery for handling the "random luck" part honestly.

## 2. The anatomy of a proper test

Our project's dataset is a real example: a retailer took 64,000 customers and randomly split them three ways — one third received an email featuring men's merchandise, one third an email featuring women's merchandise, and one third received **no email at all**. Then, for two weeks, they measured who visited the site, who bought, and how much they spent.

Every element of that sentence is a design decision:

- **The control group (no email).** Without it there is nothing to compare against. The control group tells you what would have happened anyway — the counterfactual. Customers visit and buy without any email; the control measures that baseline (in our data, 0.57% of untouched customers bought within the two weeks).
- **Random assignment.** Not "emails to loyal customers, nothing to lapsed ones" — that would compare different kinds of people, and the difference in outcomes would reflect who they are, not what you sent.
- **Fixed measurement window.** Two weeks, decided in advance. Deciding when to stop *after* watching results is one of the classic ways tests go wrong (see pitfall #1 below).
- **Metrics chosen in advance.** Visit (did the email create interest?), conversion (did it create purchases?), spend (what is it worth?). Choosing your success metric after seeing which one looks best is another classic failure.

## 3. What "statistically significant" actually means

In our test, 1.25% of customers who got the men's email bought something, versus 0.57% of the control group. The lift is +0.68 percentage points. Real, or luck?

Think of a courtroom. The defendant is the claim "this email does nothing" (statisticians call this the *null hypothesis*). We presume it innocent, and we ask: **if the email truly did nothing, how surprising would our data be?** That measure of surprise is the **p-value**.

For the men's email, the p-value is about 0.0000000000002. Translation: if the email truly did nothing, you could rerun this experiment trillions of times and essentially never see a gap this large by luck alone. The evidence is overwhelming; we reject the "does nothing" hypothesis.

The conventional threshold is p < 0.05 — we call a result "significant" when data this extreme would occur less than 5% of the time under pure luck. Three things people constantly get wrong about it:

1. **The p-value is not the probability the result is a fluke.** It's the probability of seeing data like ours *assuming* the email does nothing. Subtle, but different.
2. **0.05 is a convention, not physics.** It means: if you run many tests of genuinely useless changes, about 1 in 20 will "win" by luck. That's a business risk tolerance you're choosing.
3. **Significance says nothing about size.** With enough customers, a lift worth two cents a year becomes "significant". Which is why we always report...

## 4. Effect sizes and confidence intervals — where decisions live

The decision-grade statement about the men's email is not "p < 0.05". It's:

> Conversion lift of **+0.68 percentage points**, with a 95% confidence interval of **+0.50 to +0.86 points**.

The confidence interval is the honest range of effect sizes consistent with our data. Even in the worst plausible case (+0.50pp), the email nearly doubles conversion. That's what makes the decision easy — not the p-value alone, but a *lower bound* that is still clearly worth money.

Rule of thumb for reading any test result: **the p-value tells you whether to believe the effect exists; the confidence interval tells you whether to care.**

## 5. Power: the question to ask BEFORE the test

Flip a coin 10 times, get 7 heads — is the coin rigged? Nobody can say; 10 flips can't tell a fair coin from a mildly rigged one. Sample size determines what a test can *see*.

**Power** is the probability that your test detects an effect of a given size, if that effect is real. The industry standard is designing for 80% power. Before launching, a competent experimenter computes the **minimum detectable effect (MDE)**: the smallest lift the test can reliably catch at the planned sample size.

For our experiment (21,306 customers per arm, 0.57% baseline conversion), the MDE works out to about **0.22 percentage points**. The actual effect (+0.68pp) was three times larger — comfortably detectable. But if the true effect had been +0.1pp, this test would usually have *missed it*, and a "not significant" result would have meant "the test was too small to see it", not "the email doesn't work".

This is the single most misread situation in business experimentation: **absence of evidence is not evidence of absence.** An underpowered test that comes back "not significant" tells you almost nothing.

Power also explains why small effects are expensive to detect: halving the effect you want to see *quadruples* the sample you need.

## 6. The classic ways tests lie to you

### Pitfall 1: Peeking (early stopping)

A team checks the dashboard daily and stops the test the day it shows p < 0.05. This feels diligent and is actually cheating: each look is another lottery ticket for a false positive. Check daily for a month and your real false-positive rate isn't 5% — it can be 20–30%. Our Experiment Auditor tool includes a replay simulation that makes this visceral: p-values wander, and crossing 0.05 briefly is common even when nothing is going on. The fix: fix the duration in advance, or use sequential methods explicitly built for interim looks.

### Pitfall 2: Sample Ratio Mismatch (SRM)

You designed a 50/50 split but got 52/48. With big numbers, that's not rounding — it's a symptom that the assignment machinery is broken (a bug drops users from one arm, bots are filtered asymmetrically, redirects fail). Microsoft reports ~6% of its experiments fail this check, and the rule there is brutal and correct: an SRM'd experiment is discarded, not "adjusted". In this project, the Hillstrom test passes SRM cleanly; a popular public ad dataset fails it at first glance (it's only valid because its 96/4 split was the *design* — the auditor shows both readings).

### Pitfall 3: Multiple comparisons

Test 20 metrics, or slice results by 20 segments, and about one will be "significant" by pure luck. Whoever searches longest finds the most false gold. The fix is statistical correction (we use the Holm method) and, more fundamentally, honesty about which analyses were planned versus which were found by exploring. Found-by-exploring insights are hypotheses for the *next* test, not decisions.

### Pitfall 4: Novelty effects

Users click a new button because it's new; the effect fades in a month. Two-week tests measure two-week behavior. For decisions with long horizons, either run longer or re-test after the novelty wears off.

### Pitfall 5: Simpson's paradox

An effect can be positive in every segment yet look different in the aggregate (or vice versa) when segment sizes differ across arms. This is another reason randomization and balance checks matter — they keep segment composition identical across arms.

## 7. From statistics to money

Statistics ends with "the lift is real". Business begins with "what is it worth?"

Our result: the men's email produced **+$0.77 of revenue per emailed customer** over two weeks (95% CI: +$0.49 to +$1.06), against a send cost of about a tenth of a cent. Project that honestly — using the *conservative* end of the interval, at the scale of a 10-million-customer email file, monthly campaigns — and the annual value of choosing the right email is measured in **tens of millions of dollars**, against send costs measured in thousands.

That ratio — millions of upside decided by a test that costs almost nothing — is why Booking.com runs ~25,000 experiments a year, why Microsoft/Google/Netflix built experimentation platforms as core infrastructure, and why one famous Bing headline test was worth over $100M a year. The test isn't the expensive part. Being wrong is.

## 8. The decision playbook

When a test finishes, there are only four honest outcomes:

| Situation | Decision |
|---|---|
| Significant lift, CI lower bound still worth money | **Ship it** |
| Significant lift, but CI includes economically trivial values | Ship if costless; otherwise extend the test |
| Not significant, test was well-powered for effects you'd care about | **Don't ship**; the effect, if any, is too small to matter |
| Not significant, test was underpowered | **The test failed, not the idea.** Redesign with more sample |

And one meta-rule that outranks all of them: **if the validity checks fail (SRM, balance), there is no result.** Fix the pipeline and rerun.

---

*Next: [`02_technical_deep_dive.md`](02_technical_deep_dive.md) for the math behind each of these ideas, and [`03_stakeholder_faq.md`](03_stakeholder_faq.md) for the boardroom version.*
