# Stakeholder FAQ

*Every hard question a CMO, CFO, or product executive might ask about this test — with the two-sentence answer first and the detail underneath. Numbers are the actual results of this experiment (64,000 customers, three arms, two-week window).*

---

## The result

### "What did we learn, in one sentence?"

Sending the men's merchandise email caused purchases to more than double versus sending nothing (1.25% vs 0.57%), and every dollar of measured lift survives the strictest statistical scrutiny we can throw at it.

### "How much money is this worth?"

At test scale, the men's email generated about **$16,400 of incremental revenue per 21,000 customers emailed**, at a send cost of roughly $21. Scaled to a 10-million-customer file with monthly campaigns, choosing the men's email over no email is worth roughly **$92M/year expected — and at least $58M/year even using the most conservative end of the confidence interval** — against annual send costs around $120K.

Detail: the per-customer revenue lift is +$0.77 (95% CI +$0.49 to +$1.06). We deliberately quote the CI's lower bound alongside the expectation so the projection can't be accused of cherry-picking. The scaling assumptions (file size, frequency, ~$0.001/send) are stated in the analysis notebook and can be replaced with our actual numbers in minutes.

### "Why should I believe the email *caused* this, and it's not just better customers in that group?"

Because assignment was random, the three groups were statistically identical before the first email went out — we verified this on every customer attribute we have (past spend, recency, category, channel, tenure; all differences are 10× below the standard imbalance threshold). Identical groups plus different treatment means the outcome difference has only two possible sources — the email or luck — and the odds that luck produced a gap this size are about 1 in 7 trillion.

### "The women's email also won. Why are we not sending that?"

It beat doing nothing (+0.31pp conversion, +$0.42/customer) but was decisively beaten by the men's email — roughly half the lift on every metric. Against control both are winners; against each other there's one champion.

---

## Attacking the method

### "Couldn't we have just compared sales to last month instead of holding out a control group?"

No — sales move for dozens of reasons (seasonality, promotions, competitors, weather), and a before/after comparison attributes all of them to the email. The 21,000-customer control group is what tells us what would have happened anyway; it's the difference between measurement and guessing.

The control group is not "wasted revenue" — it's the price of knowing the other 42,000 customers' lift was real. That price: roughly $16K of foregone two-week revenue (21,306 customers × $0.77 unrealized lift), to validate a decision worth tens of millions annually.

### "95% confident — so there's a 5% chance this is wrong?"

The specific risk convention is: if we ran tests of useless emails all year, at most 5% would falsely look like winners. This particular result is vastly stronger than the threshold (p ≈ 0.0000000000002, not 0.05) — the practical probability that this winner is a fluke is negligible.

### "You measured three metrics on two emails. Isn't something bound to look good by chance?"

Yes — running six tests at 5% risk each gives about a 26% chance of at least one fluke "winner", which is why we applied a formal correction (Holm's method) that tightens the bar as the number of tests grows. All six results survive it.

### "The data is mostly people who spent $0. Can you even do statistics on that?"

Yes, and we proved it rather than assumed it: alongside the standard test we ran a bootstrap — a method that makes no assumptions about the data's shape — and the two produced confidence intervals identical to the cent. When two methods with different assumptions agree, the assumptions weren't doing the work.

### "The test 'looked significant' after three days. Why did we wait the full two weeks?"

Because stopping the moment a dashboard shows significance is statistically equivalent to buying lottery tickets until one wins — checking daily can triple the false-positive rate. We fixed the window in advance and only judged at the end; our auditor tool includes a replay showing how p-values wander across the "significant" line even for effects that aren't real.

### "How do we know the 3-way split itself wasn't botched?"

We ran the same check Microsoft and Airbnb run on every experiment — a Sample Ratio Mismatch test comparing observed group sizes to the designed split. This test passes cleanly (p = 0.90); for contrast, our auditor correctly flags a popular public ad dataset whose split looks broken until you learn its design was intentionally 96/4.

---

## Scope and follow-ups

### "Will this hold for our other 9.94 million customers?"

The test population was customers who purchased in the last 12 months, so the honest claim covers people like them — which is most of an active email file. For very different populations (5-year-lapsed customers, never-buyers), the disciplined answer is a cheap follow-up test, not extrapolation.

### "Does this mean email men's products to everyone forever?"

It means the men's email is the right *default* until a challenger beats it in a test. Effects decay (novelty, fatigue, seasonality), so mature email programs re-validate their champion a few times a year and always have a challenger running.

### "It worked even better for [some segment]. Can we act on that?"

Carefully. The segment analysis shows the men's email wins almost everywhere (which strengthens the ship decision), and hints that customers who previously bought in both categories respond most — but when we formally tested whether segment differences are real rather than noise, the key interaction did not reach significance (p = 0.28). Segment findings from a completed test are hypotheses for the next test, not decisions; acting on the best-looking slice of many is how teams ship mirages.

### "What would you test next?"

Three candidates, in order of expected value: (1) category-matched targeting — men's email to past men's buyers, women's to women's — as a proper randomized test, since exploration suggests but doesn't confirm it; (2) frequency — this test validates one send, not twelve a month; fatigue and unsubscribes need their own experiment with a guardrail metric; (3) creative iterations against the men's-email champion.

### "What did this test NOT measure?"

Three things: effects beyond two weeks (long-run brand impact, fatigue), unsubscribes and complaints (not in this dataset — in a live rerun we'd make them guardrail metrics), and profit margin (we measured revenue; margin translation needs finance's input on category mix).

---

## The big picture

### "Why do we need this ceremony for every marketing decision?"

Because intuition about what customers want has a documented failure rate near 50% — Microsoft found only about a third of its expert-designed features actually improved metrics when tested. The ceremony is cheap (this entire test cost ~$40 of email sends and two weeks); shipping the wrong default to 10 million people is not.

### "What's the one-slide version for the board?"

> We ran a randomized test on 64,000 customers: men's merchandise email vs. women's vs. no email.
> The men's email **doubled purchase rate** (0.57% → 1.25%) and added **$0.77 revenue per send** — validated by five independent statistical checks.
> Scaled to our file, this decision is worth **$58–92M/year** at ~$120K cost.
> **Recommendation: ship the men's email as default; test category-targeting next quarter.**
