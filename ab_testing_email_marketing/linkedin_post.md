# LinkedIn Post Draft

**Suggested image:** `assets/conversion_lift_ci.png` (the hero chart), or a screenshot of the auditor's FAIL verdict from `assets/audit_marketing_ab.txt` if leading with the tool angle.

---

## The post

One email test. A $90 million decision.

A retailer split 64,000 customers randomly three ways: a men's merchandise email, a women's email, or silence.

Two weeks later, the men's email had DOUBLED the purchase rate — 1.25% vs 0.57% for the no-email group. That's +$0.77 in revenue per send, for a send cost of a tenth of a cent.

Scaled to a 10-million-customer list with monthly campaigns, picking the right email is worth $58–92M a year. The test that found it cost about $40.

But here's what surprised me most while building this analysis: the statistics were the easy part. The hard part is everything that can silently invalidate a test before you ever compute a p-value — traffic splits that don't match the design, groups that weren't really random, results declared "significant" because someone stopped the test the day the dashboard looked good.

So I built an Experiment Auditor: a small tool that grades any A/B test CSV on five checks before you're allowed to believe its result. I pointed it at another public ad-campaign dataset (588k users)... and it flagged the experiment as invalid on check #1. The traffic split didn't match an equal-allocation design — a detail most published analyses of that dataset never mention.

Big tech learned this lesson the expensive way: Microsoft reports ~6% of its experiments fail that same check and get thrown away.

The takeaway for anyone spending real money on marketing: the question isn't "was the result significant?" It's "was the experiment valid?" — and that's checkable, automatically.

Full project — analysis, the auditor tool, and a plain-English guide to A/B testing theory — on GitHub: [link]

#DataScience #ABTesting #Experimentation #MarketingAnalytics #Statistics

---

## Why the post is written this way (notes to self)

- **Hook = a dollar figure, not a method.** "One email test. A $90M decision." earns the next line; "I did hypothesis testing" doesn't.
- **Exactly one statistic per paragraph.** 1.25% vs 0.57% is the only comparison a scroller needs; CIs and Holm corrections live in the repo.
- **The twist is the tool, not the math.** "My auditor invalidated a public dataset" is a story; "I applied a chi-square test" is homework.
- **Credibility borrow:** the Microsoft 6% figure signals this is how serious companies operate.
- **CTA is the repo**, where technical readers find the depth (and the docs prove it wasn't a tutorial walkthrough).
