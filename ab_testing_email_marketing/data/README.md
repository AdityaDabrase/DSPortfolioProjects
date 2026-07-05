# Data

## 1. `raw/hillstrom.csv` — The Hillstrom Email Experiment (primary dataset)

A **real randomized controlled experiment** run by a retail company and released
publicly by Kevin Hillstrom (former VP of Database Marketing, Nordstrom / Eddie Bauer)
as the "MineThatData E-Mail Analytics And Data Mining Challenge" (March 2008).

- **Source:** http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- **Announcement:** https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
- **Population:** 64,000 customers who purchased within the last 12 months
- **Design:** customers randomly assigned to one of three arms:
  - `Mens E-Mail` — received an email featuring men's merchandise (~21,307)
  - `Womens E-Mail` — received an email featuring women's merchandise (~21,387)
  - `No E-Mail` — control group, received nothing (~21,306)
- **Outcome window:** two weeks following the email send

### Data dictionary

| Column | Type | Description | Measured |
|---|---|---|---|
| `recency` | int | Months since last purchase (1–12) | pre-treatment |
| `history_segment` | str | Bucketed dollar value spent in the past year | pre-treatment |
| `history` | float | Actual dollars spent in the past year | pre-treatment |
| `mens` | int (0/1) | Bought men's merchandise in the past year | pre-treatment |
| `womens` | int (0/1) | Bought women's merchandise in the past year | pre-treatment |
| `zip_code` | str | Urban / Suburban / Rural | pre-treatment |
| `newbie` | int (0/1) | New customer in the past 12 months | pre-treatment |
| `channel` | str | Purchase channel in the past year: Phone / Web / Multichannel | pre-treatment |
| `segment` | str | **Treatment assignment** (the experiment arm) | treatment |
| `visit` | int (0/1) | Visited the website within two weeks | outcome |
| `conversion` | int (0/1) | Purchased within two weeks | outcome |
| `spend` | float | Dollars spent within two weeks | outcome |

The pre-treatment columns matter for two reasons: they let us **verify the
randomization worked** (groups should look identical on these), and they enable
**segmentation** (which customers respond to which email).

## 2. `raw/marketing_ab.csv` — Kaggle "Marketing A/B Testing" (auditor showcase)

A large digital-advertising experiment: users saw either real ads (`ad`) or a
public service announcement (`psa`), and conversion was tracked.

- **Source:** https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing
- **Rows:** 588,101 users
- **Columns:** `user id`, `test group` (ad/psa), `converted` (bool),
  `total ads`, `most ads day`, `most ads hour`

We use this dataset **deliberately because it is flawed**: the split is roughly
96% ad / 4% PSA. Our `experiment_auditor.py` flags this automatically, which
demonstrates why you audit an experiment's design before trusting its p-values.

## Licensing / provenance note

Both datasets were released publicly for analysis and education. The Hillstrom
data is the de-facto canonical benchmark for email experimentation and uplift
modeling, which is useful here: published results exist to sanity-check our
numbers against.
