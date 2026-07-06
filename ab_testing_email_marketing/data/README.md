# Data

## 1. `raw/hillstrom.csv` — the primary experiment (committed, 4 MB)

The Hillstrom "MineThatData" E-Mail Analytics Challenge dataset (2008): a **real randomized experiment** on 64,000 customers of a retailer, released publicly by Kevin Hillstrom. Each customer had purchased within the previous 12 months and was randomly assigned to one of three arms.

Source: [blog.minethatdata.com](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) — direct CSV:

```bash
curl -o raw/hillstrom.csv "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
```

### Data dictionary

| Column | Type | Timing | Meaning |
|---|---|---|---|
| `recency` | int (1–12) | pre-treatment | Months since last purchase |
| `history_segment` | category | pre-treatment | Past-year spend band (e.g. `2) $100 - $200`) |
| `history` | float | pre-treatment | Actual past-year spend in dollars |
| `mens` | 0/1 | pre-treatment | Bought men's merchandise in the past year |
| `womens` | 0/1 | pre-treatment | Bought women's merchandise in the past year |
| `zip_code` | category | pre-treatment | Urban / Surburban / Rural (sic — "Surburban" is in the source data) |
| `newbie` | 0/1 | pre-treatment | New customer within the past 12 months |
| `channel` | category | pre-treatment | Past purchase channel: Phone / Web / Multichannel |
| `segment` | category | **treatment** | Randomized arm: `Mens E-Mail` / `Womens E-Mail` / `No E-Mail` |
| `visit` | 0/1 | outcome (2 weeks) | Visited the website |
| `conversion` | 0/1 | outcome (2 weeks) | Made a purchase |
| `spend` | float | outcome (2 weeks) | Dollars spent |

The pre/post-treatment distinction matters: only pre-treatment columns may be used for balance checks and segmentation (see `docs/02_technical_deep_dive.md`, section 5).

## 2. `raw/marketing_ab.csv` — the auditor showcase (NOT committed, 21 MB)

Kaggle "Marketing A/B Testing" dataset: 588,101 users assigned ~96/4 to see product ads vs. public service announcements. Used only to demonstrate the Experiment Auditor flagging a Sample Ratio Mismatch when the non-obvious 96/4 design isn't declared.

Download (no Kaggle account needed, via `kagglehub`):

```bash
python -c "
import kagglehub, shutil, glob
path = kagglehub.dataset_download('faviovaz/marketing-ab-testing')
shutil.copy(glob.glob(path + '/**/*.csv', recursive=True)[0], 'raw/marketing_ab.csv')
"
```

| Column | Meaning |
|---|---|
| `user id` | Unique user identifier |
| `test group` | `ad` (saw product ads) or `psa` (saw public service announcements) |
| `converted` | True if the user purchased |
| `total ads` | Number of ads the user saw (post-treatment) |
| `most ads day` / `most ads hour` | When the user saw the most ads (post-treatment) |

## Licensing note

Both datasets are publicly released for analysis and education. The Hillstrom dataset was published explicitly as an open challenge dataset; cite Kevin Hillstrom / MineThatData when reusing.
