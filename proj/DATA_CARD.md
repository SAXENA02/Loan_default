# Data Card — Loan Default Prediction (Project P_137)

## Source
- **Dataset**: "Give Me Some Credit" — Kaggle, 2011 competition, sponsored by Credit Fusion.
- **Original size**: 150,000 rows, 11 columns (1 target + 10 features).
- **License**: Public Kaggle competition dataset, freely available for research/educational use.

## What is REAL data
All of the following are the original, unmodified (aside from cleaning) values from the Kaggle dataset:

| Column | Description |
|---|---|
| `default` (orig. `SeriousDlqin2yrs`) | Target: 1 = serious delinquency within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Credit utilization ratio |
| `age` | Applicant age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Count of 30-59 day late payments |
| `DebtRatio` | Debt-to-income ratio |
| `MonthlyIncome` | Monthly income (with missingness imputed, see below) |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines/loans |
| `NumberOfTimes90DaysLate` | Count of 90+ day late payments |
| `NumberRealEstateLoansOrLines` | Real estate loans/lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | Count of 60-89 day late payments |
| `NumberOfDependents` | Number of dependents (missingness imputed, see below) |

**No default labels, financial ratios, or credit-history fields were synthesized.** The
core predictive signal in this project is 100% real historical data.

## What is SYNTHETIC (clearly added, and why)

The raw Kaggle dataset does not include fields required by the project brief
(NPA/business-impact calculation, drift-monitoring demo). These were added on top:

| Column | Why it was added | How it was generated |
|---|---|---|
| `loan_amount` | Needed to calculate the 30% NPA-reduction business metric | `(MonthlyIncome × 12 / 10) × random multiplier (~2.8x avg, noise added)`, clipped to a realistic $1,000–$500,000 range. Derived from *income*, not from the target — no leakage into the default-prediction model. |
| `loan_purpose` | Adds a categorical feature for SHAP demonstration and realism | Randomly sampled from 7 realistic categories (Debt Consolidation, Home Improvement, Auto, Education, Business, Personal, Medical) with realistic weighting. Not correlated with default risk. |
| `application_date` | Needed to construct an "old vs new" split for drift monitoring (EvidentlyAI), since the raw dataset is a single static snapshot with no time dimension | Randomly assigned across a 24-month window. The most recent 10% of records were carved out as a `drift_batch.csv` with **deliberate, documented distribution shifts** (income +8%, credit utilization +12%) applied to simulate realistic population drift over time. |

## Important usage notes
- `loan_amount` and `loan_purpose` must **not** be used as predictive inputs to the
  core default-risk classifier — they exist to support the business-metric narrative
  and categorical-feature demonstration, not to inflate or fabricate model performance.
- `application_date` and the drift shift are simulation artifacts for demonstrating the
  drift-monitoring pipeline, not real observed drift.

## Cleaning steps applied to real data
See `cleaning_log.json` for exact counts. Summary:
- Dropped 1 row with impossible age (< 18).
- Capped `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio` at their 99.5th
  percentile (known Kaggle data-entry outliers).
- Capped past-due count columns at 20 (values of 96/98 in the raw data are known
  data-entry artifacts, not real counts — a widely documented quirk of this dataset).
- `MonthlyIncome`: ~19.8% missing → added a `MonthlyIncome_was_missing` indicator flag,
  then imputed using median income within each age decile.
- `NumberOfDependents`: ~2.6% missing → imputed as 0 (conservative default assumption).

## File outputs
- `loan_data_full.csv` — full cleaned + augmented dataset (149,999 rows, 15 columns)
- `train.csv` — 108,000 rows, stratified training split
- `test.csv` — 27,000 rows, stratified holdout for evaluation
- `drift_batch.csv` — 14,999 rows, most recent batch with simulated drift, for the
  EvidentlyAI monitoring stage
