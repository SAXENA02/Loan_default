"""Shared constants so training, API and monitoring never disagree."""
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "MonthlyIncome_was_missing",
]
TARGET = "default"
MODEL_PATH = "models/model.joblib"
META_PATH = "models/model_meta.json"
