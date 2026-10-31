from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ApplicantIn(BaseModel):
    """Raw applicant fields. JSON keys use the original dataset column names (aliases)."""
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"example": {
        "RevolvingUtilizationOfUnsecuredLines": 0.45, "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0, "DebtRatio": 0.31,
        "MonthlyIncome": 5400, "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 2}})

    utilization: float = Field(alias="RevolvingUtilizationOfUnsecuredLines", ge=0)
    age: int = Field(ge=18, le=110)
    past_due_30_59: int = Field(alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0, le=20)
    debt_ratio: float = Field(alias="DebtRatio", ge=0)
    monthly_income: Optional[float] = Field(alias="MonthlyIncome", default=None, ge=0)
    open_credit_lines: int = Field(alias="NumberOfOpenCreditLinesAndLoans", ge=0, le=100)
    late_90: int = Field(alias="NumberOfTimes90DaysLate", ge=0, le=20)
    real_estate_loans: int = Field(alias="NumberRealEstateLoansOrLines", ge=0, le=100)
    past_due_60_89: int = Field(alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0, le=20)
    dependents: Optional[int] = Field(alias="NumberOfDependents", default=None, ge=0, le=20)


class Factor(BaseModel):
    feature: str
    value: float
    contribution: float          # SHAP value in log-odds; positive pushes toward default
    direction: str               # "increases risk" | "decreases risk"


class PredictionOut(BaseModel):
    default_probability: float
    risk_tier: str               # Low | Medium | High
    flagged: bool                # probability >= operating threshold
    top_factors: List[Factor]
    model_version: str
    prediction_id: Optional[int] = None
