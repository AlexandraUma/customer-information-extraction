from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# ================================================= Incomes =============================================
class Income(BaseModel):
    """Pydantic model for an Income entry."""

    owner: str = Field(
        ...,
        description="Owner of the income (e.g., Client 1, Client 2, Joint).",
    )
    name: str = Field(..., description="Name or source of the income.")
    amount: float = Field(..., description="Amount of the income.")
    frequency: str = Field(
        ...,
        description="Frequency of the income (e.g., Monthly, Annually).",
    )
    net_gross: Literal["Net", "Gross"] = Field(
        ...,
        description="Indicates if the income amount is Net or Gross.",
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Timeframe related to the income (e.g., Current, Projected).",
    )


class Incomes(BaseModel):
    """Pydantic model for a list of Income entries."""

    value: Optional[List[Income]] = Field(
        default_factory=list,
        description="A list of income entries.",
    )


# ============================================== Expenditures ===================================================


class Expense(BaseModel):
    """Pydantic model for a Expense entry."""

    owner: Literal["Client 1", "Client 2", "Joint"] = Field(
        description="Owner of the expense (One of Client 1, Client 2, Joint).",
    )
    name: str = Field(description="Name or description of the expense.")
    amount: float = Field(description="Amount of the expense.")
    frequency: str = Field(
        description="Frequency of the expense (e.g., Monthly, Annually).",
    )
    priority: Literal["Low", "Medium", "High"] = Field(
        description="Priority level of the expense, one of  High, Medium, Low).",
    )
    expense_type: Literal[
        "Loan Repayment",
        "Housing",
        "Motoring",
        "Personal",
        "Professional",
        "Miscellaneous",
    ] = Field(
        description='A broad category the expense falls into. One of "Loan Repayment", "Housing", "Motoring", "Personal", "Professional", or "Miscellaneous"',
    )

    timeframe: Optional[str] = Field(
        default=None,
        description="Timeframe related to the expense (e.g., Current, Projected).",
    )


class Expenses(BaseModel):
    """Pydantic model for a list of Expense entries."""

    value: Optional[List[Expense]] = Field(
        default_factory=list,
        description="A list of expense entries.",
    )
