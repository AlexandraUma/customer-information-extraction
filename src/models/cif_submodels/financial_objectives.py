from typing import List, Optional
from pydantic import BaseModel, Field


class FinancialObjectives(BaseModel):
    """Pydantic model for Financial Objectives for the Wealth Management Client"""

    value: Optional[List[str]] = Field(
        default_factory=list,
        description="The client's overall financial and personal objectives.",
    )
