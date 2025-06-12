from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date


class Dependant(BaseModel):
    """Pydantic model for Dependants & Children."""

    name: str = Field(..., description="Full name of the dependant or child.")
    date_of_birth: date = Field(
        ...,
        description="Date of birth of the dependant or child.",
    )
    dependent_until: Optional[date] = Field(
        description="Date until which the individual is considered a dependant IF EXPLICITLY MENTIONED during the call.",
    )

class Dependants(BaseModel):
    """Pydantic model for a list of Dependant entries."""

    value: Optional[List[Dependant]] = Field(
        default_factory=list,
        description="A list of dependant entries.",
    )
