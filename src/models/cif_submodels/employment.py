from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import date


class ClientEmploymentDetails(BaseModel):
    """Pydantic model for Client 1 or Client 2 Employment Details."""

    country_domiciled: str = Field(
        ...,
        description="Country where the client is domiciled for tax purposes.",
    )
    resident_for_tax: str = Field(
        ...,
        description="Indicates if the client is resident for tax purposes.",
    )
    national_insurance_number: str = Field(
        ...,
        description="Client's National Insurance number.",
    )
    employment_status: Literal["Employed", "Self-employed", "Retired"] = Field(
        ...,
        description="Client's current employment status (e.g., Employed, Self-employed, Retired).",
    )
    desired_retirement_age: int = Field(
        ...,
        description="Client's desired age of retirement.",
    )
    occupation: str = Field(..., description="Client's current occupation.")
    employer: str = Field(..., description="Name of the client's employer.")
    employment_started: str = Field(
        ...,
        description="Date when the client's current employment started.",
    )
    highest_rate_of_tax_paid: Optional[str] = Field(
        default=None,
        description="Highest rate of income tax paid by the client.",
    )
    notes: Optional[str] = Field(
        default=None, description="Any additional notes regarding employment."
    )
