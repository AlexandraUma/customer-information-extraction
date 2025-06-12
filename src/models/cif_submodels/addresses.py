from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class CurrentAddressUK(BaseModel):
    """Pydantic model for Current Address."""

    ownership_status: str = Field(
        default=None,
        description="Status of home ownership (e.g., Owned, Rented).",
    )

    postcode: str = Field(default=None, description="Postcode of the current address.")

    house_name_number: str = Field(
        default=None,
        description="House name or number of the current address.",
    )

    street_name: str = Field(
        default=None,
        description="Street name of the current address.",
    )

    address_line_3: Optional[str] = Field(
        default=None,
        description="Third line of the current address, if applicable.",
    )

    address_line_4: Optional[str] = Field(
        default=None,
        description="Fourth line of the current address, if applicable.",
    )

    town_or_city: str = Field(
        default=None,
        description="Town or city of the current address.",
    )

    county: Optional[str] = Field(
        default=None, description="County of the current address."
    )

    move_in_date: Optional[date] = Field(
        default=None,
        description="Date the client moved into the current address.",
    )
