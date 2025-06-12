from pydantic import BaseModel, Field, FieldValidationInfo
from typing import Literal, Optional
from datetime import date


class ClientPersonalDetails(BaseModel):
    """Pydantic model for Client 1 or Client 2 Personal Details."""

    first_name: Optional[str] = Field(
        default=None, description="First name of the client."
    )

    last_name: Optional[str] = Field(
        default=None,
        description="Last name of the client.",
    )

    middle_names: Optional[str] = Field(
        default=None,
        description="Middle names of the client, if any.",
    )

    title: Optional[str] = Field(
        default=None, description="Title of the client (e.g., Mr., Ms., Dr.)"
    )

    known_as: Optional[str] = Field(
        default=None,
        description="Name the client is commonly known as.",
    )

    pronouns: Optional[str] = Field(
        default=None,
        description="Client's preferred pronouns, if they explicitly mentioned it.",
    )

    date_of_birth: Optional[date] = Field(
        default=None,
        description="Date of birth of the client.",
    )

    legal_sex: Optional[Literal["Male", "Female"]] = Field(
        default=None,
        description="Client's legal sex as per official documents.",
    )

    marital_status: Optional[
        Literal["Single", "Married/Partnered", "Divorced/Separated", "Widowed"]
    ] = Field(
        default=None,
        description="Client's current marital status.",
    )

    place_of_birth: Optional[str] = Field(
        default=None,
        description="Place where the client was born: City, Country",
    )

    nationality: Optional[str] = Field(
        default=None, description="Client's nationality."
    )

    gender: Optional[Literal["Male", "Female", "Non-binary/Other"]] = Field(
        default=None, description="Client's self-identified gender."
    )

    home_phone: Optional[str] = Field(
        default=None,
        description="Client's home phone number.",
    )

    mobile_phone: Optional[str] = Field(
        default=None,
        description="Client's mobile phone number.",
    )

    email_address: Optional[str] = Field(
        default=None,
        description="Client's email address.",
    )
