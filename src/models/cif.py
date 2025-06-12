from pydantic import BaseModel, Field
from typing import Optional, List
from src.models.cif_submodels.personal_details import ClientPersonalDetails
from src.models.cif_submodels.addresses import CurrentAddressUK
from src.models.cif_submodels.employment import ClientEmploymentDetails
from src.models.cif_submodels.dependants import Dependants
from src.models.cif_submodels.income_and_expenditure import Income, Expense


class CustomerInformationForm(BaseModel):
    """
    Main Pydantic model for the entire Customer Information Form,
    combining all individual sheet models.

    For this first version, all fields are optional so that we can fail gracefully.
    TODO: Meaningful defaults for all fields.
    """

    # For now, we assume a maximum of two clients per meeting.
    client_1_personal_details: Optional[ClientPersonalDetails] = Field(
        default=None,
        description="Personal details for Client 1.",
    )

    client_2_personal_details: Optional[ClientPersonalDetails] = Field(
        default=None,
        description="Personal details for Client 2, if applicable.",
    )

    current_address: Optional[CurrentAddressUK] = Field(
        default=None,
        description="Details of the client's current residential address.",
    )

    dependants_and_children: Optional[Dependants] = Field(
        default=None,
        description="List of dependants and children associated with the client(s).",
    )

    client_1_employment: Optional[ClientEmploymentDetails] = Field(
        default=None,
        description="Employment details for Client 1.",
    )

    client_2_employment: Optional[ClientEmploymentDetails] = Field(
        default=None,
        description="Employment details for Client 2, if applicable.",
    )

    incomes: List[Income] = Field(
        default=[],
        description="List of all income sources for the client(s).",
    )

    expenses: List[Expense] = Field(default=[], description="List of expenses")

    objectives: List[str] = Field(
        default=[],
        description="Client's overall financial and personal objectives.",
    )


if __name__ == "__main__":
    import json

    schema_json = CustomerInformationForm.model_json_schema()
    print(json.dumps(schema_json, indent=2))
