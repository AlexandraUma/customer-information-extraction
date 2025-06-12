from typing import Optional
from pydantic import BaseModel

from src.models.cif import CustomerInformationForm


class ExtractionRequest(BaseModel):
    transcript: str
    summarise_first: bool = False
