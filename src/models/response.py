from typing import Optional
from pydantic import BaseModel

from src.models.cif import CustomerInformationForm


class ExtractionStatus(BaseModel):
    task_id: str
    status: str
    message: str
    result: Optional[CustomerInformationForm] = None
    error: Optional[str] = None
