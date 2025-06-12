import uuid
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, status, Request

from src.agents.customer_info_extractor import CustomerInfoExtractor
from src.models.response import ExtractionStatus
from src.models.request import ExtractionRequest

# Import the background task functions
from src.tasks.extraction_tasks import perform_extraction_task, task_results

logger = logging.getLogger("FastAPI_ExtractionRoutes")

# Initialize the router
router = APIRouter(prefix="/extract", tags=["Extraction"])


@router.post(
    "/submit", response_model=ExtractionStatus, status_code=status.HTTP_202_ACCEPTED
)
async def submit_extraction(
    request: Request, background_tasks: BackgroundTasks, payload: ExtractionRequest
):
    """
    Submits a transcript for customer information extraction as a background task.
    Returns a task ID to check the status later.
    """
    task_id = str(uuid.uuid4())

    # Get the extractor instance from the application state (set in main.py lifespan)
    extractor_instance: "CustomerInfoExtractor" = request.app.state.extractor

    if extractor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CustomerInfoExtractor not initialized.",
        )

    # Initialize task status in the shared dictionary
    task_results[task_id] = ExtractionStatus(
        task_id=task_id, status="pending", message="Extraction task queued."
    )

    background_tasks.add_task(
        perform_extraction_task,
        logger,
        task_id,
        payload.transcript,
        payload.summarise_first,
        extractor_instance,
        task_results,
    )

    logger.info(f"Task {task_id}: Extraction task added to background.")
    return task_results[task_id]


@router.get("/status/{task_id}", response_model=ExtractionStatus)
async def get_extraction_status(task_id: str):
    """
    Retrieves the status and results of a submitted extraction task.
    """
    status_entry = task_results.get(task_id)
    if not status_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found.",
        )
    return status_entry
