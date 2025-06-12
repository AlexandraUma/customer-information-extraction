from typing import Dict

import httpx

from src.agents.customer_info_extractor import CustomerInfoExtractor
from src.models.response import ExtractionStatus
from src.settings import settings


# This dictionary will still need to be accessible from the routes to check status.
# We will pass a reference to this dictionary when we add the task.
# In a production setup, this would be a Redis client or Azure table or something.
task_results: Dict[str, ExtractionStatus] = {}


async def perform_extraction_task(
    logger,
    task_id: str,
    transcript: str,
    summarise_first: bool,
    extractor_instance: CustomerInfoExtractor,
    shared_task_results: Dict[str, ExtractionStatus],
):
    """
    This function runs in the background to perform the actual extraction.
    It receives the CustomerInfoExtractor instance and a shared results dictionary.
    """
    logger.info(f"Task {task_id}: Starting extraction.")

    # Use the passed shared_task_results dictionary
    if task_id not in shared_task_results:

        # This shouldn't happen if called correctly, but for safety
        shared_task_results[task_id] = ExtractionStatus(
            task_id=task_id, status="unknown", message="Task initialized."
        )

    shared_task_results[task_id].status = "processing"
    shared_task_results[task_id].message = "Extraction in progress."

    try:
        customer_form = await extractor_instance.extract_customer_information(
            transcript=transcript, summarise_first=summarise_first
        )
        shared_task_results[task_id].status = "completed"
        shared_task_results[task_id].message = "Extraction completed successfully."
        shared_task_results[task_id].result = customer_form
        logger.info(f"Task {task_id}: Extraction completed.")

        # Post the customer form to the callback URL if provided
        if settings.CALLBACK_URL:
            logger.info(
                f"Task {task_id}: Posting results to callback URL: {settings.CALLBACK_URL}"
            )
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        settings.CALLBACK_URL, json=customer_form.model_dump()
                    )  # Assuming Pydantic model, use .model_dump()
                    response.raise_for_status()  # Raise an exception for bad status codes
                    logger.info(
                        f"Task {task_id}: Successfully posted to callback URL. Status: {response.status_code}"
                    )

            except httpx.RequestError as e:
                logger.error(
                    f"Task {task_id}: Error posting to callback URL {settings.CALLBACK_URL}: {e}",
                    exc_info=True,
                )
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Task {task_id}: HTTP error posting to callback URL {settings.CALLBACK_URL}: {e.response.status_code} - {e.response.text}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"Task {task_id}: Extraction failed - {e}", exc_info=True)
        shared_task_results[task_id].status = "failed"
        shared_task_results[task_id].message = "Extraction failed."
        shared_task_results[task_id].error = str(e)
