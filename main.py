import logging
from typing import Optional, Dict
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.settings import settings
from src.agents.customer_info_extractor import CustomerInfoExtractor
from src.routes import extraction


# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NevisWealth_App")

# --- Application Lifespan Context ---
app_extractor_instance: Optional[CustomerInfoExtractor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")

    # Initialize CustomerInfoExtractor
    extractor_instance = CustomerInfoExtractor(app_settings=settings, logger=logger)
    logger.info("CustomerInfoExtractor initialized.")

    # Attach the extractor instance to the app's state for access in routes
    app.state.extractor = extractor_instance
    logger.info("CustomerInfoExtractor attached to app state.")

    yield # Application starts here

    # Clean up resources when the app shuts down (if any)
    logger.info("Shutting down FastAPI application.")
    # Detach to clean up
    app.state.extractor = None


app = FastAPI(
    title="Customer Information Extraction Service",
    description="API for extracting customer information from meeting transcripts, with optional summarization.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Include your routers ---
app.include_router(extraction.router)


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Customer Information Extraction API. Use /docs for API documentation."
    }
