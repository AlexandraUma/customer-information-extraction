# Customer Information Form Filler

An LLM-based pipeline to extract the client’s answers to the questions from the transcript of a wealth-management fact-finding call, and automatically complete the Customer Information Form (CIF).

## Demo: Customer Information Extractor Application

This repository contains a multi-service application for extracting customer information from transcripts.

It consists of a few components:

- A **FastAPI backend** that performs the extraction using the pipeline. It receives extraction requests, initiates background extraction tasks, stores results, and, if a callback URL is provided, posts the extracted customer form to that URL upon task completion. You can try out the API directly using http://127.0.0.1:8000/docs
- A **Gradio-based user interface** for optionally testing the API. This offers an alternative way to test out the API.
- A **simple Flask webhook** to simulate callback handling. It sends extraction requests to the FastAPI backend and the user can refresh to get the results (TODO:get polling to work).

### Setup and Running Instructions

Follow these steps to set up and run the entire application locally.

#### 1. Prerequisites

- Python 3.8+
- `pip` (Python package installer)

#### 2. Environment Variables

Create a `.env` file in the root directory of your project (e.g., next to `fastapi_app.py` and `gradio_app.py`). See `.env.example` for an example.

#### 3. Installation

Install the requirements.txt file using pip.

#### 5. Running the Services

You will need three separate terminal windows/tabs to run all components simultaneously.

A. Run the Webhook Receiver (Flask App)
This service will listen for incoming data from the FastAPI backend when an extraction task is complete and a callback URL is specified.

Terminal 1:
Bash

`python webhook_receiver.py`

Expected Output: You should see output similar to \* Running on http://127.0.0.1:5000 (or another available port).
B. Run the FastAPI Backend
This is the core API that handles extraction requests and manages the background tasks.

Terminal 2: Bash

`uvicorn fastapi_app:app --reload --port 8000`

fastapi_app: Refers to your Python file (e.g., fastapi_app.py).
app: Refers to the FastAPI application instance within that file (e.g., app = FastAPI()).
--reload: Restarts the server automatically on code changes (useful for development).
--port 8000: Specifies the port for the FastAPI service.
Expected Output: You should see output indicating Uvicorn is running, typically on http://127.0.0.1:8000.

Testing the API (Optional):
While the FastAPI backend is running, you can open your web browser and navigate to:

http://127.0.0.1:8000/docs: This will open the interactive OpenAPI (Swagger UI) documentation for your FastAPI application, where you can explore and test your endpoints directly (e.g., try POST /extract/submit and GET /extract/status/{task_id}).
https://www.google.com/search?q=http://127.0.0.1:8000/redoc: Another generated API documentation style.

C. Run the Gradio App
This is the user interface you'll interact with in your browser.

Terminal 3: Bash

`python gradio_app.py`
Expected Output: You should see a message indicating the Gradio app is running, typically on http://127.0.0.1:7860.

6. Interacting with the Application
   Open your web browser and go to the address provided by the Gradio app (e.g., http://127.0.0.1:7860).

a. Enter a transcript in the provided textbox.
b. Click the "Submit for Extraction" button. The Gradio UI will show a Task ID and Submission Status. In Terminal 2 (FastAPI), you'll see logs indicating the task submission and the background extraction starting.
c. After a short delay (simulating extraction time), check Terminal 1 (Webhook Receiver). You should see logs indicating that it has received the callback POST request with the extracted customer data.
d. Back in the Gradio UI, you can click "Refresh Status" (or if you configured gr.Timer, it will update automatically) to see the "Extraction completed successfully" message and the "Extracted Information (JSON)" populated.

### Important Notes

- Callback Handler for Demo Only: The webhook_receiver.py (Flask app) is a simple, stateless webhook listener provided purely for demonstration purposes in a local development environment. In a production scenario, this would typically be a more robust service (e.g., another FastAPI application, a dedicated microservice, or a serverless function) that performs complex actions (e.g., saving data to a database, triggering further workflows) upon receiving the extracted customer form.

- Asynchronous Tasks: The FastAPI backend uses BackgroundTasks for the extraction process. For larger-scale, long-running, or more resilient tasks, you would typically integrate with a dedicated task queue system like Celery, RabbitMQ, or Redis Queue (RQ).

- Shared State: The task_results dictionary in the FastAPI app is an in-memory dictionary for storing task statuses and results. This is suitable for prototyping but should be replaced with a persistent store (e.g., Redis, PostgreSQL, Azure Table Storage) in a production environment to ensure data durability and scalability.

## Agent Evaluation Test Suite

Coming soon.