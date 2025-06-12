import gradio as gr
import requests
import os

# --- Configuration ---
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
SUBMIT_ENDPOINT = f"{FASTAPI_BASE_URL}/extract/submit"
STATUS_ENDPOINT = f"{FASTAPI_BASE_URL}/extract/status"

print(f"Gradio App: Configured to connect to FastAPI at: {FASTAPI_BASE_URL}")


def submit_extraction_request(transcript: str, summarize_first: bool):
    """
    Sends the extraction request to the FastAPI backend.
    """
    if not transcript.strip():
        # Correctly return empty strings/None for components that expect them
        return "Error: Transcript cannot be empty.", "", "", None

    payload = {
        "transcript": transcript,
        "summarize_first": summarize_first
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(SUBMIT_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        task_id = data.get("task_id")
        message = data.get("message", "Task submitted successfully.")
        # Ensure outputs match the expected types for Gradio components
        return message, task_id, "Processing...", None # Initialize JSON output as None or {}
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to FastAPI backend at {FASTAPI_BASE_URL}. Is it running and accessible?", "", "", None
    except requests.exceptions.RequestException as e:
        return f"Error submitting request to {SUBMIT_ENDPOINT}: {e}", "", "", None


def get_extraction_status(task_id: str):
    """
    Polls the FastAPI backend for the status of a given task ID.
    """
    if not task_id:
        return "Please submit a transcript first to get a Task ID.", "", {}, "" # Return empty dict for gr.Json

    try:
        response = requests.get(f"{STATUS_ENDPOINT}/{task_id}")
        response.raise_for_status()
        data = response.json()

        status = data.get("status", "unknown")
        message = data.get("message", "No message.")
        # Pass the Python dictionary directly to gr.Json
        extracted_result_data = data.get("result") # This is already a Python dict or None
        error = data.get("error", "") # Default to empty string for consistent output type


        # If there's an error in the response, we might want to also display it in result
        if error and not extracted_result_data:
            extracted_result_data = {"error": error, "status": status, "message": message}
        elif extracted_result_data is None: # Explicitly handle None case
            extracted_result_data = {} # gr.Json handles empty dict well

        return message, status, extracted_result_data, error # Pass the dictionary/None directly

    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to FastAPI backend at {FASTAPI_BASE_URL}. Is it running and accessible?", "Error", {}, "Connection Error"
    except requests.exceptions.RequestException as e:
        return f"Error checking status from {STATUS_ENDPOINT}/{task_id}: {e}", "Error", {}, str(e)


# --- Define Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# Customer Information Extractor")
    gr.Markdown("Submit a meeting transcript for extraction and check its status.")

    with gr.Row():
        with gr.Column():
            transcript_input = gr.Textbox(
                lines=10,
                label="Meeting Transcript",
                placeholder="Paste your meeting transcript here...",
                value="Client 1, John Doe, is 35 years old and lives at 123 High Street, London. He works as a Software Engineer at TechCorp. His main objective is to save for a house deposit. He has a monthly income of £3000 from his salary. Monthly expenses include £800 for rent and £200 for groceries."
            )
            summarize_checkbox = gr.Checkbox(
                label="Summarize Transcript First (Optional)",
                value=False
            )
            submit_btn = gr.Button("Submit for Extraction")

        with gr.Column():
            submit_message = gr.Textbox(label="Submission Status")
            task_id_output = gr.Textbox(label="Task ID")
            status_message = gr.Textbox(label="Current Status")
            status_check_btn = gr.Button("Refresh Status")
            extraction_result = gr.Json(label="Extracted Information (JSON)")
            error_output = gr.Textbox(label="Error Details", visible=True)


    # Connect components to functions
    submit_btn.click(
        submit_extraction_request,
        inputs=[transcript_input, summarize_checkbox],
        outputs=[submit_message, task_id_output, status_message, error_output]
    )

    status_check_btn.click(
        get_extraction_status,
        inputs=[task_id_output],
        outputs=[submit_message, status_message, extraction_result, error_output]
    )


if __name__ == "__main__":
    demo.launch()