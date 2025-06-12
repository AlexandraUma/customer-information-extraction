import json
import logging
import asyncio
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, List, Dict, Optional, Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.settings import settings

# --- Constants ---
# We retry on rate limits, timeouts, connection issues, and temporary server errors
RETRYABLE_OPENAI_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
# Exceptions that can occur during response parsing or validation
PARSING_VALIDATION_EXCEPTIONS = (json.JSONDecodeError, ValidationError)

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


class OpenAIClientWrapper:
    """
    A class to wrap the OpenAI client, providing initialization, retryable completion methods,
    and concurrency control.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        concurrency_limit: int = 10,  # Changed default to non-Optional and type hinted
    ):
        """
        Initializes an Async OpenAI client.

        Args:
            logger: The logger object.
            model: The model name to use. Defaults to settings.LLM_NAME if not provided.
            api_key: The API key for authentication. Defaults to settings.LLM_API_KEY if not provided.
            base_url: The base URL for the API endpoint. Defaults to settings.LLM_BASE_URL if not provided.
            concurrency_limit: The maximum number of concurrent LLM requests allowed.
        """
        self.logger = logger
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_NAME
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self._client: Optional[AsyncOpenAI] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """
        Internal method to initialize the Async OpenAI client.
        Logs an error if initialization fails.
        """
        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self.logger.info(
                f"Async OpenAI client initialized successfully for base_url: {self.base_url}"
            )
        except Exception as exc:
            self.logger.error(
                f"Failed to initialize Async OpenAI client: {exc}. Details: {exc.__cause__}",
                exc_info=True,  # Added exc_info for full traceback in logs
            )
            self._client = None

    @property
    def client(self) -> Optional[AsyncOpenAI]:
        """
        Returns the initialized Async OpenAI client, or None if initialization failed.
        """
        return self._client

    @retry(
        wait=wait_exponential(multiplier=0.2, min=1, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(
            RETRYABLE_OPENAI_EXCEPTIONS + PARSING_VALIDATION_EXCEPTIONS
        ),
        reraise=True,  # Ensure exceptions are re-raised for tenacity to catch
    )
    async def _get_completion_with_retries(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        as_json: bool,
        schema: Optional[Type[T]],
    ) -> Optional[Union[T, Dict, str]]:
        """
        Internal method to get a completion from the LLM with retry logic.
        This method handles the direct API call and response parsing/validation.
        """
        if not self._client:
            self.logger.error("Async OpenAI client is not initialized.")
            return None
        if not messages:
            self.logger.warning("Messages list is empty for completion request.")
            return None

        self.logger.debug(f"Requesting completion from model: {self.model}.")
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=(
                    {"type": "json_object"} if as_json else None
                ),
            )

            raw_content = (
                response.choices[0].message.content if response.choices else None
            )
            if not raw_content:
                self.logger.warning(
                    "Received response but no content found in choices."
                )
                return None

            if as_json:
                try:
                    json_data = json.loads(raw_content)
                    if schema:
                        return schema(**json_data)
                    return json_data
                except (json.JSONDecodeError, ValidationError) as e:
                    self.logger.error(
                        f"JSON parsing or Pydantic validation failed: {e}"
                    )
                    raise  # Re-raise to trigger tenacity retry for parsing/validation issues
            return raw_content

        except RETRYABLE_OPENAI_EXCEPTIONS as exc:
            self.logger.warning(
                f"A retryable OpenAI error occurred: {repr(exc)}. Retrying...",
                exc_info=True,
            )
            raise  # Re-raise to trigger tenacity retry
        except Exception as exc:
            self.logger.error(
                f"An unexpected error occurred during completion: {repr(exc)}",
                exc_info=True,
            )
            raise  # Re-raise non-retryable exceptions

    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        as_json: bool = False,
        schema: Optional[Type[T]] = None,
    ) -> Optional[Union[T, Dict, str]]:
        """
        Gets a completion from the LLM with optional JSON parsing and schema validation,
        applying concurrency control.

        Args:
            messages: The conversation history.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens for the completion.
            as_json: Whether to parse the response as JSON. If True, sets response_format to "json_object".
            schema: Optional Pydantic schema to validate the parsed response.

        Returns:
            - If schema is given: a validated Pydantic object.
            - If as_json is True: parsed JSON as dict.
            - Else: raw string response.
            Returns None if the client is not initialized or an unrecoverable error occurs.
        """
        async with self._semaphore:
            self.logger.debug("Semaphore acquired for completion.")
            try:
                return await self._get_completion_with_retries(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    as_json=as_json,
                    schema=schema,
                )
            except Exception as exc:
                self.logger.error(
                    f"Final attempt for completion failed after retries: {repr(exc)}",
                    exc_info=True,
                )
                return None

    async def get_batch_completion(
        self,
        all_messages: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        as_json: bool = False,
        schema: Optional[Type[T]] = None,
    ) -> List[Optional[Union[T, Dict, str]]]:
        """
        Runs multiple get_completion calls concurrently using the internal semaphore.

        Args:
            all_messages: A list of message batches, each representing a separate request.
            temperature: LLM temperature for all requests in the batch.
            as_json: Whether to parse responses as JSON for all requests in the batch.
            schema: Optional Pydantic schema to validate responses for all requests in the batch.

        Returns:
            List of responses (validated objects, dicts, or strings).
            Each element corresponds to the input messages list. None for failed requests.
        """
        if not all_messages:
            self.logger.warning("Batch completion called with empty list of messages.")
            return []

        self.logger.info(f"Starting batch completion for {len(all_messages)} requests.")
        tasks = [
            self.get_completion(messages, temperature, max_tokens, as_json, schema)
            for messages in all_messages
        ]
        results = await asyncio.gather(
            *tasks, return_exceptions=False
        )  # return_exceptions=False means awaited exceptions are propagated
        self.logger.info(f"Completed batch completion with {len(results)} results.")
        return results


if __name__ == "__main__":
    import asyncio
    import logging

    # Configure basic logging for demonstration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    test_logger = logging.getLogger(__name__)

    class SimpleResponseSchema(BaseModel):
        summary: str
        keywords: List[str]

    async def main():
        test_logger.info("--- Starting OpenAIClientWrapper demonstration ---")

        # --- Test Case 1: Basic String Completion ---
        test_logger.info("\n--- Test Case 1: Basic String Completion ---")
        llm_client_wrapper = OpenAIClientWrapper(
            logger=test_logger,
            model="gpt-4o-mini", 
            concurrency_limit=10,
        )

        if llm_client_wrapper.client:
            conversation_history_str = [
                {
                    "role": "user",
                    "content": "Explain the concept of exponential backoff for API retries in simple terms.",
                }
            ]
            try:
                completion_response_str = await llm_client_wrapper.get_completion(
                    messages=conversation_history_str, temperature=0.5
                )
                if completion_response_str:
                    print("\n[TEST 1] Completion Response (String):")
                    print(
                        completion_response_str[:200] + "..."
                    )
                else:
                    print("\n[TEST 1] No completion or content received.")
            except Exception as e:
                test_logger.error(
                    f"[TEST 1] An error occurred during string completion: {e}"
                )
        else:
            test_logger.error("OpenAI client wrapper failed to initialize for Test 1.")

        # --- Test Case 2: JSON Completion with Schema Validation ---
        test_logger.info(
            "\n--- Test Case 2: JSON Completion with Schema Validation ---"
        )
        if llm_client_wrapper.client:
            conversation_history_json = [
                {
                    "role": "user",
                    "content": "Provide a JSON object with a 'summary' of quantum computing and 'keywords' as a list.",
                }
            ]
            try:
                completion_response_json = await llm_client_wrapper.get_completion(
                    messages=conversation_history_json,
                    as_json=True,
                    schema=SimpleResponseSchema,
                    temperature=0.7,
                )
                if completion_response_json:
                    print("\n[TEST 2] Completion Response (JSON with Schema):")
                    print(completion_response_json)
                    print(f"Type of response: {type(completion_response_json)}")
                    print(
                        f"Is Pydantic model: {isinstance(completion_response_json, SimpleResponseSchema)}"
                    )
                else:
                    print("\n[TEST 2] No JSON completion or content received.")
            except Exception as e:
                test_logger.error(
                    f"[TEST 2] An error occurred during JSON completion with schema: {e}"
                )
        else:
            test_logger.error("OpenAI client wrapper failed to initialize for Test 2.")

        # --- Test Case 3: Batch Completion ---
        test_logger.info("\n--- Test Case 3: Batch Completion ---")
        if llm_client_wrapper.client:
            batch_messages = [
                [{"role": "user", "content": "What is Python?"}],
                [{"role": "user", "content": "What is asyncio?"}],
                [{"role": "user", "content": "Explain OOP in Python."}],
                [{"role": "user", "content": "Write a very short poem about stars."}],
            ]
            try:
                batch_responses = await llm_client_wrapper.get_batch_completion(
                    all_messages=batch_messages, temperature=0.4
                )
                print(
                    f"\n[TEST 3] Batch Completion Responses ({len(batch_responses)} total):"
                )
                for i, res in enumerate(batch_responses):
                    print(f"  Request {i+1}: {'Success' if res else 'Failed'}")
                    if res:
                        print(f"    Content: {str(res)[:100]}...")
            except Exception as e:
                test_logger.error(
                    f"[TEST 3] An error occurred during batch completion: {e}"
                )
        else:
            test_logger.error("OpenAI client wrapper failed to initialize for Test 3.")

        test_logger.info("--- OpenAIClientWrapper demonstration finished ---")

    asyncio.run(main())
