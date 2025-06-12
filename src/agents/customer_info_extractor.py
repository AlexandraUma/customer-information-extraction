import time
import asyncio
import logging
from textwrap import dedent
from typing import List, Tuple, Type, Dict, Any, Optional

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_log,
    after_log,
    retry_if_exception_type,
)

# Import specific OpenAI exceptions
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

from agents import Agent, Runner, set_default_openai_key, set_tracing_disabled

from src.models.cif import (
    ClientPersonalDetails,
    CurrentAddressUK,
    Dependants,
    ClientEmploymentDetails,
    CustomerInformationForm,
)
from src.models.cif_submodels.income_and_expenditure import (
    Incomes,
    Expenses,
)
from src.models.cif_submodels.financial_objectives import FinancialObjectives

# Define your retryable exceptions
RETRYABLE_OPENAI_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


class CustomerInfoExtractor:
    """
    A class to extract various pieces of customer information from a transcript
    using multiple OpenAI agents in parallel. Optionally summarises the transcript
    before extraction.

    It manages the initialization of agents, defines extraction tasks, and
    orchestrates the parallel execution of these tasks while respecting
    API rate limits using an asynchronous semaphore.
    """

    def __init__(self, app_settings: Any, logger: logging.Logger):
        """
        Initializes the CustomerInfoExtractor with application settings and a logger.

        Args:
            app_settings: An object containing application configuration,
                          including LLM_NAME, LLM_API_KEY, DISABLE_TRACING,
                          and MAX_CONCURRENT_LLM_CALLS.
            logger: A logging.Logger instance for logging messages.
        """
        self.settings = app_settings
        self.logger = logger

        # Configure the OpenAI client and LLM settings
        self.model_name: str = app_settings.LLM_NAME
        set_default_openai_key(app_settings.LLM_API_KEY)
        set_tracing_disabled(app_settings.DISABLE_TRACING)

        # Initialize an asynchronous semaphore to manage concurrent LLM calls
        # This helps in respecting API rate limits.
        self.semaphore = asyncio.Semaphore(app_settings.MAX_CONCURRENT_LLM_CALLS)
        self.logger.info(
            f"Initialized LLM call semaphore with limit: {app_settings.MAX_CONCURRENT_LLM_CALLS}"
        )

        # Define the extraction tasks, mapping a field name to its Pydantic model
        # and a descriptive instruction for the agent.
        self.extraction_tasks: Dict[str, Tuple[Type[BaseModel], str]] = (
            self._setup_extraction_tasks()
        )

        # Initialize the Agent instances for each defined extraction task.
        self.extraction_agents: Dict[str, Agent] = self._initialize_extraction_agents()

        # Initialize the summarisation agent
        self.summary_agent: Agent = self._initialize_summary_agent()

    def _setup_extraction_tasks(self) -> Dict[str, Tuple[Type[BaseModel], str]]:
        """
        Defines the specific information extraction tasks.

        Each task is a tuple containing:
        - The Pydantic model type that the agent should output.
        - A concise description of the information to be extracted.

        Returns:
            A dictionary mapping a descriptive field name to its extraction task tuple.
        """
        return {
            "client_1_personal_details": (
                ClientPersonalDetails,
                "Extract Client 1 personal details.",
            ),
            "client_2_personal_details": (
                ClientPersonalDetails,
                "Extract Client 2 personal details.",
            ),
            "current_address": (
                CurrentAddressUK,
                "Extract the client's current residential address.",
            ),
            "dependants_and_children": (
                Dependants,
                "Extract details of all dependants and children.",
            ),
            "client_1_employment": (
                ClientEmploymentDetails,
                "Extract Client 1 employment details.",
            ),
            "client_2_employment": (
                ClientEmploymentDetails,
                "Extract Client 2 employment details.",
            ),
            "incomes": (Incomes, "Extract all income sources."),
            "expenses": (Expenses, "Extract all expenses."),
            "objectives": (
                FinancialObjectives,
                "Extract the client's overall financial and personal objectives.",
            ),
        }

    def _initialize_extraction_agents(self) -> Dict[str, Agent]:
        """
        Initializes an Agent instance for each defined extraction task.

        Each agent is configured with a specific name, detailed instructions
        for extraction, the LLM model to use, and the expected Pydantic
        output type.

        Returns:
            A dictionary mapping field names to their corresponding Agent instances.
        """
        agents = {}
        for field_name, (output_type, description) in self.extraction_tasks.items():
            agent_name = f"Extract_{field_name}"
            agents[field_name] = Agent(
                name=agent_name,
                instructions=dedent(
                    f"""
                You are an expert information extraction agent. Your task is to accurately extract specific customer information from a wealth management 'fact finding' meeting transcript.
                Focus solely on extracting the details related to '{field_name}'.
                {description}
                If the information is not present in the transcript, you should return an empty or default value for the given Pydantic model.
                For lists, return an empty list if no items are found.
                For optional fields or properties within a model, return null or omit the property if not found.
                """
                ),
                model=self.model_name,
                output_type=output_type,
            )
        return agents

    def _initialize_summary_agent(self) -> Agent:
        """
        Initializes a dedicated agent for summarising transcripts.

        Returns:
            An Agent instance configured for summarisation.
        """
        return Agent(
            name="Transcriptsummariser",
            instructions=(
                "You are an expert summarisation agent. "
                "Your task is to create a comprehensive summary of the provided meeting 'fact finding' wealth management meeting transcript.\n"
                "The summary should capture all key discussion points, decisions made, and client(s) information exchanged.\n"
                "Focus on information relevant to the client, their personal details, health, financial planning and wealth management, "
                "without omitting any client-related details. "
            ),
            model=self.model_name,
        )

    # Apply tenacity for retries, only for retryable exceptions
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before=before_log(logging.getLogger(__name__), logging.INFO),
        after=after_log(logging.getLogger(__name__), logging.INFO),
        retry=retry_if_exception_type(RETRYABLE_OPENAI_EXCEPTIONS),
    )
    async def _extract_single_field(
        self, field_name: str, transcript: str
    ) -> Optional[BaseModel]:
        """
        Runs a single agent to extract information for a specific field.
        This method uses an asynchronous semaphore to limit concurrent LLM calls.

        Args:
            field_name: The name of the field to extract (key in self.agents).
            transcript: The input text transcript from which to extract information.

        Returns:
            A Pydantic object representing the extracted data, or None if an
            error occurred during extraction (either non-retryable or all retries failed).
        """
        self.logger.info(f"Attempting to extract '{field_name}'...")
        start_time = time.time()

        async with self.semaphore:
            try:
                agent = self.extraction_agents[field_name]
                result = await Runner.run(agent, transcript)
                extracted_data = result.final_output

                self.logger.info(
                    f"Successfully extracted '{field_name}' in {time.time() - start_time:.2f} seconds."
                )
                return extracted_data
            except RETRYABLE_OPENAI_EXCEPTIONS as e:
                # Log and re-raise for tenacity to catch and retry
                self.logger.warning(
                    f"Retryable error extracting '{field_name}': {type(e).__name__} - {e}. Retrying..."
                )
                raise  # Tenacity will catch and retry this exception
            except Exception as e:
                # Catch any other exception (non-retryable, including BadRequestError, AuthenticationError)
                self.logger.error(
                    f"Non-retryable error extracting '{field_name}': {type(e).__name__} - {e}. "
                    f"Returning None after {time.time() - start_time:.2f} seconds."
                )
                return None  # This will be returned directly to asyncio.gather

    async def _summarise_transcript(self, transcript: str) -> Optional[str]:
        """
        summarises the given transcript using the dedicated summarisation agent.

        Args:
            transcript: The full transcript to summarise.

        Returns:
            The summarised text as a string, or None if summarisation fails.
        """
        self.logger.info("Attempting to summarise transcript...")
        start_time = time.time()
        async with self.semaphore:
            try:
                result = await Runner.run(self.summary_agent, transcript)
                summary = result.final_output
                self.logger.info(
                    f"Successfully summarised transcript in {time.time() - start_time:.2f} seconds."
                )
                return summary
            except Exception as e:
                self.logger.error(
                    f"Error summarising transcript: {e}. "
                    f"Proceeding with original transcript for extraction."
                )
                return None

    async def extract_customer_information(
        self, transcript: str, summarise_first: bool = True
    ) -> CustomerInformationForm:
        """
        Extracts all customer information from the provided transcript in parallel
        using multiple agents.

        This method orchestrates the concurrent execution of all defined extraction
        tasks, aggregates their results, and composes a final CustomerInformationForm object.

        Args:
            transcript: The input text transcript containing customer information.
            summarise_first: If True, the transcript will be summarised by an agent
                            and the summary used as input to the extractor agents.

        Returns:
            A CustomerInformationForm object populated with the extracted data.
            If a general error occurs during the overall extraction process,
            an empty CustomerInformationForm is returned.
        """
        text_for_extraction = transcript

        if summarise_first:
            summarised_text = await self._summarise_transcript(transcript)
            if summarised_text:
                self.logger.info(
                    "Using summarised transcript for information extraction."
                )
                text_for_extraction = summarised_text
                print(text_for_extraction)
            else:
                self.logger.warning(
                    "summarisation failed. Proceeding with original transcript."
                )

        extraction_tasks_coros: List[asyncio.Task] = []

        for field_name in self.extraction_tasks.keys():
            # Create a coroutine for each extraction task
            task = asyncio.create_task(
                self._extract_single_field(field_name, text_for_extraction)
            )
            extraction_tasks_coros.append(task)

        self.logger.info("Starting parallel extraction of customer information...")
        start_time = time.time()

        try:
            # Run all extraction tasks concurrently.
            # `return_exceptions=True` ensures that even if some tasks fail,
            # the `gather` call completes, and we can process successful results.
            results: List[Optional[BaseModel]] = await asyncio.gather(
                *extraction_tasks_coros, return_exceptions=True
            )

            # Compose the final CustomerInformationForm from the extracted data.
            extracted_data_dict = self._compose_customer_form(results)

            # Validate and construct the final Pydantic object.
            final_form = CustomerInformationForm(**extracted_data_dict)
            self.logger.info(
                "Successfully extracted and composed CustomerInformationForm in "
                f"{time.time() - start_time:.2f} seconds."
            )
            return final_form

        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during overall customer information extraction: {e}. "
                "Returning an empty CustomerInformationForm."
            )
            return CustomerInformationForm()

    def _compose_customer_form(self, results: List[Any]) -> Dict[str, Any]:
        """
        Composes a dictionary of extracted data suitable for initializing
        the CustomerInformationForm Pydantic model.

        It maps the results from the parallel extraction tasks back to their
        corresponding field names. Handles cases where extraction failed (None)
        or specific fields require special handling (e.g., list values).

        Args:
            results: A list of results from the parallel extraction tasks.
                     Each item can be a Pydantic model, an Exception, or None.

        Returns:
            A dictionary where keys are field names of CustomerInformationForm
            and values are the extracted Pydantic models or default values.
        """
        extracted_data_dict: Dict[str, Any] = {}
        field_names = list(self.extraction_tasks.keys())

        # It's crucial that `results` order matches `field_names` order,
        # which is guaranteed by `asyncio.gather` when tasks are added in order.
        for i, field_name in enumerate(field_names):
            result = results[i]

            if isinstance(result, Exception):
                # If an exception occurred during extraction, log it and set the field to None.
                self.logger.warning(
                    f"Extraction for '{field_name}' failed with an exception: {result}. "
                    "Setting field to None."
                )
                extracted_data_dict[field_name] = None
            elif result is None:
                # If extraction returned None (e.g., due to an internal error or no data found),
                # set the field to None.
                self.logger.info(
                    f"Extraction for '{field_name}' returned None. Setting field to None."
                )
                extracted_data_dict[field_name] = None
            else:
                if field_name in ["incomes", "expenses", "objectives"]:
                    extracted_data_dict[field_name] = (
                        result.value
                        if hasattr(result, "value") and result.value is not None
                        else []
                    )
                else:
                    extracted_data_dict[field_name] = result

        return extracted_data_dict


if __name__ == "__main__":
    import logging

    from tests.agent_eval.data.toy_transcripts import (
        transcript_with_all_info,
        transcript_with_missing_information,
    )
    from src.settings import settings

    # Configure basic logging
    logging.basicConfig(level=logging.INFO)
    test_logger = logging.getLogger(name="[CustomerInfoExtractor]")

    async def main():

        extractor = CustomerInfoExtractor(app_settings=settings, logger=test_logger)

        test_logger.info(
            "======= Extraction from test transcript with complete information ======="
        )
        customer_form = await extractor.extract_customer_information(
            transcript_with_all_info
        )
        print(customer_form.model_dump_json(indent=2))
        test_logger.info(
            "====================================================================\n"
        )

        test_logger.info(
            "======= Extraction from test transcript with missing info & no client 2 ======="
        )
        incomplete_form = await extractor.extract_customer_information(
            transcript_with_missing_information
        )
        print(incomplete_form.model_dump_json(indent=2))

    asyncio.run(main())
