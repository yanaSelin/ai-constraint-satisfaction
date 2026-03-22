"""AzureOpenAI client factory and retry-aware chat helper."""

import logging
import os
import time

from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, RateLimitError

load_dotenv()
logger = logging.getLogger(__name__)


def create_client() -> AzureOpenAI:
    """Create an AzureOpenAI client from environment variables.

    Fails fast if required variables are missing.
    """
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )


def chat(
    client: AzureOpenAI,
    messages: list[dict],
    response_format: type | None = None,
    temperature: float | None = None,
    max_retries: int = 3,
):
    """Send a chat completion request with exponential backoff on rate limits.

    Args:
        client: AzureOpenAI client instance.
        messages: Chat messages list.
        response_format: Pydantic model for structured output, or None.
        temperature: Sampling temperature, or None to use model default.
        max_retries: Maximum retry attempts on transient errors.

    Returns:
        Parsed Pydantic model if response_format is set, else raw message content.
    """
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    temp_kwargs = {"temperature": temperature} if temperature is not None else {}

    for attempt in range(max_retries):
        try:
            if response_format is not None:
                completion = client.beta.chat.completions.parse(
                    model=deployment,
                    messages=messages,
                    response_format=response_format,
                    **temp_kwargs,
                )
                return completion.choices[0].message.parsed
            else:
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    **temp_kwargs,
                )
                return completion.choices[0].message.content

        except RateLimitError as exc:
            wait = 2 ** (attempt + 1)
            logger.warning("Rate limited (attempt %d/%d), retrying in %ds: %s", attempt + 1, max_retries, wait, exc)
            if attempt + 1 == max_retries:
                raise
            time.sleep(wait)

        except APIError as exc:
            logger.error("API error (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt + 1 == max_retries:
                raise
            time.sleep(2 ** attempt)
