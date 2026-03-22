"""Naive single-pass baseline prompt — the control for evaluation."""

import logging

from openai import AzureOpenAI

from src.client import chat
from src.models import BaselineEval
from src.prompts import format_baseline_messages

logger = logging.getLogger(__name__)


def analyze(client: AzureOpenAI, rules_text: str, data_csv: str) -> BaselineEval:
    """Run zero-shot baseline constraint violation detection.

    Args:
        client: AzureOpenAI client.
        rules_text: Business rules as a numbered list string.
        data_csv: Dataset in CSV format.

    Returns:
        BaselineEval with detected violations.
    """
    messages = format_baseline_messages(rules_text, data_csv)
    return chat(client, messages, response_format=BaselineEval, temperature=0.0)
