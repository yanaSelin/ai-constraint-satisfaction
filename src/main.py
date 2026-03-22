"""Interactive console for ReAct multi-hop constraint evaluation.

Usage:
    python src/main.py
"""

import itertools
import logging
import os
import sys
import threading
import time

from src.client import create_client, chat
from src.models import ReActEval
from src.prompts import format_react_messages, format_steps_for_display

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    filename="checker.log",
)
logger = logging.getLogger(__name__)


def _spinner(stop_event: threading.Event) -> None:
    """Show animated spinner while waiting for API response."""
    frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not stop_event.is_set():
        sys.stdout.write(f"\r  Analysing {next(frames)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()


def analyze(client, rules_text: str, data_csv: str) -> ReActEval:
    """Run ReAct constraint violation detection.

    Args:
        client: AzureOpenAI client.
        rules_text: Business rules as a newline-separated string.
        data_csv: Dataset in CSV format.

    Returns:
        Parsed ReActEval with reasoning chain and violations.
    """
    rules_count = len([l for l in rules_text.splitlines() if l.strip()])
    rows_count = max(0, len([l for l in data_csv.splitlines() if l.strip()]) - 1)
    logger.info("Starting analysis: %d rules, %d data rows", rules_count, rows_count)
    result = chat(client, format_react_messages(rules_text, data_csv), response_format=ReActEval, temperature=0.0)
    assert result is not None, "API returned no parsed result"
    logger.info("Analysis complete: %d violations found (rules=%d, rows=%d)", len(result.violations), rules_count, rows_count)
    return result


def display_result(result: ReActEval) -> None:
    """Print reasoning chain and detected violations."""
    print()
    print(format_steps_for_display(result.steps))
    print(f"\nConfidence: {result.confidence}")
    print(f"\nViolations found: {len(result.violations)}")
    if result.violations:
        for v in result.violations:
            rules = ", ".join(v.rules_violated)
            print(f"  {v.row_id}  [{rules}]  — {v.reason}")
    else:
        print("  (none)")
    print()


def read_block(prompt: str) -> str:
    """Read multi-line input until empty line. Returns 'quit' immediately if typed."""
    lines: list[str] = []
    print(prompt)
    try:
        while True:
            line = input("> " if not lines else "  ").rstrip()
            if line.lower() in ("quit", "exit", "q"):
                return "quit"
            if not line:
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def main() -> None:
    """Run the interactive constraint compliance checker.

    Prompts for rules, then data. Press Enter on empty line to submit each block.
    Type 'quit' to exit.
    """
    client = create_client()
    print("Multi-hop Constraint Compliance Checker (ReAct).")
    print("Type 'quit' at any prompt to exit, or press Ctrl+C.\n")

    while True:
        rules_text = read_block("Enter business rules (one per line, empty line to finish):")
        if rules_text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            return
        if not rules_text:
            continue

        data_csv = read_block("\nEnter data table (CSV with header, empty line to finish):")
        if data_csv.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            return
        if not data_csv:
            continue

        try:
            stop = threading.Event()
            t = threading.Thread(target=_spinner, args=(stop,), daemon=True)
            t.start()
            try:
                result = analyze(client, rules_text, data_csv)
            finally:
                stop.set()
                t.join()
            display_result(result)
        except Exception:
            logger.exception("Analysis failed")
            print("Sorry, something went wrong. Please try again.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")

