"""Pydantic schemas for structured LLM output."""

from typing import Literal

from pydantic import BaseModel, Field


class ReActStep(BaseModel):
    """Single analysis step in the ReAct chain."""

    thought: str = Field(description="Brief reasoning before acting (1-2 sentences)")
    action: Literal["PARSE_RULES", "MAP_COLUMNS", "EVALUATE", "FLAG", "VERIFY"] = Field(
        description="Which analysis action this step performs"
    )
    observation: str = Field(description="Findings from this step (2-3 sentences, cite column names and values)")


class ViolationResult(BaseModel):
    """A single detected rule violation."""

    row_id: str = Field(description="The ID value from the first column of the violating row")
    rules_violated: list[str] = Field(description="List of rule names violated (e.g. ['Rule-A', 'Rule-C'])")
    reason: str = Field(description="Brief explanation citing the specific column values that trigger the violation")


class ReActEval(BaseModel):
    """Full ReAct constraint evaluation with reasoning chain."""

    steps: list[ReActStep] = Field(description="Exactly 5 analysis steps in order")
    violations: list[ViolationResult] = Field(description="All detected violations — empty list if none found")
    confidence: str = Field(description="low, medium, or high — based on rule clarity and data completeness")


class BaselineEval(BaseModel):
    """Simple zero-shot constraint violation detection."""

    violations: list[ViolationResult] = Field(description="All detected violations — empty list if none found")
