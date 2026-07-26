"""
Pydantic schemas for the research analysis API endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from app.graph.state import Claim, Source, Contradiction


class AnalyzeRequest(BaseModel):
    """Request payload schema for research fact verification analysis."""
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The research query, statement, or topic to verify",
        examples=["Global renewable electricity share exceeded 30% in 2024."],
    )
    model_provider: Optional[str] = Field(
        default="groq",
        description="Primary LLM provider override ('groq', 'openai', 'gemini', or 'claude')",
    )

    @field_validator("model_provider")
    @classmethod
    def validate_model_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_clean = v.strip().lower()
            if v_clean not in ("groq", "openai", "gemini", "claude"):
                raise ValueError("model_provider must be one of: 'groq', 'openai', 'gemini', 'claude'")
            return v_clean
        return v


class AnalysisSummary(BaseModel):
    """Summary metrics of the research analysis run."""
    total_claims: int = Field(..., description="Total number of extracted claims")
    supported_claims: int = Field(..., description="Number of claims verified as SUPPORTED")
    refuted_claims: int = Field(..., description="Number of claims verified as REFUTED")
    inconclusive_claims: int = Field(..., description="Number of claims marked as INCONCLUSIVE")
    total_sources: int = Field(..., description="Total unique web sources consulted")
    contradictions_detected: int = Field(..., description="Total source-level contradictions identified")


class AnalyzeResponse(BaseModel):
    """Response payload schema for research fact verification analysis."""
    job_id: str = Field(..., description="Unique research job identifier")
    status: str = Field(..., description="Analysis status: 'completed' or 'failed'")
    query: str = Field(..., description="Original user research query")
    created_at: str = Field(..., description="ISO creation timestamp")
    completed_at: Optional[str] = Field(default=None, description="ISO completion timestamp")
    summary: AnalysisSummary = Field(..., description="Run breakdown summary metrics")
    claims: List[Claim] = Field(default_factory=list, description="Extracted & verified claims")
    sources: List[Source] = Field(default_factory=list, description="Consulted web sources")
    contradictions: List[Contradiction] = Field(default_factory=list, description="Detected source contradictions")
    report_markdown: Optional[str] = Field(default=None, description="Final generated Markdown report")
    errors: List[str] = Field(default_factory=list, description="Any execution errors recorded during run")
