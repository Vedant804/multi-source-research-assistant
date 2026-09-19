from typing import Literal

from pydantic import BaseModel, Field


# ---------- API ----------
class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)


class DocumentOut(BaseModel):
    id: str
    filename: str
    kind: Literal["text", "data"]
    created_at: str
    summary: dict


# ---------- Planner ----------
class PlanTask(BaseModel):
    id: str = ""
    agent: Literal["web", "rag", "data"]
    question: str = Field(description="A self-contained question/instruction for the worker agent")
    rationale: str = ""


class Plan(BaseModel):
    objective: str = Field(description="One-sentence restatement of what the user wants")
    tasks: list[PlanTask] = Field(min_length=1, max_length=6)


# ---------- Data agent ----------
class DataOp(BaseModel):
    table: str = Field(description="Exact table name from the provided schema")
    op: Literal["describe", "value_counts", "groupby_agg", "top_n", "correlation", "time_trend"]
    column: str | None = Field(default=None, description="Target column")
    group_by: str | None = Field(default=None, description="Grouping column (groupby_agg)")
    date_column: str | None = Field(default=None, description="Date column (time_trend)")
    agg: Literal["sum", "mean", "count", "min", "max", "median"] = "sum"
    freq: Literal["D", "W", "M", "Q", "Y"] = "M"
    n: int = 10
    ascending: bool = False


class DataPlan(BaseModel):
    operations: list[DataOp] = Field(default_factory=list, max_length=5)


# ---------- Fact checker ----------
class ClaimCheck(BaseModel):
    claim: str
    verdict: Literal["supported", "partial", "unsupported"]
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""


class FactCheckResult(BaseModel):
    claims: list[ClaimCheck]
    corrected_draft: str = Field(
        description="The draft with unsupported claims removed, partial claims qualified, valid citations kept"
    )
    confidence: float = Field(ge=0.0, le=1.0)