from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.common import all_evidence, evidence_block
from app.agents.events import emit, safe_agent
from app.agents.state import ResearchState
from app.llm import structured
from app.schemas import FactCheckResult

SYSTEM = """You are the Fact Checker. Verify the draft against the evidence.
1. Extract the 5-15 most important factual claims (numbers, dates, causal or comparative statements).
2. For each, decide: "supported" (evidence clearly states it), "partial" (evidence is weaker/narrower than claimed),
   or "unsupported" (no evidence, or contradicted). List the evidence IDs you checked.
3. Produce corrected_draft: the same draft with unsupported claims removed, partial claims qualified with
   appropriate hedging, and all valid [ID] citations preserved. Do not add new facts.
4. confidence: 0-1 overall reliability of the corrected draft."""


@safe_agent("fact_checker", {"fact_check": {"claims": [], "confidence": 0.0}})
async def fact_checker_node(state: ResearchState):
    emit(agent="fact_checker", status="running", message="Verifying claims against the evidence…")

    evidence = all_evidence(state)
    draft = state["draft"]

    if not evidence:
        emit(agent="fact_checker", status="done", message="No evidence to verify against; draft passed through with caveat",
             data={"claims": [], "confidence": 0.0})
        return {
            "verified_draft": draft,
            "fact_check": {"claims": [], "confidence": 0.0},
        }

    result: FactCheckResult = await structured(FactCheckResult).ainvoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"## Draft\n{draft}\n\n## Evidence\n{evidence_block(evidence[:40], 1500)}"),
        ]
    )
    claims = [c.model_dump() for c in result.claims]
    supported = sum(1 for c in claims if c["verdict"] == "supported")

    emit(
        agent="fact_checker",
        status="done",
        message=f"{supported}/{len(claims)} claims fully supported · confidence {result.confidence:.0%}",
        data={"claims": claims, "confidence": result.confidence},
    )
    return {
        "verified_draft": result.corrected_draft,
        "fact_check": {"claims": claims, "confidence": result.confidence},
    }


# Failure fallback must still provide verified_draft -> handled in report node via state.get