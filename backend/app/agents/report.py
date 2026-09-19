from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.common import all_evidence, render_sources
from app.agents.events import emit
from app.agents.state import ResearchState
from app.llm import get_llm

SYSTEM = """You are the Report Generator. Turn the verified draft into a polished markdown research report.
Structure:
# <Descriptive title>
## Executive Summary  (3-5 sentences)
## Key Findings       (bullets, each with citations)
## Detailed Analysis  (sections with ### subheadings; include tables where useful, esp. for numeric data)
## Caveats & Limitations (use the fact-check notes: mention partially supported or removed claims and evidence gaps)
Rules: keep the [W#]/[D#]/[F#] citations exactly as given; never invent citations, numbers, or sources;
do NOT write a Sources/References section (it is appended automatically)."""


async def report_generator_node(state: ResearchState):
    emit(agent="report_generator", status="running", message="Writing the final report…")

    evidence = all_evidence(state)
    fc = state.get("fact_check", {}) or {}
    weak = [c for c in fc.get("claims", []) if c["verdict"] != "supported"]
    weak_txt = "\n".join(f"- ({c['verdict']}) {c['claim']} — {c.get('note','')}" for c in weak) or "None"
    draft = state.get("verified_draft") or state.get("draft", "")

    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"User query: {state['query']}\n\n"
                f"## Verified draft\n{draft}\n\n"
                f"## Fact-check flags (partial/unsupported)\n{weak_txt}\n"
                f"Overall confidence: {fc.get('confidence', 'n/a')}"
            )
        ),
    ]

    text = ""
    async for chunk in get_llm(0.3, streaming=True).astream(msgs):
        piece = chunk.content
        if piece:
            text += piece
            emit(type="token", text=piece)

    sources = render_sources(text, evidence)
    if sources:
        text += sources
        emit(type="token", text=sources)

    emit(type="final", report=text)
    emit(agent="report_generator", status="done", message="Report complete")
    return {"report": text}