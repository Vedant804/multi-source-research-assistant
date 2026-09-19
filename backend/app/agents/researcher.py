from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.common import all_evidence
from app.agents.events import emit
from app.agents.state import ResearchState
from app.llm import get_llm

SYSTEM = """You are the Research Agent. Synthesize the workers' findings into a coherent, well-organized draft answer.
Rules:
- Use ONLY the provided findings and evidence. Never invent facts, numbers, or evidence IDs.
- Cite every factual claim inline with evidence IDs in square brackets, e.g. [W1] [D2] [F3].
- Explicitly point out conflicts between sources and gaps where evidence is missing.
- Write in markdown. Be thorough but not padded."""


def _findings(title: str, findings: list[dict]) -> str:
    if not findings:
        return ""
    body = "\n".join(f"- Task: {f['question']}\n  Finding: {f['summary']}" for f in findings)
    return f"### {title}\n{body}\n"


async def research_agent_node(state: ResearchState):
    emit(agent="research_agent", status="running", message="Synthesizing findings from all sources…")

    evidence = all_evidence(state)
    index = "\n".join(f"[{e['id']}] {e['title']}: {e['content'][:600]}" for e in evidence) or "(no evidence)"
    findings = (
        _findings("Web findings", state.get("web_findings", []))
        + _findings("Document findings", state.get("rag_findings", []))
        + _findings("Data-analysis findings", state.get("data_findings", []))
    ) or "(no findings)"

    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"User query: {state['query']}\n"
                f"Objective: {state['plan']['objective']}\n\n"
                f"## Worker findings\n{findings}\n\n## Evidence index\n{index}"
            )
        ),
    ]
    draft = (await get_llm(0.2).ainvoke(msgs)).content
    emit(agent="research_agent", status="done", message=f"Draft synthesized from {len(evidence)} evidence items")
    return {"draft": draft}