import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm


def evidence_block(items: list[dict], limit: int = 1200) -> str:
    return "\n\n".join(
        f"[{e['id']}] ({e['type']}) {e['title']} — {e['ref']}\n{e['content'][:limit]}" for e in items
    )


def all_evidence(state: dict) -> list[dict]:
    return (
        state.get("web_evidence", []) + state.get("rag_evidence", []) + state.get("data_evidence", [])
    )


async def summarize_task(question: str, evidence: list[dict]) -> str:
    if not evidence:
        return "No relevant evidence was found."
    msgs = [
        SystemMessage(
            content=(
                "You extract findings from evidence. Answer the question using ONLY the evidence. "
                "Cite evidence IDs in square brackets, e.g. [W1]. If the evidence is insufficient, say so. "
                "Be concise (max 150 words)."
            )
        ),
        HumanMessage(content=f"Question: {question}\n\nEvidence:\n{evidence_block(evidence)}"),
    ]
    return (await get_llm(0.1).ainvoke(msgs)).content


def render_sources(report_text: str, evidence: list[dict]) -> str:
    """Deterministic Sources section built from real evidence — never LLM-written."""
    cited = set(re.findall(r"\[([WDF]\d+)\]", report_text))
    used = [e for e in evidence if e["id"] in cited]
    if not used:
        return ""
    lines = ["", "", "---", "", "## Sources", ""]
    for e in used:
        if e["type"] == "web":
            lines.append(f"- **[{e['id']}]** [{e['title'] or e['ref']}]({e['ref']})")
        elif e["type"] == "document":
            lines.append(f"- **[{e['id']}]** {e['ref']} *(uploaded document)*")
        else:
            lines.append(f"- **[{e['id']}]** {e['title']} *(uploaded data)*")
    return "\n".join(lines) + "\n"