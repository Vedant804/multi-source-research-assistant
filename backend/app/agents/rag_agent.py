import asyncio

from app.agents.common import summarize_task
from app.agents.events import emit, safe_agent
from app.agents.state import ResearchState
from app.llm import get_embeddings
from app.services.vectorstore import search_chunks

NAME = "rag_agent"


@safe_agent(NAME, {"rag_evidence": [], "rag_findings": []})
async def rag_agent_node(state: ResearchState):
    tasks = [t for t in state["plan"]["tasks"] if t["agent"] == "rag"]
    if not tasks:
        emit(agent=NAME, status="skipped", message="No document tasks planned")
        return {"rag_evidence": [], "rag_findings": []}

    emit(agent=NAME, status="running", message=f"Searching uploaded documents ({len(tasks)} queries)…")
    vectors = await get_embeddings().aembed_documents([t["question"] for t in tasks])
    hits_per_task = await asyncio.gather(*(search_chunks(state["session_id"], v, k=5) for v in vectors))

    evidence, seen, n = [], set(), 1
    for t, hits in zip(tasks, hits_per_task):
        for h in hits:
            if h["id"] in seen:
                continue
            seen.add(h["id"])
            evidence.append(
                {"id": f"D{n}", "type": "document", "title": h["filename"],
                 "ref": f"{h['filename']} · chunk {h['chunk_index']}", "content": h["content"],
                 "score": round(float(h["score"]), 3), "task_id": t["id"]}
            )
            n += 1

    summaries = await asyncio.gather(
        *(summarize_task(t["question"], [e for e in evidence if e["task_id"] == t["id"]]) for t in tasks)
    )
    findings = [
        {"task_id": t["id"], "question": t["question"], "summary": s} for t, s in zip(tasks, summaries)
    ]

    emit(
        agent=NAME,
        status="done",
        message=f"Retrieved {len(evidence)} passages",
        data={"sources": [{"id": e["id"], "title": e["title"], "ref": e["ref"]} for e in evidence]},
    )
    return {"rag_evidence": evidence, "rag_findings": findings}