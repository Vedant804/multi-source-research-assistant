import asyncio

from app.agents.common import summarize_task
from app.agents.events import emit, safe_agent
from app.agents.state import ResearchState
from app.services.websearch import web_search

NAME = "web_agent"


@safe_agent(NAME, {"web_evidence": [], "web_findings": []})
async def web_agent_node(state: ResearchState):
    tasks = [t for t in state["plan"]["tasks"] if t["agent"] == "web"]
    if not tasks:
        emit(agent=NAME, status="skipped", message="No web tasks planned")
        return {"web_evidence": [], "web_findings": []}

    emit(agent=NAME, status="running", message=f"Searching the web ({len(tasks)} queries)…")
    results = await asyncio.gather(*(web_search(t["question"]) for t in tasks))

    evidence, seen, n = [], set(), 1
    for t, items in zip(tasks, results):
        for it in items:
            if not it["url"] or it["url"] in seen:
                continue
            seen.add(it["url"])
            evidence.append(
                {"id": f"W{n}", "type": "web", "title": it["title"], "ref": it["url"],
                 "content": it["content"], "task_id": t["id"]}
            )
            n += 1

    emit(agent=NAME, status="running", message=f"Reading {len(evidence)} sources…")
    summaries = await asyncio.gather(
        *(summarize_task(t["question"], [e for e in evidence if e["task_id"] == t["id"]]) for t in tasks)
    )
    findings = [
        {"task_id": t["id"], "question": t["question"], "summary": s} for t, s in zip(tasks, summaries)
    ]

    emit(
        agent=NAME,
        status="done",
        message=f"Collected {len(evidence)} web sources",
        data={"sources": [{"id": e["id"], "title": e["title"], "ref": e["ref"]} for e in evidence]},
    )
    return {"web_evidence": evidence, "web_findings": findings}