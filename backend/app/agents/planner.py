import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.events import emit
from app.agents.state import ResearchState
from app.llm import structured
from app.schemas import Plan, PlanTask
from app.services.documents import list_documents

logger = logging.getLogger(__name__)

SYSTEM = """You are the Planner Agent of a multi-agent research system.
Break the user's query into 1-6 focused, self-contained tasks and assign each to exactly one worker:
- "web":  needs current or external public information (news, facts, market data, general knowledge lookups)
- "rag":  should be answered from the user's uploaded text documents (ONLY if such documents exist)
- "data": needs numeric analysis of the user's uploaded CSV/Excel tables (ONLY if such tables exist)
Each task's question must make sense on its own. Prefer fewer, sharper tasks. Do not assign work to
agents whose resources are unavailable."""


async def planner_node(state: ResearchState):
    emit(agent="planner", status="running", message="Breaking the query into tasks…")

    docs = await list_documents(state["session_id"])
    text_docs = [d["filename"] for d in docs if d["kind"] == "text"]
    tables = [t for d in docs if d["kind"] == "data" for t in d["summary"]["tables"]]

    resources = (
        f"Uploaded text documents: {text_docs or 'NONE'}\n"
        f"Uploaded tables: "
        + (
            "; ".join(f"{t['name']} (columns: {', '.join(t['columns'][:15])})" for t in tables)
            if tables
            else "NONE"
        )
    )

    try:
        plan: Plan = await structured(Plan).ainvoke(
            [SystemMessage(content=SYSTEM), HumanMessage(content=f"{resources}\n\nUser query: {state['query']}")]
        )
        tasks = [t for t in plan.tasks if not (t.agent == "rag" and not text_docs) and not (t.agent == "data" and not tables)]
        objective = plan.objective
    except Exception:
        logger.exception("Planner failed; using fallback plan")
        tasks, objective = [], state["query"]

    if not tasks:
        tasks = [PlanTask(agent="web", question=state["query"], rationale="Fallback: general web research")]

    for i, t in enumerate(tasks, 1):
        t.id = f"t{i}"

    plan_dict = {"objective": objective, "tasks": [t.model_dump() for t in tasks]}
    emit(
        agent="planner",
        status="done",
        message=f"Created {len(tasks)} task(s)",
        data={"objective": objective, "tasks": plan_dict["tasks"]},
    )
    return {"plan": plan_dict}