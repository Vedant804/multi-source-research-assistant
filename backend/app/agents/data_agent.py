import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.common import summarize_task
from app.agents.events import emit, safe_agent
from app.agents.state import ResearchState
from app.llm import structured
from app.schemas import DataPlan
from app.services.data_ops import execute_op, read_table
from app.services.documents import get_data_files

logger = logging.getLogger(__name__)
NAME = "data_agent"

SYSTEM = """You are the Data Agent. Given table schemas and a question, choose up to 4 analysis operations.
You cannot write code. Use ONLY these operations: describe, value_counts, groupby_agg, top_n, correlation, time_trend.
Use exact table and column names from the schema. Use sum/mean only on numeric columns."""


def _schema_text(tables: list[dict]) -> str:
    out = []
    for t in tables:
        cols = ", ".join(f"{c['name']} ({c['dtype']})" for c in t["columns"])
        out.append(f"TABLE {t['name']} — {t['rows']} rows\n  columns: {cols}\n  sample: {t['sample'][:2]}")
    return "\n\n".join(out)


@safe_agent(NAME, {"data_evidence": [], "data_findings": []})
async def data_agent_node(state: ResearchState):
    tasks = [t for t in state["plan"]["tasks"] if t["agent"] == "data"]
    if not tasks:
        emit(agent=NAME, status="skipped", message="No data-analysis tasks planned")
        return {"data_evidence": [], "data_findings": []}

    files = await get_data_files(state["session_id"])
    table_index = {t["name"]: (f["path"], t) for f in files for t in f["tables"]}
    if not table_index:
        emit(agent=NAME, status="skipped", message="No data files uploaded")
        return {"data_evidence": [], "data_findings": []}

    emit(agent=NAME, status="running", message=f"Planning analysis for {len(tasks)} task(s)…")
    schema = _schema_text([t for _, t in table_index.values()])
    planner = structured(DataPlan)

    plans = await asyncio.gather(
        *(
            planner.ainvoke(
                [SystemMessage(content=SYSTEM), HumanMessage(content=f"{schema}\n\nQuestion: {t['question']}")]
            )
            for t in tasks
        )
    )

    evidence, n, cache = [], 1, {}
    for t, plan in zip(tasks, plans):
        for op in plan.operations:
            if op.table not in table_index:
                continue
            path = table_index[op.table][0]
            try:
                if op.table not in cache:
                    cache[op.table] = await asyncio.to_thread(read_table, path, op.table)
                result = await asyncio.to_thread(execute_op, cache[op.table], op)
            except Exception as exc:
                logger.warning("Data op failed (%s): %s", op.op, exc)
                continue
            label = op.op + (f" on {op.column}" if op.column else "") + (f" by {op.group_by}" if op.group_by else "")
            evidence.append(
                {"id": f"F{n}", "type": "data", "title": f"{label} · {op.table}", "ref": op.table,
                 "content": result, "task_id": t["id"]}
            )
            n += 1

    emit(agent=NAME, status="running", message=f"Interpreting {len(evidence)} result tables…")
    summaries = await asyncio.gather(
        *(summarize_task(t["question"], [e for e in evidence if e["task_id"] == t["id"]]) for t in tasks)
    )
    findings = [
        {"task_id": t["id"], "question": t["question"], "summary": s} for t, s in zip(tasks, summaries)
    ]

    emit(
        agent=NAME,
        status="done",
        message=f"Ran {len(evidence)} analyses",
        data={"sources": [{"id": e["id"], "title": e["title"], "ref": e["ref"]} for e in evidence]},
    )
    return {"data_evidence": evidence, "data_findings": findings}