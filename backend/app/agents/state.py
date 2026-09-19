from typing import TypedDict


class ResearchState(TypedDict, total=False):
    # inputs
    run_id: str
    session_id: str
    query: str

    # planner
    plan: dict

    # workers (each worker writes only its own keys -> safe parallel merge)
    web_evidence: list[dict]
    web_findings: list[dict]
    rag_evidence: list[dict]
    rag_findings: list[dict]
    data_evidence: list[dict]
    data_findings: list[dict]

    # downstream
    draft: str
    fact_check: dict
    verified_draft: str
    report: str