from langgraph.graph import END, START, StateGraph

from app.agents.data_agent import data_agent_node
from app.agents.fact_checker import fact_checker_node
from app.agents.planner import planner_node
from app.agents.rag_agent import rag_agent_node
from app.agents.report import report_generator_node
from app.agents.researcher import research_agent_node
from app.agents.state import ResearchState
from app.agents.web_agent import web_agent_node


def build_graph():
    g = StateGraph(ResearchState)

    # NOTE: node names must not collide with state keys (e.g. "report", "plan").
    g.add_node("planner", planner_node)
    g.add_node("web_agent", web_agent_node)
    g.add_node("rag_agent", rag_agent_node)
    g.add_node("data_agent", data_agent_node)
    g.add_node("research_agent", research_agent_node)
    g.add_node("fact_checker", fact_checker_node)
    g.add_node("report_generator", report_generator_node)

    g.add_edge(START, "planner")

    # Parallel fan-out
    for worker in ("web_agent", "rag_agent", "data_agent"):
        g.add_edge("planner", worker)

    # Fan-in: research_agent runs only after ALL three workers finish
    g.add_edge(["web_agent", "rag_agent", "data_agent"], "research_agent")

    g.add_edge("research_agent", "fact_checker")
    g.add_edge("fact_checker", "report_generator")
    g.add_edge("report_generator", END)

    return g.compile()


graph = build_graph()