"""
LangGraph StateGraph builder and workflow assembly module.
Wires the 5 agents (Claim Extractor, Search Retriever, Fact Verifier, Contradiction Detector, Report Generator) into an executable state graph.
"""

from datetime import datetime, timezone
import logging
import threading
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState, Claim

logger = logging.getLogger(__name__)

_graph_lock = threading.Lock()
_compiled_graph: Optional[Any] = None


async def claim_extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Claim Extractor Node: Decomposes user_query into atomic Claims.
    """
    job_id = state.get("job_id", "")
    query = state.get("user_query", "")
    provider = state.get("model_provider")
    print(f"DEBUG: Entering claim_extractor_node for job_id='{job_id}'")

    try:
        from app.agents.claim_extractor import extract_claims
        claims = await extract_claims(query=query, provider=provider)
        print(f"DEBUG: Exiting claim_extractor_node with {len(claims)} claims for job_id='{job_id}'")
        return {"claims": claims}
    except Exception as e:
        logger.error(f"[Node: claim_extractor] Failed for job_id='{job_id}': {e}", exc_info=True)
        print(f"DEBUG: Error in claim_extractor_node: {e}")
        errors = state.get("errors", []) + [f"Claim extraction failed: {str(e)}"]
        return {"claims": [], "errors": errors}


async def search_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Search & Retrieval Node: Searches Tavily, generates embeddings, loads FAISS index.
    """
    job_id = state.get("job_id", "")
    claims = state.get("claims", [])
    print(f"DEBUG: Entering search_retriever_node for job_id='{job_id}' with {len(claims)} claims")

    try:
        from app.agents.search_retriever import retrieve_and_index_evidence
        sources, faiss_index_id = await retrieve_and_index_evidence(job_id=job_id, claims=claims)
        print(f"DEBUG: Exiting search_retriever_node with {len(sources)} sources for job_id='{job_id}'")
        return {"sources": sources, "faiss_index_id": faiss_index_id}
    except Exception as e:
        logger.error(f"[Node: search_retriever] Failed for job_id='{job_id}': {e}", exc_info=True)
        print(f"DEBUG: Error in search_retriever_node: {e}")
        errors = state.get("errors", []) + [f"Evidence retrieval failed: {str(e)}"]
        return {"sources": [], "faiss_index_id": job_id, "errors": errors}


async def fact_verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Fact Verification Node: Cross-checks claims against retrieved sources using single BATCH call.
    """
    job_id = state.get("job_id", "")
    claims = state.get("claims", [])
    sources = state.get("sources", [])
    provider = state.get("model_provider")
    print(f"DEBUG: Entering fact_verifier_node for job_id='{job_id}' with {len(claims)} claims and {len(sources)} sources")

    errors = list(state.get("errors", []))

    try:
        from app.agents.fact_verifier import verify_claims_batch
        verified_claims = await verify_claims_batch(claims=claims, sources=sources, job_id=job_id, provider=provider)
        print(f"DEBUG: Exiting fact_verifier_node with {len(verified_claims)} verified claims for job_id='{job_id}'")
        return {"claims": verified_claims, "errors": errors}
    except Exception as e:
        logger.error(f"[Node: fact_verifier] Failed for job_id='{job_id}': {e}", exc_info=True)
        print(f"DEBUG: Error in fact_verifier_node: {e}")
        errors.append(f"Fact verification failed: {str(e)}")
        return {"claims": claims, "errors": errors}


async def contradiction_detector_node(state: AgentState) -> Dict[str, Any]:
    """
    Contradiction Detector Node: Cross-examines sources for discrepancies.
    """
    job_id = state.get("job_id", "")
    claims = state.get("claims", [])
    sources = state.get("sources", [])
    provider = state.get("model_provider")
    print(f"DEBUG: Entering contradiction_detector_node for job_id='{job_id}'")

    try:
        from app.agents.contradiction_detector import detect_contradictions
        contradictions = await detect_contradictions(claims=claims, sources=sources, provider=provider)
        print(f"DEBUG: Exiting contradiction_detector_node with {len(contradictions)} contradictions for job_id='{job_id}'")
        return {"contradictions": contradictions}
    except Exception as e:
        logger.error(f"[Node: contradiction_detector] Failed for job_id='{job_id}': {e}", exc_info=True)
        print(f"DEBUG: Error in contradiction_detector_node: {e}")
        errors = state.get("errors", []) + [f"Contradiction detection failed: {str(e)}"]
        return {"contradictions": [], "errors": errors}


async def report_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Report Generator Node: Synthesizes final Markdown report.
    """
    job_id = state.get("job_id", "")
    query = state.get("user_query", "")
    claims = state.get("claims", [])
    sources = state.get("sources", [])
    contradictions = state.get("contradictions", [])
    provider = state.get("model_provider")
    print(f"DEBUG: Entering report_generator_node for job_id='{job_id}'")

    try:
        from app.agents.report_generator import generate_report
        report_markdown = await generate_report(
            user_query=query,
            claims=claims,
            sources=sources,
            contradictions=contradictions,
            provider=provider,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        print(f"DEBUG: Exiting report_generator_node for job_id='{job_id}'")
        return {"final_report": report_markdown, "completed_at": completed_at}
    except Exception as e:
        logger.error(f"[Node: report_generator] Failed for job_id='{job_id}': {e}", exc_info=True)
        print(f"DEBUG: Error in report_generator_node: {e}")
        completed_at = datetime.now(timezone.utc).isoformat()
        errors = state.get("errors", []) + [f"Report generation failed: {str(e)}"]
        return {"final_report": f"# Research Report: {query}\n\nReport generation failed: {str(e)}", "completed_at": completed_at, "errors": errors}


def build_graph() -> StateGraph:
    """
    Constructs and wires the multi-agent StateGraph pipeline.

    Execution Flow:
    START -> claim_extractor -> search_retriever -> fact_verifier -> contradiction_detector -> report_generator -> END
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("claim_extractor", claim_extractor_node)
    builder.add_node("search_retriever", search_retriever_node)
    builder.add_node("fact_verifier", fact_verifier_node)
    builder.add_node("contradiction_detector", contradiction_detector_node)
    builder.add_node("report_generator", report_generator_node)

    # Wire sequential edges
    builder.add_edge(START, "claim_extractor")
    builder.add_edge("claim_extractor", "search_retriever")
    builder.add_edge("search_retriever", "fact_verifier")
    builder.add_edge("fact_verifier", "contradiction_detector")
    builder.add_edge("contradiction_detector", "report_generator")
    builder.add_edge("report_generator", END)

    return builder


def get_graph():
    """
    Returns the compiled LangGraph execution graph instance with thread-safe lazy compilation.
    """
    global _compiled_graph
    if _compiled_graph is None:
        with _graph_lock:
            if _compiled_graph is None:
                logger.info("Compiling LangGraph workflow lazily on first request...")
                print("DEBUG: Compiling LangGraph workflow lazily...")
                workflow = build_graph()
                _compiled_graph = workflow.compile()
                print("DEBUG: LangGraph workflow compiled successfully")
    return _compiled_graph
