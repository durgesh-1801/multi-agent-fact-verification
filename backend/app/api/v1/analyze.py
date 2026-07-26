"""
Research Analysis Endpoint (POST /api/v1/analyze).
Executes the Autonomous Multi-Agent workflow for fact verification and research synthesis.
"""

from datetime import datetime, timezone
import logging
import uuid
from fastapi import APIRouter, HTTPException, status

from app.graph.state import AgentState
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, AnalysisSummary
from app.services.mongo_service import mongo_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Multi-Agent Research & Fact Verification Analysis",
)
async def analyze_research_query(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Triggers the 5-agent LangGraph workflow to extract claims, retrieve web evidence,
    cross-verify facts, detect source contradictions, and synthesize a Markdown research paper.
    """
    print(f"DEBUG: Entering POST /api/v1/analyze endpoint with query='{payload.query[:30]}...'")
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    created_at = datetime.now(timezone.utc).isoformat()
    clean_query = payload.query.strip()

    logger.info(f"Starting research analysis job_id='{job_id}' (provider={payload.model_provider}) for query: '{clean_query[:60]}...'")

    initial_state: AgentState = {
        "job_id": job_id,
        "user_query": clean_query,
        "model_provider": payload.model_provider,
        "created_at": created_at,
        "completed_at": None,
        "claims": [],
        "sources": [],
        "faiss_index_id": None,
        "contradictions": [],
        "final_report": None,
        "errors": [],
    }

    print(f"DEBUG: Created initial_state for job_id='{job_id}'")

    try:
        from app.graph.builder import get_graph
        graph = get_graph()
        print(f"DEBUG: Invoking graph.ainvoke() for job_id='{job_id}'...")
        final_state: AgentState = await graph.ainvoke(initial_state)
        print(f"DEBUG: Completed graph.ainvoke() for job_id='{job_id}'")

        completed_at = final_state.get("completed_at") or datetime.now(timezone.utc).isoformat()
        claims = final_state.get("claims", [])
        sources = final_state.get("sources", [])
        contradictions = final_state.get("contradictions", [])
        errors = final_state.get("errors", [])
        final_report = final_state.get("final_report", "")

        supported_count = sum(1 for c in claims if c.verdict == "SUPPORTED")
        refuted_count = sum(1 for c in claims if c.verdict == "REFUTED")
        inconclusive_count = sum(1 for c in claims if c.verdict not in ("SUPPORTED", "REFUTED"))

        summary = AnalysisSummary(
            total_claims=len(claims),
            supported_claims=supported_count,
            refuted_claims=refuted_count,
            inconclusive_claims=inconclusive_count,
            total_sources=len(sources),
            contradictions_detected=len(contradictions),
        )

        job_status = "completed" if not errors or final_report else "failed"

        mongo_doc = {
            "job_id": job_id,
            "user_query": clean_query,
            "model_provider": payload.model_provider,
            "status": job_status,
            "created_at": created_at,
            "completed_at": completed_at,
            "summary": summary.model_dump(),
            "claims": [c.model_dump() for c in claims],
            "sources": [s.model_dump() for s in sources],
            "contradictions": [ctr.model_dump() for ctr in contradictions],
            "report_markdown": final_report,
            "errors": errors,
        }

        try:
            print(f"DEBUG: Persisting analysis document to MongoDB for job_id='{job_id}'...")
            await mongo_service.create_analysis(mongo_doc)
            print(f"DEBUG: Successfully persisted document to MongoDB for job_id='{job_id}'")
        except Exception as db_err:
            logger.error(f"Failed to persist analysis doc to MongoDB for job_id='{job_id}': {db_err}")

        logger.info(
            f"Successfully finished analysis job_id='{job_id}' (status={job_status}, "
            f"claims={summary.total_claims}, sources={summary.total_sources})."
        )

        print(f"DEBUG: Returning AnalyzeResponse for job_id='{job_id}'")
        return AnalyzeResponse(
            job_id=job_id,
            status=job_status,
            query=clean_query,
            created_at=created_at,
            completed_at=completed_at,
            summary=summary,
            claims=claims,
            sources=sources,
            contradictions=contradictions,
            report_markdown=final_report,
            errors=errors,
        )

    except Exception as e:
        logger.error(f"Error executing analysis pipeline for job_id='{job_id}': {e}", exc_info=True)
        print(f"DEBUG: Exception in POST /api/v1/analyze: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research analysis workflow failed: {str(e)}",
        ) from e
