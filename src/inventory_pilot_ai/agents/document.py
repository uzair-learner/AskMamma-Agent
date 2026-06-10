"""Document agent definition."""

from __future__ import annotations

from inventory_pilot_ai.agents.base import AgentDefinition, AgentTool
from inventory_pilot_ai.rag.retriever import document_search


def build_document_agent() -> AgentDefinition:
    return AgentDefinition(
        name="DocumentAgent",
        system_prompt=(
            "You are the Inventory Pilot AI document retrieval specialist. Answer from retrieved evidence only and "
            "cite the originating file names in concise language."
        ),
        responsibilities=[
            "Search uploaded PDFs, TXTs, and DOCX-compatible text extractions.",
            "Surface the most relevant chunks from the vector store.",
            "Keep answers grounded in retrieval evidence instead of invented facts.",
        ],
        routing_rules=[
            "Use for policy, contract, file upload, PDF, TXT, DOCX, and knowledge-base questions."
        ],
        tools=[AgentTool("DocumentSearchTool", "Search the vector store.", document_search, {}, {"type": "object"})],
        logging_rules=[
            "Record retrieved file names and chunk identifiers.",
            "Store the retriever backend used for the request.",
        ],
        trace_tags=["document", "rag"],
        trace_metadata={"team": "knowledge", "stage": "specialist"},
    )
