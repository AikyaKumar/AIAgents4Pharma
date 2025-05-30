"""
LangGraph PDF Retrieval-Augmented Generation (RAG) Tool

This tool answers user questions by retrieving and ranking relevant text chunks from PDFs
and invoking an LLM to generate a concise, source-attributed response. It supports
single or multiple PDF sources—such as Zotero libraries, arXiv papers, or direct uploads.

Workflow:
  1. (Optional) Load PDFs from diverse sources into a FAISS vector store of embeddings.
  2. Rerank candidate papers using NVIDIA NIM semantic re-ranker.
  3. Retrieve top-K diverse text chunks via Maximal Marginal Relevance (MMR).
  4. Build a context-rich prompt combining retrieved chunks and the user question.
  5. Invoke the LLM to craft a clear answer with source citations.
  6. Return the answer in a ToolMessage for LangGraph to dispatch.
"""

import logging
import os
import time
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from .utils.generate_answer import load_hydra_config
from .utils.retrieve_chunks import retrieve_relevant_chunks
from .utils.tool_helper import QAToolHelper

# Helper for managing state, vectorstore, reranking, and formatting
helper = QAToolHelper()
# Load configuration and start logging
config = load_hydra_config()

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class QuestionAndAnswerInput(BaseModel):
    """
    Pydantic schema for the PDF Q&A tool inputs.

    Fields:
      question: User's free-text query to answer based on PDF content.
      tool_call_id: LangGraph-injected call identifier for tracking.
      state: Shared agent state dict containing:
        - article_data: metadata mapping of paper IDs to info (e.g., 'pdf_url', title).
        - text_embedding_model: embedding model instance for chunk indexing.
        - llm_model: chat/LLM instance for answer generation.
        - vector_store: optional pre-built Vectorstore for retrieval.
    """

    question: str = Field(
        description="User question for generating a PDF-based answer."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    LangGraph tool for Retrieval-Augmented Generation over PDFs.

    Given a user question, this tool applies the following pipeline:
      1. Validates that embedding and LLM models, plus article metadata, are in state.
      2. Initializes or reuses a FAISS-based Vectorstore for PDF embeddings.
      3. Loads one or more PDFs (from Zotero, arXiv, uploads) as text chunks into the store.
      4. Uses NVIDIA NIM semantic re-ranker to select top candidate papers.
      5. Retrieves the most relevant and diverse text chunks via Maximal Marginal Relevance.
      6. Constructs an LLM prompt combining contextual chunks and the query.
      7. Invokes the LLM to generate an answer, appending source attributions.
      8. Returns a LangGraph Command with a ToolMessage containing the answer.

    Args:
      question (str): The free-text question to answer.
      state (dict): Injected agent state; must include:
        - article_data: mapping paper IDs → metadata (pdf_url, title, etc.)
        - text_embedding_model: embedding model instance.
        - llm_model: chat/LLM instance.
      tool_call_id (str): Internal identifier for this tool invocation.

    Returns:
      Command[Any]: updates conversation state with a ToolMessage(answer).

    Raises:
      ValueError: when required models or metadata are missing in state.
      RuntimeError: when no relevant chunks can be retrieved for the query.
    """
    call_id = f"qa_call_{time.time()}"
    logger.info(
        "Starting PDF Question and Answer tool call %s for question: %s",
        call_id,
        question,
    )

    # Get required models from state
    text_embedding_model = state.get("text_embedding_model")
    if not text_embedding_model:
        error_msg = "No text embedding model found in state."
        logger.error("%s: %s", call_id, error_msg)
        raise ValueError(error_msg)

    llm_model = state.get("llm_model")
    if not llm_model:
        error_msg = "No LLM model found in state."
        logger.error("%s: %s", call_id, error_msg)
        raise ValueError(error_msg)

    # Get article data from state
    article_data = state.get("article_data", {})
    if not article_data:
        error_msg = "No article_data found in state."
        logger.error("%s: %s", call_id, error_msg)
        raise ValueError(error_msg)

    # Use shared pre-built Vectorstore if provided, else create a new one
    if prebuilt_vector_store is not None:
        vector_store = prebuilt_vector_store
        logger.info("Using shared pre-built vector store from the memory")
    else:
        vector_store = Vectorstore(embedding_model=text_embedding_model)
        logger.info("Initialized new vector store (no pre-built store found)")

    # Check if there are papers from different sources
    has_uploaded_papers = any(
        paper.get("source") == "upload"
        for paper in article_data.values()
        if isinstance(paper, dict)
    )

    has_zotero_papers = any(
        paper.get("source") == "zotero"
        for paper in article_data.values()
        if isinstance(paper, dict)
    )

    has_arxiv_papers = any(
        paper.get("source") == "arxiv"
        for paper in article_data.values()
        if isinstance(paper, dict)
    )

    # Choose papers to use
    selected_paper_ids = []

    if paper_ids:
        # Use explicitly specified papers
        selected_paper_ids = [pid for pid in paper_ids if pid in article_data]
        logger.info(
            "%s: Using explicitly specified papers: %s", call_id, selected_paper_ids
        )

        if not selected_paper_ids:
            logger.warning(
                "%s: None of the provided paper_ids %s were found", call_id, paper_ids
            )

    elif use_all_papers or has_uploaded_papers or has_zotero_papers or has_arxiv_papers:
        # Use all available papers if explicitly requested or if we have papers from any source
        selected_paper_ids = list(article_data.keys())
        logger.info(
            "%s: Using all %d available papers", call_id, len(selected_paper_ids)
        )

    else:
        # Use semantic ranking to find relevant papers
        # First ensure papers are loaded
        for paper_id, paper in article_data.items():
            pdf_url = paper.get("pdf_url")
            if pdf_url and paper_id not in vector_store.loaded_papers:
                try:
                    vector_store.add_paper(paper_id, pdf_url, paper)
                except (IOError, ValueError) as e:
                    logger.error("Error loading paper %s: %s", paper_id, e)
                    raise

        # Now rank papers
        ranked_papers = vector_store.rank_papers_by_query(
            question, top_k=config.top_k_papers
        )
        selected_paper_ids = [paper_id for paper_id, _ in ranked_papers]
        logger.info(
            "%s: Selected papers based on semantic relevance: %s",
            call_id,
            selected_paper_ids,
        )

    if not selected_paper_ids:
        # Fallback to all papers if selection failed
        selected_paper_ids = list(article_data.keys())
        logger.info(
            "%s: Falling back to all %d papers", call_id, len(selected_paper_ids)
        )

    # Load selected papers if needed
    for paper_id in selected_paper_ids:
        if paper_id not in vector_store.loaded_papers:
            pdf_url = article_data[paper_id].get("pdf_url")
            if pdf_url:
                try:
                    vector_store.add_paper(paper_id, pdf_url, article_data[paper_id])
                except (IOError, ValueError) as e:
                    logger.warning(
                        "%s: Error loading paper %s: %s", call_id, paper_id, e
                    )

    # Ensure vector store is built
    if not vector_store.vector_store:
        vector_store.build_vector_store()

    # Retrieve relevant chunks across selected papers
    relevant_chunks = vector_store.retrieve_relevant_chunks(
        query=question, paper_ids=selected_paper_ids, top_k=config.top_k_chunks
    )
    if not relevant_chunks:
        msg = f"No relevant chunks found for question: '{question}'"
        logger.warning("%s: %s", call_id, msg)
        raise RuntimeError(msg)

    # Generate answer and format with sources
    response_text = helper.format_answer(
        question, relevant_chunks, llm_model, article_data
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=response_text,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
