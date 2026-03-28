#!/usr/bin/env python3
"""
Smart Query Assistant — Production AI Pipeline
===============================================
Multi-source retrieval + decision-based answering with persistent memory.

Architecture
------------
  User Input (user_id, thread_id, query)
      │
      ▼
  [thread_check]          ← detect new vs existing thread
      │
      ▼
  [memory]                ← retrieve relevant past exchanges from FAISS
      │
      ├────────────────────────────────────┐
      ▼                                    ▼
  [youtube_retriever]              [wiki_retriever]     ← parallel
  (FAISS vector search)            (Wikipedia via MCP)
      │                                    │
      └──────────────┬─────────────────────┘
                     ▼
                 [merge]            ← deduplicate + combine results
                     │
                     ▼
                 [decision]         ← LLM decides best source (strict JSON)
                     │
          ┌──────────┼──────────┬──────────┐
          ▼          ▼          ▼          ▼
    [yt_answer] [wiki_answer] [hybrid] [fallback]   ← conditional routing
          │          │          │          │
          └──────────┴──────────┴──────────┘
                     │
                     ▼
                  [save]            ← persist Q&A to memory vector store
                     │
                     ▼
                   END

Stack: LangGraph · LangChain · Groq (llama-3.3-70b) · MCP · FAISS
"""

# ─────────────────────────────────────────────────────────────────────────────
# MCP SERVER MODE
# This file doubles as the Wikipedia MCP server.
# When launched with --mcp-server it serves tools over stdio and exits.
# The main pipeline launches it as a subprocess via MultiServerMCPClient.
# ─────────────────────────────────────────────────────────────────────────────

import sys

if "--mcp-server" in sys.argv:
    import logging as _logging
    _logging.basicConfig(level=_logging.ERROR)   # silence MCP server INFO logs

    from mcp.server.fastmcp import FastMCP
    import wikipedia as _wiki

    _mcp_app = FastMCP("wikipedia-server")

    @_mcp_app.tool()
    def search_wikipedia(query: str) -> str:
        """
        Search Wikipedia and return concise summaries for the given query.

        Args:
            query: The topic or question to search for on Wikipedia.

        Returns:
            A formatted string with Wikipedia article summaries.
        """
        try:
            titles = _wiki.search(query, results=3)
            if not titles:
                return f"No Wikipedia results found for: {query}"

            summaries = []
            for title in titles[:2]:
                try:
                    summary = _wiki.summary(title, sentences=5, auto_suggest=False)
                    summaries.append(f"[Wikipedia: {title}]\n{summary}")
                except _wiki.exceptions.DisambiguationError as e:
                    # Use the first suggested page on disambiguation
                    try:
                        summary = _wiki.summary(e.options[0], sentences=5, auto_suggest=False)
                        summaries.append(f"[Wikipedia: {e.options[0]}]\n{summary}")
                    except Exception:
                        continue
                except Exception:
                    continue

            return "\n\n".join(summaries) if summaries else f"Could not retrieve content for: {query}"

        except Exception as exc:
            return f"Wikipedia search error: {exc}"

    _mcp_app.run(transport="stdio")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS  (only executed in pipeline mode, not in MCP server mode)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import asyncio
import logging
import warnings
from pathlib import Path
from typing import TypedDict, List

# ── Suppress noisy third-party logs (safe to remove for debugging) ────────────
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"   # Fast + highly capable Groq model
EMBED_MODEL  = "all-MiniLM-L6-v2"          # Used for the in-memory memory store

# YouTube ChromaDB — must match the model used in youtube/02_build_rag.py
YT_EMBED_MODEL  = "sentence-transformers/LaBSE"
YT_CHROMA_DIR   = str(Path(__file__).parent / "youtube" / "chroma_db")
YT_COLLECTION   = "genai_playlist"

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set. Check your .env file.")


# ─────────────────────────────────────────────────────────────────────────────
# STATE  —  single TypedDict that flows through every node
# ─────────────────────────────────────────────────────────────────────────────

class QueryState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query:           str
    user_id:         str
    thread_id:       str

    # ── Thread metadata ───────────────────────────────────────────────────────
    is_new_thread:   bool

    # ── Retrieved context ─────────────────────────────────────────────────────
    memory_context:  str
    youtube_results: List[str]
    wiki_results:    List[str]
    merged_context:  str

    # ── Decision ──────────────────────────────────────────────────────────────
    selected_source: str   # "youtube" | "wiki" | "hybrid" | "fallback"

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer:    str


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STORES
# In production these would be replaced with Redis + a persistent vector DB.
# ─────────────────────────────────────────────────────────────────────────────

_thread_registry:    dict[str, bool]       = {}   # thread_id → has_been_seen
_conversation_store: dict[str, list[dict]] = {}   # thread_id → list of turns


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_embeddings() -> HuggingFaceEmbeddings:
    """Memory store embeddings — lightweight all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_yt_embeddings() -> HuggingFaceEmbeddings:
    """
    YouTube retrieval embeddings — LaBSE (matches youtube/02_build_rag.py).
    Must use the SAME model that was used when building the ChromaDB index.
    """
    return HuggingFaceEmbeddings(model_name=YT_EMBED_MODEL)


def build_youtube_store(yt_embeddings: HuggingFaceEmbeddings):
    """
    Load the persistent ChromaDB YouTube index built by youtube/02_build_rag.py.

    Falls back to an in-memory FAISS stub with a warning message if the
    ChromaDB has not been built yet, so the pipeline still runs end-to-end
    during development.
    """
    if Path(YT_CHROMA_DIR).exists():
        print(f"  [YouTubeStore] Loading ChromaDB from {YT_CHROMA_DIR}")
        return Chroma(
            embedding_function=yt_embeddings,
            persist_directory=YT_CHROMA_DIR,
            collection_name=YT_COLLECTION,
        )

    # ── Fallback: ChromaDB not built yet ──────────────────────────────────────
    print("  [YouTubeStore] ⚠  ChromaDB not found.")
    print("  [YouTubeStore]    Run  python youtube/02_build_rag.py  to build it.")
    print("  [YouTubeStore]    Falling back to in-memory FAISS stub.\n")

    stub_docs = [Document(
        page_content="YouTube RAG not yet built. Run youtube/02_build_rag.py first.",
        metadata={"source": "youtube", "title": "Setup Required"},
    )]
    # Stub uses the same yt_embeddings so type is consistent
    return FAISS.from_documents(stub_docs, yt_embeddings)


def build_memory_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Build an initially empty FAISS memory store.
    Seeded with one placeholder document so FAISS initialises cleanly.
    """
    seed = [Document(
        page_content="memory store initialised",
        metadata={"type": "seed"},
    )]
    return FAISS.from_documents(seed, embeddings)


def build_llm() -> ChatGroq:
    """Initialise the Groq LLM client."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.1,   # Low temperature for consistent, factual answers
    )


# ─────────────────────────────────────────────────────────────────────────────
# NODE IMPLEMENTATIONS
# Each node is a pure function (or factory that returns one) that receives
# the current QueryState and returns a partial dict of updated fields.
# ─────────────────────────────────────────────────────────────────────────────

# ── Node 1: Thread Check ──────────────────────────────────────────────────────

def thread_check_node(state: QueryState) -> dict:
    """
    Detect whether this is the first message in a thread.
    Registers new threads and initialises their conversation history.
    """
    thread_id = state["thread_id"]
    is_new    = thread_id not in _thread_registry

    if is_new:
        _thread_registry[thread_id]    = True
        _conversation_store[thread_id] = []

    print(f"  [ThreadCheck] thread={thread_id!r}  new={is_new}")
    return {"is_new_thread": is_new}


# ── Node 2: Memory ────────────────────────────────────────────────────────────

def build_memory_node(memory_store: FAISS):
    """
    Factory: returns a node that retrieves the most relevant past Q&A
    exchanges from the FAISS memory store using semantic similarity.
    """
    def memory_node(state: QueryState) -> dict:
        if state.get("is_new_thread", True):
            print("  [Memory] New thread — no prior context to retrieve.")
            return {"memory_context": ""}

        docs = memory_store.similarity_search(state["query"], k=2)

        # Exclude the seed document
        relevant = [
            d.page_content for d in docs
            if d.metadata.get("type") != "seed"
        ]

        context = "\n\n".join(relevant)
        print(f"  [Memory] Retrieved {len(relevant)} relevant memory snippet(s).")
        return {"memory_context": context}

    return memory_node


# ── Node 3a: YouTube Retriever ────────────────────────────────────────────────

def build_youtube_retriever_node(youtube_store: FAISS):
    """
    Factory: returns a node that performs semantic search over the YouTube
    transcript FAISS index and returns the top-3 matching chunks.
    """
    def youtube_retriever_node(state: QueryState) -> dict:
        docs = youtube_store.similarity_search(state["query"], k=3)
        results = [
            f"[YouTube — {d.metadata.get('title', 'Unknown')}]\n{d.page_content}"
            for d in docs
        ]
        print(f"  [YouTubeRetriever] Found {len(results)} result(s).")
        return {"youtube_results": results}

    return youtube_retriever_node


# ── Node 3b: Wikipedia Retriever (MCP) ───────────────────────────────────────

def build_wiki_retriever_node(wiki_tool):
    """
    Factory: returns an ASYNC node that calls the Wikipedia MCP tool.

    The wiki_tool is a LangChain BaseTool created by MultiServerMCPClient
    from the MCP server's `search_wikipedia` definition.
    Using MCP here (not direct API calls) satisfies the MCP requirement
    and keeps retrieval decoupled from the tool implementation.
    """
    async def wiki_retriever_node(state: QueryState) -> dict:
        print(f"  [WikiRetriever] Invoking MCP tool: search_wikipedia({state['query']!r})")
        try:
            result  = await wiki_tool.ainvoke({"query": state["query"]})
            results = [result] if isinstance(result, str) else [str(result)]
        except Exception as exc:
            print(f"  [WikiRetriever] MCP call failed: {exc}")
            results = []

        print(f"  [WikiRetriever] Received {len(results)} result(s).")
        return {"wiki_results": results}

    return wiki_retriever_node


# ── Node 4: Merge ─────────────────────────────────────────────────────────────

def merge_node(state: QueryState) -> dict:
    """
    Combine YouTube and Wikipedia results into a single context string.

    Strategy:
    - YouTube results are listed first (higher priority).
    - Wikipedia results are appended if they add new information.
    - Near-duplicate chunks are removed using an 80-char fingerprint.
    """
    youtube = state.get("youtube_results", [])
    wiki    = state.get("wiki_results",    [])

    seen:   set[str]  = set()
    merged: list[str] = []

    for chunk in youtube + wiki:
        fingerprint = chunk[:80].strip().lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            merged.append(chunk)

    merged_context = "\n\n---\n\n".join(merged)
    print(f"  [Merge] Combined into {len(merged)} unique chunk(s).")
    return {"merged_context": merged_context}


# ── Node 5: Decision ──────────────────────────────────────────────────────────

def build_decision_node(llm: ChatGroq):
    """
    Factory: returns a node that asks the LLM to evaluate which source
    best answers the user's query and returns a strict JSON decision.

    Output keys: "youtube" | "wiki" | "hybrid" | "fallback"
    """
    DECISION_PROMPT = """\
You are a routing agent. Evaluate which knowledge source best answers the user query.

User Query:
{query}

YouTube Results (first 800 chars):
{yt_preview}

Wikipedia Results (first 800 chars):
{wiki_preview}

Routing rules:
- YouTube results are strongly relevant and self-sufficient  → "youtube"
- Wikipedia results are strongly relevant and self-sufficient → "wiki"
- Both sources are relevant and complementary               → "hybrid"
- Neither source provides useful information                → "fallback"

Respond with ONLY valid JSON — no markdown, no explanation:
{{"decision": "<youtube|wiki|hybrid|fallback>"}}"""

    def decision_node(state: QueryState) -> dict:
        yt_preview   = "\n".join(state.get("youtube_results", [])[:2])[:800] or "No results."
        wiki_preview = "\n".join(state.get("wiki_results",    [])[:1])[:800] or "No results."

        prompt = DECISION_PROMPT.format(
            query        = state["query"],
            yt_preview   = yt_preview,
            wiki_preview = wiki_preview,
        )

        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        # Safely parse the JSON response (strip code fences if present)
        try:
            clean    = raw.replace("```json", "").replace("```", "").strip()
            parsed   = json.loads(clean)
            decision = parsed.get("decision", "fallback")
            if decision not in ("youtube", "wiki", "hybrid", "fallback"):
                decision = "fallback"
        except (json.JSONDecodeError, AttributeError, KeyError):
            print(f"  [Decision] Could not parse LLM response: {raw!r}. Defaulting to fallback.")
            decision = "fallback"

        print(f"  [Decision] Selected source: {decision!r}")
        return {"selected_source": decision}

    return decision_node


# ── Nodes 6a-6d: Answer Nodes ─────────────────────────────────────────────────

def _build_answer_messages(
    query:        str,
    context:      str,
    memory:       str,
    source_label: str,
) -> list:
    """Shared helper: assemble the system + user messages for any answer node."""
    system_content = (
        f"You are a knowledgeable AI assistant. Answer the user's question clearly "
        f"and accurately using the provided {source_label} context. "
        f"If the context is insufficient, say so honestly."
    )
    memory_block = f"\n\nRelevant conversation history:\n{memory}" if memory else ""
    user_content = (
        f"Context ({source_label}):\n{context}"
        f"{memory_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]


def build_youtube_answer_node(llm: ChatGroq):
    """Generates an answer using only YouTube transcript context."""
    def youtube_answer_node(state: QueryState) -> dict:
        context  = "\n\n".join(state.get("youtube_results", []))
        messages = _build_answer_messages(
            state["query"], context, state.get("memory_context", ""),
            source_label="YouTube Transcripts",
        )
        answer = llm.invoke(messages).content.strip()
        print("  [YouTubeAnswer] Answer generated.")
        return {"final_answer": answer}
    return youtube_answer_node


def build_wiki_answer_node(llm: ChatGroq):
    """Generates an answer using Wikipedia context retrieved via MCP."""
    def wiki_answer_node(state: QueryState) -> dict:
        context  = "\n\n".join(state.get("wiki_results", []))
        messages = _build_answer_messages(
            state["query"], context, state.get("memory_context", ""),
            source_label="Wikipedia (via MCP)",
        )
        answer = llm.invoke(messages).content.strip()
        print("  [WikiAnswer] Answer generated.")
        return {"final_answer": answer}
    return wiki_answer_node


def build_hybrid_answer_node(llm: ChatGroq):
    """Generates an answer by combining YouTube and Wikipedia context."""
    def hybrid_answer_node(state: QueryState) -> dict:
        context  = state.get("merged_context", "")
        messages = _build_answer_messages(
            state["query"], context, state.get("memory_context", ""),
            source_label="YouTube + Wikipedia (Hybrid)",
        )
        answer = llm.invoke(messages).content.strip()
        print("  [HybridAnswer] Answer generated.")
        return {"final_answer": answer}
    return hybrid_answer_node


def build_fallback_answer_node(llm: ChatGroq):
    """
    Generates an answer from the LLM's parametric knowledge when both
    retrieval sources are insufficient.
    """
    def fallback_answer_node(state: QueryState) -> dict:
        prompt = (
            f"Answer the following question using your general knowledge. "
            f"Be honest if the topic is outside your expertise.\n\n"
            f"Question: {state['query']}"
        )
        answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print("  [FallbackAnswer] Answer generated from general knowledge.")
        return {"final_answer": answer}
    return fallback_answer_node


# ── Node 7: Save ──────────────────────────────────────────────────────────────

def build_save_node(memory_store: FAISS):
    """
    Factory: returns a node that persists the completed Q&A exchange to:
    1. In-memory conversation history (per thread).
    2. FAISS memory vector store (for future semantic retrieval).
    """
    def save_node(state: QueryState) -> dict:
        thread_id = state["thread_id"]
        query     = state["query"]
        answer    = state.get("final_answer",    "")
        source    = state.get("selected_source", "unknown")

        # Persist to conversation history
        _conversation_store.setdefault(thread_id, []).append({
            "query":  query,
            "answer": answer,
            "source": source,
        })

        # Embed the Q&A pair and add it to the memory vector store
        memory_doc = Document(
            page_content=f"Q: {query}\nA: {answer}",
            metadata={"thread_id": thread_id, "source": source, "type": "history"},
        )
        memory_store.add_documents([memory_doc])

        turn_count = len(_conversation_store[thread_id])
        print(f"  [Save] Exchange saved. Thread {thread_id!r} now has {turn_count} turn(s).")
        return {}

    return save_node


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def route_by_decision(state: QueryState) -> str:
    """
    Conditional edge function: maps the decision value to an answer node name.
    LangGraph calls this after the decision node to determine the next node.
    """
    return state.get("selected_source", "fallback")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    llm:           ChatGroq,
    youtube_store: FAISS,
    memory_store:  FAISS,
    wiki_tool,
) -> StateGraph:
    """
    Wire all nodes and edges into a compiled LangGraph StateGraph.

    Parallel retrieval is achieved by fanning out from [memory] to both
    [youtube_retriever] and [wiki_retriever]. LangGraph executes them
    concurrently and waits for both before running [merge].
    """
    builder = StateGraph(QueryState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("thread_check",       thread_check_node)
    builder.add_node("memory",             build_memory_node(memory_store))
    builder.add_node("youtube_retriever",  build_youtube_retriever_node(youtube_store))
    builder.add_node("wiki_retriever",     build_wiki_retriever_node(wiki_tool))
    builder.add_node("merge",              merge_node)
    builder.add_node("decision",           build_decision_node(llm))
    builder.add_node("youtube_answer",     build_youtube_answer_node(llm))
    builder.add_node("wiki_answer",        build_wiki_answer_node(llm))
    builder.add_node("hybrid_answer",      build_hybrid_answer_node(llm))
    builder.add_node("fallback_answer",    build_fallback_answer_node(llm))
    builder.add_node("save",               build_save_node(memory_store))

    # ── Sequential edges ──────────────────────────────────────────────────────
    builder.add_edge(START,          "thread_check")
    builder.add_edge("thread_check", "memory")

    # ── Fan-out: parallel retrieval ───────────────────────────────────────────
    builder.add_edge("memory", "youtube_retriever")
    builder.add_edge("memory", "wiki_retriever")

    # ── Fan-in: both retrievers must complete before merge ────────────────────
    builder.add_edge("youtube_retriever", "merge")
    builder.add_edge("wiki_retriever",    "merge")

    # ── Decision ──────────────────────────────────────────────────────────────
    builder.add_edge("merge", "decision")

    # ── Conditional routing based on LLM decision ─────────────────────────────
    builder.add_conditional_edges(
        source   = "decision",
        path     = route_by_decision,
        path_map = {
            "youtube":  "youtube_answer",
            "wiki":     "wiki_answer",
            "hybrid":   "hybrid_answer",
            "fallback": "fallback_answer",
        },
    )

    # ── All answer nodes converge at save → END ───────────────────────────────
    for answer_node in ("youtube_answer", "wiki_answer", "hybrid_answer", "fallback_answer"):
        builder.add_edge(answer_node, "save")

    builder.add_edge("save", END)

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(user_id: str, thread_id: str, query: str) -> dict:
    """
    End-to-end pipeline execution for a single user query.

    Steps:
      1. Initialise embeddings, vector stores, and LLM.
      2. Launch the Wikipedia MCP server as a subprocess via stdio.
      3. Build the LangGraph and run it with the provided query.
      4. Return the final answer and selected source.

    Args:
        user_id:   Unique identifier for the user.
        thread_id: Conversation thread identifier.
        query:     The user's natural-language question.

    Returns:
        dict with keys: answer, selected_source, user_id, thread_id.
    """
    print(f"\n{'═' * 62}")
    print(f"  Query   : {query}")
    print(f"  User    : {user_id}   │   Thread : {thread_id}")
    print(f"{'═' * 62}")

    # ── 1. Initialise components ──────────────────────────────────────────────
    print("\n[Init] Loading embeddings and vector stores …")
    mem_embeddings = build_embeddings()     # all-MiniLM-L6-v2  (memory store)
    yt_embeddings  = build_yt_embeddings()  # LaBSE             (YouTube ChromaDB)
    youtube_store  = build_youtube_store(yt_embeddings)
    memory_store   = build_memory_store(mem_embeddings)
    llm            = build_llm()
    print("[Init] Ready.\n")

    # ── 2. Launch Wikipedia MCP server and connect client ─────────────────────
    # The MCP server is this same file, invoked with --mcp-server.
    # In langchain-mcp-adapters ≥ 0.1.0, MultiServerMCPClient is NOT a context
    # manager. Instantiate it directly and call get_tools() — each tool invocation
    # opens its own stdio session to the subprocess automatically.
    mcp_server_config = {
        "wikipedia": {
            "command":   sys.executable,              # same Python interpreter
            "args":      [__file__, "--mcp-server"],  # this file in server mode
            "transport": "stdio",
        }
    }

    mcp_client = MultiServerMCPClient(mcp_server_config)
    tools      = await mcp_client.get_tools()
    wiki_tool  = next((t for t in tools if t.name == "search_wikipedia"), None)

    if wiki_tool is None:
        raise RuntimeError(
            "Wikipedia MCP tool 'search_wikipedia' not found. "
            "Ensure the MCP server started correctly."
        )

    print(f"[MCP] Wikipedia tool connected: {wiki_tool.name!r}\n")

    # ── 3. Build and invoke the graph ─────────────────────────────────────────
    graph = build_graph(llm, youtube_store, memory_store, wiki_tool)

    initial_state: QueryState = {
        "query":           query,
        "user_id":         user_id,
        "thread_id":       thread_id,
        "is_new_thread":   False,    # determined by thread_check_node
        "memory_context":  "",
        "youtube_results": [],
        "wiki_results":    [],
        "merged_context":  "",
        "selected_source": "",
        "final_answer":    "",
    }

    final_state = await graph.ainvoke(initial_state)

    # ── 4. Return result ──────────────────────────────────────────────────────
    result = {
        "answer":          final_state["final_answer"],
        "selected_source": final_state["selected_source"],
        "user_id":         user_id,
        "thread_id":       thread_id,
    }

    print(f"\n{'═' * 62}")
    print(f"  Source  : {result['selected_source'].upper()}")
    preview = result["answer"][:400]
    suffix  = " …" if len(result["answer"]) > 400 else ""
    print(f"  Answer  : {preview}{suffix}")
    print(f"{'═' * 62}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Test case (as specified in requirements) ──────────────────────────────
    result = asyncio.run(
        run_pipeline(
            user_id   = "user_1",
            thread_id = "thread_1",
            query     = "What is overfitting?",
        )
    )

    print("[Final Result]")
    print(json.dumps(result, indent=2))
