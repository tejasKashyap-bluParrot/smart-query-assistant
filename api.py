#!/usr/bin/env python3
"""
api.py  —  FastAPI wrapper for the Smart Query Assistant pipeline
=================================================================
Run:
    conda activate bluParrot
    uvicorn api:app --reload --port 8000

Then POST to:
    http://localhost:8000/query
"""

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_pipeline

app = FastAPI(title="Smart Query Assistant")


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:     str
    user_id:   str = "user_1"
    thread_id: str = "thread_1"

class QueryResponse(BaseModel):
    answer:          str
    selected_source: str
    user_id:         str
    thread_id:       str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = await run_pipeline(
        user_id   = request.user_id,
        thread_id = request.thread_id,
        query     = request.query,
    )
    return result
