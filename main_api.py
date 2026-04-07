from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agents.planner_agent import plan_task
from agents.data_agent import handle_data_query
from agents.insight_agent import generate_insight
from agents.evaluator_agent import evaluate_data, evaluate_llm
from rag.rag_agent import retrieve_context
from agents.forecast_agent import run_forecast
from typing import List, Optional

app = FastAPI()


# ─────────────────────────────────────────────────────────────────────────────
# Global error handler — catches Pydantic validation errors cleanly
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Invalid request format"}
    )


@app.get("/")
def home():
    return {"status": "success", "message": "API is running"}


# ─────────────────────────────────────────────────────────────────────────────
# Planner Agent — returns boolean flags for n8n parallel routing
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/planner")
def planner_route(query: str):
    if not query:
        return {"status": "error", "message": "Query is empty"}

    # Prompt injection protection
    malicious_patterns = ["ignore instructions", "override", "act as", "instead say"]
    if any(pattern in query.lower() for pattern in malicious_patterns):
        return {"status": "error", "message": "Invalid query"}

    plan = plan_task(query)
    return {"status": "success", "data": plan}


# ─────────────────────────────────────────────────────────────────────────────
# RAG Agent — retrieves relevant document context with threshold filtering
# ─────────────────────────────────────────────────────────────────────────────

class RagInput(BaseModel):
    query: str

@app.post("/rag")
def rag_route(data: RagInput):
    """
    Retrieve relevant document chunks for the given query.
    Returns INVALID_QUERY if no chunk passes similarity threshold.
    """
    if not data.query.strip():
        return {"status": "error", "data": {"context": "INVALID_QUERY"}}

    context = retrieve_context(data.query, top_k=3)
    return {"status": "success", "data": {"context": context}}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator Agent — validates agent outputs
# ─────────────────────────────────────────────────────────────────────────────

class EvaluateInput(BaseModel):
    query: str
    rows: list = []
    row_count: int = 0
    context: str = ""
    forecast: dict = {}

@app.post("/evaluate")
def evaluate_route(data: EvaluateInput):
    print(f"Evaluator received query: {data.query}")
    
    # Extract fields safely
    query = data.query
    rows = data.rows if data.rows else []
    row_count = data.row_count if data.row_count else 0
    context = "" if data.context == "INVALID_QUERY" else data.context
    forecast = data.forecast if data.forecast else {}
    
    # Rule based evaluation
    is_valid = bool(rows and row_count >= 5)
    
    confidence = 0.5
    if rows:
        if forecast:
            confidence = 0.9
        else:
            confidence = 0.7
            
    return {
        "status": "success",
        "data": {
            "valid": is_valid,
            "confidence": confidence,
            "query": query,
            "rows": rows,
            "row_count": row_count,
            "context": context,
            "forecast": forecast
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Agent — executes SQL and returns structured rows
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/data")
def data(query: str = ""):
    print("DATA AGENT CALLED ONCE")
    print("User query received (DATA):", query)

    if not query:
        return {"status": "error", "message": "Query is empty"}
    
    # Prompt injection protection
    malicious_patterns = ["ignore instructions", "override", "act as", "instead say"]
    if any(pattern in query.lower() for pattern in malicious_patterns):
        return {"status": "error", "message": "Invalid query"}

    result = handle_data_query(query)

    if result.get("status") == "error":
        return result

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Insight Agent — generates business insights from rows + RAG context
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/insight")
def insight_route(payload: dict):
    """
    Generate a business insight from structured DB rows, RAG context,
    and optional forecast data.

    Handles gracefully:
      - Missing rows → generates insight from context/forecast alone
      - context == "INVALID_QUERY" → ignores context, uses rows only
      - Both missing → returns error
    """
    query     = payload.get("query", "")
    rows      = payload.get("rows", [])
    row_count = payload.get("row_count", 0)
    context   = payload.get("context", "")
    forecast  = payload.get("forecast", [])   # list of {date, predicted_value}
    trend     = payload.get("trend", "")       # "increasing" / "decreasing" / "stable"

    # Prompt injection protection
    malicious_patterns = ["ignore instructions", "override", "act as", "instead say"]
    query_lower = query.lower() if isinstance(query, str) else str(query).lower()
    if any(pattern in query_lower for pattern in malicious_patterns):
        return {"status": "error", "message": "Invalid query"}

    # Ensure query is a plain string
    if isinstance(query, dict):
        query = query.get("query", "")
    if not isinstance(query, str):
        query = str(query)

    # Sanitize context
    if context == "INVALID_QUERY":
        print("Insight: RAG context was INVALID_QUERY — ignoring")
        context = ""

    print(f"Insight request received | rows: {row_count} | context length: {len(context)} | forecast points: {len(forecast) if isinstance(forecast, list) else 0}")

    # Block only if there is absolutely nothing to work with
    has_rows     = bool(rows) and row_count > 0
    has_forecast = isinstance(forecast, list) and len(forecast) > 0
    has_context  = bool(context.strip())

    if not has_rows and not has_forecast and not has_context:
        return {"status": "error", "message": "No data found for this query"}

    # Just pass everything directly into the new consulting-style insight function
    insight = generate_insight(query, rows, row_count, context=context, forecast=forecast, trend=trend)

    return {
        "status": "success",
        "data": {
            "insight": insight,
            "row_count": row_count
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forecast Agent — predicts future values from time-series data
# ─────────────────────────────────────────────────────────────────────────────

def transform_to_timeseries(rows: list) -> list:
    """
    Convert Data Agent rows into the format Forecast Agent expects.

    Data Agent may return different schemas:
      - {"date": "…", "num_events": 5}
      - {"date": "…", "count": 5}
      - {"date": "…", "value": 5}

    This normalises them all into:
      [{"date": "YYYY-MM-DD", "value": number}, ...]

    Returns an empty list if no valid rows can be converted.
    """
    value_keys = ["value", "num_events", "count", "total", "revenue", "amount"]

    result = []
    for row in rows:
        if "date" not in row:
            continue

        val = None
        for k in value_keys:
            if k in row:
                try:
                    val = float(row[k])
                except (ValueError, TypeError):
                    continue
                break

        if val is not None:
            result.append({"date": str(row["date"]), "value": val})

    return result


@app.post("/forecast")
def forecast_route(payload: dict):
    """
    Forecast the next 7 days from time-series data.

    Accepts EITHER format:
      A) {"time_series": [{"date": "…", "value": …}, ...]}
      B) {"rows": [{"date": "…", "num_events": …}, ...]}

    If 'rows' is provided, it is auto-converted to time_series.
    """
    print("FORECAST AGENT CALLED ONCE")

    # Step 1: Build time_series from whichever input was sent
    time_series = []

    if "time_series" in payload and payload["time_series"]:
        raw = payload["time_series"]
        time_series = [
            {"date": str(p.get("date", "")), "value": float(p.get("value", 0))}
            for p in raw
            if "date" in p and "value" in p
        ]
    elif "rows" in payload and payload["rows"]:
        time_series = transform_to_timeseries(payload["rows"])

    print(f"Forecast input | transformed data points: {len(time_series)}")

    # Step 2: Validate
    if len(time_series) < 5:
        return {
            "status": "error",
            "message": "Insufficient data for forecasting"
        }

    # Step 3: Run forecast
    result = run_forecast(time_series)
    return {"status": "success", "data": result}