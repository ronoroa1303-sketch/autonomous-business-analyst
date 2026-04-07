import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def _rule_based_plan(query: str) -> dict:
    """
    Fallback planner: simple keyword matching.
    Returns boolean flags for which agents should run.
    """
    q = query.lower()

    data_keywords = [
        "total", "sum", "average", "count", "revenue", "sales", "price",
        "payment", "top", "orders", "customers", "ranking", "list", "how many",
        "highest", "lowest", "most", "least"
    ]

    rag_keywords = [
        "why", "reason", "explain", "impact", "analysis", "analyse", "analyze",
        "context", "background", "cause", "because", "insight"
    ]

    forecast_keywords = [
        "forecast", "predict", "future", "next month", "next quarter",
        "next year", "projection", "trend"
    ]

    use_data = any(k in q for k in data_keywords)
    use_rag = any(k in q for k in rag_keywords)
    use_forecast = any(k in q for k in forecast_keywords)

    # Composite override: if both data + forecast, also run RAG for context
    if use_data and use_forecast:
        use_rag = True

    # Default: if nothing matched, at least run data agent
    if not use_data and not use_rag and not use_forecast:
        use_data = True

    return {
        "use_data_agent": use_data,
        "use_rag_agent": use_rag,
        "use_forecast_agent": use_forecast
    }


def plan_task(query: str) -> dict:
    """
    Use Groq LLM to decide which agents to run.

    Returns simple boolean flags:
      {
        "use_data_agent": true/false,
        "use_rag_agent": true/false,
        "use_forecast_agent": true/false
      }

    Falls back to rule-based logic if the LLM call fails.
    """
    print(f"plan_task called with query: {query}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("No Groq API key — using rule-based fallback")
        return _rule_based_plan(query)

    try:
        client = Groq(api_key=api_key)

        prompt = f"""You are a planner agent for a multi-agent business intelligence system.
Decide which agents are needed for the user query.

AGENT RULES:
- use_data_agent = true  → query involves numbers, metrics, counts, totals, rankings
  Examples: "top customers", "revenue", "orders", "sales", "how many"
- use_rag_agent = true   → query needs explanation, reasoning, or business context
  Examples: "why", "reason", "impact", "analysis", "explain"
- use_forecast_agent = true → query asks about the future or predictions
  Examples: "forecast", "predict", "future", "next month"

MULTIPLE agents may be true for mixed queries.

User Query: "{query}"

Return ONLY valid JSON with no extra text:
{{
  "use_data_agent": true or false,
  "use_rag_agent": true or false,
  "use_forecast_agent": true or false
}}"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=80,
        )

        raw = chat_completion.choices[0].message.content.strip()
        print(f"LLM planner raw response: '{raw}'")

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        flags = json.loads(raw)

        # Validate shape
        required_keys = {"use_data_agent", "use_rag_agent", "use_forecast_agent"}
        if required_keys.issubset(flags.keys()) and all(
            isinstance(flags[k], bool) for k in required_keys
        ):
            # Apply composite override
            if flags["use_data_agent"] and flags["use_forecast_agent"]:
                flags["use_rag_agent"] = True

            print(f"LLM plan: {flags}")
            return flags

        print("LLM response has unexpected shape, using rule-based fallback")
        return _rule_based_plan(query)

    except Exception as e:
        print(f"LLM planner error: {e} — using rule-based fallback")
        return _rule_based_plan(query)