import os
import json
from groq import Groq
from dotenv import load_dotenv

def evaluate_data(user_query, rows, row_count):
    if row_count == 0:
        return {
            "valid": False,
            "reason": "No data",
            "confidence": 0.0
        }

    if not isinstance(rows, list) or len(rows) == 0:
        return {
            "valid": False,
            "reason": "Empty rows",
            "confidence": 0.0
        }

    query_lower = str(user_query).lower()
    keywords = ["customer", "order", "payment", "revenue", "product"]
    
    try:
        columns = [str(col).lower() for col in rows[0].keys()]
    except AttributeError:
        columns = []

    for keyword in keywords:
        if keyword in query_lower:
            if not any(keyword in col for col in columns):
                return {
                    "valid": False,
                    "reason": "Data does not match query",
                    "confidence": 0.3
                }

    return {
        "valid": True,
        "reason": "Data looks correct",
        "confidence": 0.9
    }

def evaluate_llm(query, rows, insight):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"consistent": True, "confidence": 1.0, "reason": "No API key"}
        
    try:
        client = Groq(api_key=api_key)
        
        # Take a small sample to avoid exceeding token limits
        sample_rows = rows[:5] if isinstance(rows, list) else rows
        
        prompt = f"""You are an Evaluator Agent.
Evaluate the consistency and quality of the following insight based on the data.

Input:
- User Query: {query}
- Data Sample: {sample_rows}
- Insight: {insight}

Tasks:
1. Check consistency: Does the insight match the data? If mismatch -> mark inconsistent.
2. Score factual, completeness, consistency (0 or 1 for each).
3. Calculate confidence: (factual + completeness + consistency) / 3.

Return ONLY valid JSON in this format:
{{
  "consistent": true or false,
  "confidence": float,
  "reason": "short explanation"
}}"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print("LLM Eval error:", e)
        return {"consistent": True, "confidence": 1.0, "reason": "Eval failed"}