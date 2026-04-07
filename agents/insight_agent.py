import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def generate_insight(query: str, rows: list, row_count: int, context: str = "", forecast: list = None, trend: str = ""):
    """
    Generate structured, professional business insights.
    """
    if forecast is None:
        forecast = []

    if row_count < 3:
        return "Not enough data for meaningful analysis"

    print(f"Generating consulting-style insight for query: {query}")
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Insight generation failed: No Groq API key found."

        client = Groq(api_key=api_key)

        # Take only first 50 rows to avoid token overload
        sample_rows = rows[:50] if rows else []

        prompt = f"""You are a senior business analyst at a top consulting firm (McKinsey/Bain/BCG).

Analyze the data and provide a structured business insight.

INPUT:

User Query:
{query}

Structured Data (PRIMARY SOURCE):
{sample_rows}

Forecast:
Trend: {trend}
Forecast Values: {forecast}

External Context (OPTIONAL):
{context}

INSTRUCTIONS:
- Treat structured data as the most reliable source
- Use forecast for future trends
- Use context only if relevant (ignore if empty or INVALID_QUERY)
- Do NOT hallucinate
- Be concise and professional

OUTPUT FORMAT:
1. KEY FINDINGS:
- Main patterns, trends, or anomalies

2. BUSINESS INTERPRETATION:
- Explain why these patterns are happening

3. FUTURE OUTLOOK:
- Use forecast and trend to describe what will happen next

4. RECOMMENDATIONS:
- 2–4 practical business actions

If data is limited, clearly say so but still provide best possible insight."""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=600  # increased to allow full 4-section response
        )

        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Insight generation failed: {str(e)}"