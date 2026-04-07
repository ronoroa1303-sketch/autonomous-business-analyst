import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_sql(query: str):
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("No Groq API key found")  
        return None

    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Create prompt for SQL generation
        prompt = f"""
Convert this user query into SQL.

Database Schema:
orders(order_id, customer_id, order_purchase_timestamp)
payments(order_id, payment_value)
order_items(order_id, product_id, price)

SCHEMA RULES:
payment_value exists ONLY in payments table
price exists ONLY in order_items table
customer_id exists ONLY in orders table
orders table does NOT contain payment_value or price
order_items table does NOT contain customer_id or payment_value
payments table does NOT contain price or customer_id
Always JOIN payments when using payment_value
Always JOIN order_items when using price
To get customer_id with payments or order_items, MUST join through orders table
Use correct joins using order_id
Always prefix columns with table name or alias
Avoid ambiguous columns
Double-check that every column used exists in the referenced table

SQL DIALECT RULES (SQLite):
Use ONLY SQLite-compatible SQL
DO NOT use DATE_DIFF, UNIX_TIMESTAMP, SEPARATOR, or advanced functions
DO NOT use window functions (LAG, LEAD)
Use only basic SQL: SELECT, JOIN, GROUP BY, ORDER BY, LIMIT
payments table has NO timestamp column
orders table contains order_purchase_timestamp

ANALYSIS RULES:
For trend/time queries → GROUP BY time using strftime('%Y-%m', order_purchase_timestamp)
For "increasing", "growth", "over time" → return aggregated values (SUM, AVG) grouped by time
Do NOT return raw row-level data
Always return structured aggregated results

SIMPLICITY RULES:
Use simple SQL only
Avoid unnecessary joins
Avoid subqueries unless absolutely necessary
Prefer GROUP BY over complex logic
Do NOT format output inside SQL

TREND HANDLING RULE (CRITICAL):
If query asks for "increase", "growth", "trend", "over time", "decrease":
→ You MUST return: customer_id, strftime('%Y-%m', order_purchase_timestamp) AS time_period, SUM(payment_value) AS total_spent
→ STRICT REQUIREMENTS:
   • ALWAYS include aggregation using SUM(payment_value)
   • ALWAYS include time grouping using strftime('%Y-%m', order_purchase_timestamp)
   • NEVER return only customer_id
   • NEVER skip GROUP BY
   • NEVER return raw row-level data
→ EXPECTED SQL FORMAT:
SELECT
  o.customer_id,
  strftime('%Y-%m', o.order_purchase_timestamp) AS time_period,
  SUM(p.payment_value) AS total_spent
FROM orders o
JOIN payments p ON o.order_id = p.order_id
GROUP BY o.customer_id, time_period
ORDER BY o.customer_id, time_period;

SPECIAL RULES:
- If query mentions "customer" → include o.customer_id
- If query mentions "customers" → GROUP BY o.customer_id
- If query mentions both "customer" and "trend/increase"
  → GROUP BY o.customer_id AND strftime('%Y-%m', order_purchase_timestamp)

CRITICAL COLUMN RULES:
- order_purchase_timestamp exists ONLY in orders table (use o.order_purchase_timestamp)
- NEVER use order_purchase_timestamp from payments or order_items
- If using time → MUST use orders table alias (o)

VALIDATION RULE:
- Before returning SQL, verify:
  - Each column exists in the correct table
  - If column does not belong to table → fix it

COMMON MISTAKE PREVENTION:
- payments table has NO timestamp columns
- ALWAYS use orders table for date/time operations

OUTPUT RULES:
Return ONLY ONE SQL query
Do NOT include explanations
Do NOT include markdown
Output must start with SELECT

User Query: {query}

Return ONLY SQL query.
"""
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=500
        )
        
        # Extract SQL from response
        sql = chat_completion.choices[0].message.content.strip()
        
        # Debug: Print raw LLM output
        print("Raw LLM output:", chat_completion.choices[0].message.content)
        
        # Remove markdown formatting
        if "```" in sql:
            sql = sql.replace("```sql", "").replace("```", "").strip()

        # Keep only first SQL statement
        if ";" in sql:
            sql = sql.split(";")[0] + ";"

        # Ensure SQL starts from SELECT
        if "SELECT" in sql:
            sql = sql[sql.find("SELECT"):]
        
        # Debug: Print final SQL
        print("Final SQL:", sql)
        
        return sql
        
    except Exception as e:
        print("Groq error:", e)
        return None