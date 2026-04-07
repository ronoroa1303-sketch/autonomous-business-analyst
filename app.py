import streamlit as st
import requests
import pandas as pd

# CONFIG — n8n webhook URL (orchestration layer)

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/analyst-agent"

st.title("Autonomous Business Analyst Agent")

user_query = st.text_input("Enter your query:")

if st.button("Analyze"):
    if user_query:
        try:
            clean_query = str(user_query).strip()
            with st.spinner("Agents are working on your query..."):
                response = requests.post(
                    N8N_WEBHOOK_URL,
                    params={"query": clean_query},
                    timeout=60
                )
                result = response.json()

            print("DEBUG — n8n response:", result)

            # ── Extract top-level status ──
            status = result.get("status", "")

            if status == "error":
                st.error(f"Error: {result.get('message', 'Unknown error')}")
            elif status == "success":
                st.success("Analysis completed!")

                # ── Extract nested data safely ──
                data = result.get("data", {})

                query    = data.get("query", clean_query)
                rows     = data.get("rows", [])
                row_count = data.get("row_count", 0)
                insight  = data.get("insight", "")
                context  = data.get("context", "")
                forecast = data.get("forecast", {})
                trend    = data.get("trend", "")

                # If forecast is a dict with nested "trend", extract it
                if isinstance(forecast, dict) and not trend:
                    trend = forecast.get("trend", "")

                st.write(f"**Query:** {query}")
                st.write(f"Found **{row_count}** results.")

                # ── Insight Section ──
                if insight:
                    st.subheader("📊 Business Insight")
                    st.markdown(insight)

                # ── Data Table Section ──
                if rows:
                    st.subheader("📋 Data")
                    df = pd.DataFrame(rows)

                    # Convert revenue column to numeric for proper charting
                    if "revenue" in df.columns:
                        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

                    st.dataframe(df)

                    # Optional: Show line chart if month + revenue columns exist
                    if "month" in df.columns and "revenue" in df.columns:
                        st.line_chart(df.set_index("month")["revenue"])

                # ── Forecast Section ──
                forecast_values = []
                if isinstance(forecast, dict):
                    forecast_values = forecast.get("forecast", [])
                elif isinstance(forecast, list):
                    forecast_values = forecast

                if forecast_values or trend:
                    st.subheader("🔮 Forecast")
                    if trend:
                        st.write(f"**Trend:** {trend}")
                    if forecast_values:
                        forecast_df = pd.DataFrame(forecast_values)
                        st.dataframe(forecast_df)

                # ── RAG Context Section ──
                if context and context != "INVALID_QUERY":
                    with st.expander("📄 Retrieved Context (RAG)"):
                        st.write(context)

            else:
                st.error(f"Unexpected response from n8n: {result}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to n8n. Make sure n8n is running on port 5678.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The query may be too complex or n8n is not responding.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
    else:
        st.warning("Please enter a query.")