import os
import json
import ast
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def plan_task(query: str):
    """
    Use Groq LLM to decide which agents should handle the query.
    Returns a list of agent names (strings) from the valid set:
    ["data_agent", "insight_agent", "rag_agent", "forecast_agent"]
    """
    print(f"planner_agent.llm_planner.plan_task called with query: {query}")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("No Groq API key found in llm_planner")
        # Fallback to default agents
        return ["data_agent", "insight_agent"]
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""You are a planner agent. Your job is to decide which system components should handle the user query.

Available agents:
- data_agent: for queries needing database or SQL
- insight_agent: for analysis, trends, business insights
- rag_agent: for retrieval-augmented generation (if available)
- forecast_agent: for forecasting queries (if available)

Rules:
- If the query involves numbers, data, tables -> include data_agent
- If the query involves trends, patterns, analysis -> include insight_agent
- If the query involves retrieval from external knowledge or documents -> include rag_agent
- If the query involves forecasting or predictions -> include forecast_agent

Return ONLY a list of agent names in the format of a Python list (e.g., ["data_agent", "insight_agent"]) or a JSON list.
Do not include any other text.

User Query:
{query}"""
        
    
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=10
        )
        
        response = chat_completion.choices[0].message.content.strip()
        print(f"LLM planner raw response: '{response}'")
        
        agents = None
        try:
            agents = json.loads(response)
            if not isinstance(agents, list):
                agents = None
        except:
            pass
        
        if agents is None:
            try:
                agents = ast.literal_eval(response)
                if not isinstance(agents, list):
                    agents = None
            except:
                pass
        
        if agents is None:
            # Find the first '[' and last ']'
            start = response.find('[')
            end = response.rfind(']')
            if start != -1 and end != -1 and start < end:
                list_str = response[start:end+1]
                try:
                    agents = json.loads(list_str)
                    if not isinstance(agents, list):
                        agents = None
                except:
                    try:
                        agents = ast.literal_eval(list_str)
                        if not isinstance(agents, list):
                            agents = None
                    except:
                        agents = None
        
        if agents is None:
            print("LLM planner failed to parse response, using default agents")
            agents = ["data_agent", "insight_agent"]
        else:
            # validate that each agent in the list is from the valid set
            valid_agents = {"data_agent", "insight_agent", "rag_agent", "forecast_agent"}
            agents = [agent for agent in agents if agent in valid_agents]
            
            if not agents:
                print("LLM planner returned no valid agents, using default agents")
                agents = ["data_agent", "insight_agent"]
        
        print(f"LLM planner returning agents: {agents}")
        return agents
        
    except Exception as e:
        print(f"Error in LLM planner: {e}")
        # fallback to default agents on any error
        return ["data_agent", "insight_agent"]