
from agents.planner_agent import plan_task
from agents.data_agent import handle_data_query

def test_query(query):
    print(f"\n{'='*50}")
    print(f"User query: {query}")
    
    # Send to planner agent
    plan = plan_task(query)
    print(f"Planner decision: {plan}")
    
    
    if plan["use_data_agent"]:
        
        result = handle_data_query(query)
        if result is not None:
            print("Query result:")
            print(result)
        else:
            print("Query failed.")
    else:
        print("No data agent needed for this query.")
    print(f"{'='*50}")

def main():
    test_queries = [
        "total sales",
        "show all data",
        "average revenue",
        "maximum sales",
        "minimum revenue",
        "count all records",
        "display sales"
    ]
    
    for query in test_queries:
        test_query(query)

if __name__ == "__main__":
    main()