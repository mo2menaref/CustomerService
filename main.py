import os
from modules.graph_builder import build_agent_graph
from modules.rag_engine import setup_rag

def main():
    # Set your Gemini API key here
    os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
    
    print("Initializing system...")
    
    # Setup RAG (Ensure you have a PDF in the 'data' folder)
    retriever = setup_rag()
    agent_app = build_agent_graph(retriever)  # pass it in
    
    print("\nSystem ready! Type 'exit' to quit.")
    print("-" * 40)
    
    # Interactive chat loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting system. Goodbye!")
            break
            
        # Run the graph and stream responses
        events = agent_app.stream(
            {"messages": [("user", user_input)]}, stream_mode="values"
        )
        
        # Display the final output from the agent
        for event in events:
            event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
