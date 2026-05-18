import os
from modules.graph_builder import build_agent_graph
from modules.rag_engine import setup_rag

def main():
    """Main entry point for the customer service agent."""
    # Get API key from environment or use placeholder
    api_key = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
    
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠️  Warning: GOOGLE_API_KEY not set. Please set it before running.")
        print("   Run: export GOOGLE_API_KEY='your-key-here'")
        return
    
    os.environ["GOOGLE_API_KEY"] = api_key
    
    print("\n🚀 Initializing Customer Service Agent...")
    
    # Setup RAG (Ensure you have PDFs in the 'data' folder)
    retriever = setup_rag()
    
    # Build the LangGraph Agent
    try:
        agent_app = build_agent_graph(retriever)
        print("\n✓ System ready! Type 'exit' or 'quit' to quit.")
        print("-" * 50)
    except Exception as e:
        print(f"✗ Error building agent: {e}")
        return
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Exiting system. Goodbye!")
                break
            
            # Run the graph and stream responses
            print("\n🤖 Agent:", end=" ")
            events = agent_app.stream(
                {"messages": [("user", user_input)]}, stream_mode="values"
            )
            
            # Display the final output from the agent
            for event in events:
                if event["messages"]:
                    event["messages"][-1].pretty_print()
        
        except KeyboardInterrupt:
            print("\n\n👋 System interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            continue

if __name__ == "__main__":
    main()
