import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.custom_tools import agent_tools

# Define the state of the graph (Memory)
class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_agent_graph():
    # Initialize the graph
    graph_builder = StateGraph(State)
    
    # Initialize the Gemini model and bind the custom tools
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 
    llm_with_tools = llm.bind_tools(agent_tools)
    
    # Define the chatbot node
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # Add nodes to the graph
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=agent_tools)
    graph_builder.add_node("tools", tool_node)
    
    # Define routing (Edges)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition, # Routes to tools if LLM calls a tool, else END
    )
    graph_builder.add_edge("tools", "chatbot") # Return to chatbot after tool execution
    
    # Compile the graph
    return graph_builder.compile()