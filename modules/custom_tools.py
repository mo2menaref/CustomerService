from langchain_core.tools import tool

# Mock database for students
STUDENT_DB = {
    "1024": {"name": "Ahmed", "attendance": "85%", "grades": "A", "late_tasks": 0},
    "2048": {"name": "Omar", "attendance": "60%", "grades": "C", "late_tasks": 2}
}

# Global retriever for RAG context
rag_retriever = None

def set_rag_retriever(retriever):
    """Set the RAG retriever for the tools to use."""
    global rag_retriever
    rag_retriever = retriever

@tool
def get_student_progress(student_id: str) -> str:
    """Use this tool to get the progress, attendance, and grades of a student using their ID."""
    # Retrieve student data from the database
    student = STUDENT_DB.get(student_id)
    if student:
        return f"Student {student['name']} has attendance {student['attendance']} and grade {student['grades']}."
    return "Student ID not found in the database."

@tool
def send_warning_email(student_id: str) -> str:
    """Use this tool to send a warning email to a student with late tasks or low attendance."""
    # Verify student exists and requires a warning
    student = STUDENT_DB.get(student_id)
    if student and student["late_tasks"] > 0:
        return f"Warning email successfully sent to {student['name']}."
    return "No warning needed or student not found."

@tool
def query_course_policies(query: str) -> str:
    """Use this tool to search course policies from the knowledge base."""
    global rag_retriever
    if not rag_retriever:
        return "Course policies database not available. Please add PDF files to the 'data' folder."
    
    try:
        docs = rag_retriever.invoke(query)
        if not docs:
            return "No relevant policies found for your query."
        
        # Combine the retrieved documents
        response = "\n".join([doc.page_content for doc in docs[:3]])
        return f"Course Policies:\n{response}"
    except Exception as e:
        return f"Error querying policies: {e}"

# List of tools to be passed to the agent
agent_tools = [get_student_progress, send_warning_email, query_course_policies]
