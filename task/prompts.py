#TODO:
# Create Prompt that will:
# - explain to LLM its role, its role is Multi Agent System coordination assistant
# - explain the task
# - give the context about available agents and their capabilities
# - provide instructions with how LLM should handle such task
COORDINATION_REQUEST_SYSTEM_PROMPT = """
{YOUR_PROMPT}
"""


#TODO:
# Create Prompt that will:
# - explain to LLM its role
# - provide LLM with context that it is working in finalization step in multi-agent system
# - provide the information about augmented user prompt (context and user request)
# - give a task
FINAL_RESPONSE_SYSTEM_PROMPT = """
{YOUR_PROMPT}
"""
