from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    GPA = "GPA"
    UMS = "UMS"


class CoordinationRequest(BaseModel):
    agent_name: AgentName = Field(
        description=(
            "Agent name. GPA (General-purpose Agent) is used to work with general task and answering user questions, "
            "WEB search, RAG Search through documents, Content retrieval from documents, Calculations with PythonCodeInterpreter. "
            "UMS (Users Management Service agent) is used to work with users withing Users Management Service.")
    )
    additional_instructions: Optional[str] = Field(
        default=None,
        description="**Optional**: Additional instructions to Agent."
    )