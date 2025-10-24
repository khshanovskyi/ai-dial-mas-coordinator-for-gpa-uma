from enum import StrEnum
from typing import Optional, List

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    GPA = "GPA"
    UMS = "UMS"


class Subtask(BaseModel):
    task_id: int = Field(description="Unique identifier for this subtask")
    agent_name: AgentName = Field(description="Which agent should handle this subtask")
    task_description: str = Field(description="What this agent needs to do")
    depends_on: Optional[List[int]] = Field(
        default=None,
        description="Indices of subtasks that must complete before this one"
    )


class TaskDecomposition(BaseModel):
    requires_collaboration: bool = Field(
        description="Whether multiple agents need to work together on this request"
    )
    subtasks: List[Subtask] = Field(
        default_factory=list,
        description="List of subtasks to be executed by different agents"
    )
    execution_strategy: str = Field(
        default="sequential",
        description="'parallel' or 'sequential' execution of independent subtasks"
    )


class AgentResult(BaseModel):
    task_id: int
    agent_name: AgentName
    content: str
    success: bool = True
    error: Optional[str] = None