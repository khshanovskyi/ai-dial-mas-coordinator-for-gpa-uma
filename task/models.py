from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    GPA = "GPA"
    UMS = "UMS"


class Subtask(BaseModel):
    task_id: int = Field(description="Unique identifier for this subtask")
    agent_name: AgentName = Field(description="Which agent should handle this subtask")
    task_description: str = Field(description="What this agent needs to do")
    depends_on: Optional[int] = Field(  # Changed from list[int] to int
        default=None,
        description="Task ID that must complete before this one"
    )


class TaskDecomposition(BaseModel):
    subtasks: list[Subtask] = Field(
        default_factory=list,
        description="List of subtasks to be executed by different agents. Leave empty when no subtasks to execute"
    )
    stop: bool = Field(
        default=False,
        description="If no subtasks to execute - mark as true, it will indicate that we are ready to the final step"
    )


class AgentResult(BaseModel):
    task_id: int
    agent_name: AgentName
    content: str
    success: bool = True
    error: Optional[str] = None

class TaskResult(BaseModel):
    task: Subtask
    agent_result: AgentResult