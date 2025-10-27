import json
from copy import deepcopy
from typing import Any, Optional

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.constants import UMS_CONVERSATION_ID, GPA_MESSAGES
from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import TaskDecomposition, Subtask, AgentResult, AgentName, TaskResult
from task.prompts import TASK_DECOMPOSITION_SYSTEM_PROMPT, AGGREGATION_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint
        self.gpa_gateway = GPAGateway(endpoint=self.endpoint)
        self.ums_gateway = UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint)

        self.gpa_intermediate_state: list = []
        self.ums_conversation_id: Optional[str] = None

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
        )
        task_results: list[TaskResult] = []

        task_calls = 0
        max_task_calls = 10
        while task_calls < max_task_calls:
            task_calls += 1  # to avoid infinite loop

            task_decomposition = await self._decompose_task(
                client=client,
                request=request,
                choice=choice,
                previous_results=task_results
            )

            if not task_decomposition.subtasks or task_decomposition.stop:
                logger.warning(f"No subtasks in iteration, stopping")
                break

            iteration_results = await self._execute_subtasks(
                choice=choice,
                request=request,
                subtasks=task_decomposition.subtasks,
            )

            task_results.extend(iteration_results)


        final_response = await self._aggregate_results(
            client=client,
            request=request,
            choice=choice,
            task_results=task_results,
        )

        choice.set_state(
            {
                GPA_MESSAGES: self.gpa_intermediate_state,
                UMS_CONVERSATION_ID: self.ums_conversation_id
            }
        )

        logger.info(f"Final aggregated response: {final_response.json()}")
        return final_response

    async def _decompose_task(
            self,
            client: AsyncDial,
            choice: Choice,
            request: Request,
            previous_results: list[TaskResult]
    ) -> TaskDecomposition:
        stage = StageProcessor.open_stage(choice, f"Task Decomposition")
        stage.append_content("## Content:\n")

        msgs = self._prepare_messages(request, TASK_DECOMPOSITION_SYSTEM_PROMPT)

        if previous_results and previous_results:
            results_context = "## Results from previous tasks:\n\n"
            for result in previous_results:
                results_context += f"{result.model_dump_json()}\n"

            msgs[-1]["content"] = f"{results_context}\n---\n\n{msgs[-1]['content']}"

        stage.append_content(f"\n```text\n{request.messages[-1].content}\n```\n")

        response = await client.chat.completions.create(
            messages=msgs,
            deployment_name=self.deployment_name,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "task_decomposition",
                        "schema": TaskDecomposition.model_json_schema()
                    }
                },
            },
            api_version="2024-08-01-preview"
        )

        dict_content = json.loads(response.choices[0].message.content)
        task_decomposition = TaskDecomposition.model_validate(dict_content)

        logger.info(f"Task decomposition: {task_decomposition.model_dump_json()}")

        stage.append_content("## Decomposed task:\n")
        stage.append_content(f"```json\n{task_decomposition.model_dump_json(indent=2)}\n```\n")
        StageProcessor.close_stage_safely(stage)

        return task_decomposition

    async def _execute_subtasks(self, choice: Choice, request: Request, subtasks: list[Subtask]) -> list[TaskResult]:
        results: dict[int, TaskResult] = {}
        subtasks_by_id = {subtask.task_id: subtask for subtask in subtasks}
        execution_order = self._topological_sort(subtasks)

        for task_id in execution_order:
            subtask = subtasks_by_id[task_id]

            context = self._gather_context(subtask, results)

            stage = StageProcessor.open_stage(choice, f"Task {task_id}: {subtask.agent_name}")

            try:
                message = await self._call_agent(
                    agent_name=subtask.agent_name,
                    task_description=subtask.task_description,
                    choice=choice,
                    request=request,
                    stage=stage,
                    context=context
                )
                if message.custom_content and message.custom_content.state:
                    state: dict[str, Any] = message.custom_content.state
                    if state.get(UMS_CONVERSATION_ID):
                        self.ums_conversation_id = state[UMS_CONVERSATION_ID]
                    elif state.get(GPA_MESSAGES):
                        self.gpa_intermediate_state.extend(state[GPA_MESSAGES])

                results[task_id] = TaskResult(
                    task=subtask,
                    agent_result=AgentResult(
                        task_id=subtask.task_id,
                        agent_name=subtask.agent_name,
                        content=message.content,
                        success=True
                    )
                )

            except Exception as e:
                logger.error(f"Error executing task {task_id}: {e}", exc_info=True)
                results[task_id] = TaskResult(
                    task=subtask,
                    agent_result=AgentResult(
                        task_id=subtask.task_id,
                        agent_name=subtask.agent_name,
                        content="",
                        success=False,
                        error=str(e)
                    ))
            finally:
                StageProcessor.close_stage_safely(stage)

        return list(results.values())

    def _topological_sort(self, subtasks: list[Subtask]) -> list[int]:
        children = {subtask.task_id: [] for subtask in subtasks}
        parent_count = {subtask.task_id: 0 for subtask in subtasks}

        for subtask in subtasks:
            if subtask.depends_on is not None:
                children[subtask.depends_on].append(subtask.task_id)
                parent_count[subtask.task_id] = 1

        result = []
        queue = [st.task_id for st in subtasks if parent_count[st.task_id] == 0]

        while queue:
            task_id = queue.pop(0)
            result.append(task_id)

            for child_id in sorted(children[task_id]):
                parent_count[child_id] -= 1
                if parent_count[child_id] == 0:
                    queue.append(child_id)

        return result

    def _gather_context(self, subtask: Subtask, results: dict[int, TaskResult]) -> Optional[str]:
        if subtask.depends_on is None:
            return None

        if subtask.depends_on in results and results[subtask.depends_on].agent_result.success:
            return results[subtask.depends_on].model_dump_json()

        return None

    async def _call_agent(
            self,
            agent_name: AgentName,
            task_description: str,
            choice: Choice,
            request: Request,
            stage: Stage,
            context: Optional[str]
    ) -> Message:
        instruction = task_description
        if context:
            instruction = f"{context}\n\n---\n\nYour Task: {task_description}"

        if agent_name == AgentName.GPA:
            return await self.gpa_gateway.response(
                choice=choice,
                request=request,
                stage=stage,
                task_description=instruction,
                gpa_intermediate_state=self.gpa_intermediate_state,
            )
        elif agent_name == AgentName.UMS:
            return await self.ums_gateway.response(
                request=request,
                stage=stage,
                task_description=instruction,
                ums_conversation_id=self.ums_conversation_id,
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    async def _aggregate_results(
            self,
            client: AsyncDial,
            choice: Choice,
            request: Request,
            task_results: list[TaskResult],
    ) -> Message:
        stage = StageProcessor.open_stage(choice, "Aggregating Results")

        results_context = "# Tasks Results:\n\n"
        for result in task_results:
            results_context += f"{result.model_dump_json()}\n"

        msgs = self._prepare_messages(request, AGGREGATION_SYSTEM_PROMPT)
        original_user_request = msgs[-1]["content"]

        msgs[-1]["content"] = (
            f"{results_context}\n"
            f"---\n\n"
            f"# Original User Request:\n{original_user_request}\n\n"
            f"Please synthesize the agent results into a coherent response for the user."
        )

        stage.append_content("## Request: ")
        stage.append_content(f"\n```text\n{msgs[-1]["content"]}\n```\n")
        stage.append_content("## Response: \n")

        chunks = await client.chat.completions.create(
            stream=True,
            messages=msgs,
            deployment_name=self.deployment_name
        )

        content = ''
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    stage.append_content(delta.content)
                    choice.append_content(delta.content)
                    content += delta.content

        StageProcessor.close_stage_safely(stage)
        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
        )

    def _prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        msgs = [
            {
                "role": Role.SYSTEM,
                "content": system_prompt,
            }
        ]

        for msg in request.messages:
            if msg.role == Role.USER and msg.custom_content:
                copied_msg = deepcopy(msg)
                msgs.append(
                    {
                        "role": Role.USER,
                        "content": StrictStr(copied_msg.content),
                    }
                )
            else:
                msgs.append(msg.dict(exclude_none=True))

        return msgs
