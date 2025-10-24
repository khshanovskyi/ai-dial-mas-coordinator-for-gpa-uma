import asyncio
import json
from copy import deepcopy
from typing import Any, List, Dict, Optional

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import TaskDecomposition, Subtask, AgentResult, AgentName
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

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
        )

        decomposition_stage = StageProcessor.open_stage(choice, "Task Decomposition")
        task_decomposition = await self._decompose_task(
            client=client,
            request=request,
        )
        logger.info(f"Task decomposition: {task_decomposition.model_dump_json()}")
        decomposition_stage.append_content(
            f"```json\n{task_decomposition.model_dump_json(indent=2)}\n```\n"
        )
        StageProcessor.close_stage_safely(decomposition_stage)

        if not task_decomposition.requires_collaboration:
            result = await self._execute_single_agent(
                choice=choice,
                request=request,
                subtask=task_decomposition.subtasks[0]
            )
            return result
        else:
            agent_results = await self._execute_multi_agent(
                choice=choice,
                request=request,
                task_decomposition=task_decomposition
            )

            aggregation_stage = StageProcessor.open_stage(choice, "Aggregating Results")
            final_response = await self._aggregate_results(
                client=client,
                request=request,
                choice=choice,
                agent_results=agent_results,
                stage=aggregation_stage
            )
            StageProcessor.close_stage_safely(aggregation_stage)

            logger.info(f"Final aggregated response: {final_response.json()}")
            return final_response

    async def _decompose_task(
            self,
            client: AsyncDial,
            request: Request
    ) -> TaskDecomposition:
        response = await client.chat.completions.create(
            messages=self._prepare_messages(request, TASK_DECOMPOSITION_SYSTEM_PROMPT),
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
        return TaskDecomposition.model_validate(dict_content)

    async def _execute_single_agent(
            self,
            choice: Choice,
            request: Request,
            subtask: Subtask
    ) -> Message:
        stage = StageProcessor.open_stage(choice, f"Executing {subtask.agent_name} Agent")

        result = await self._call_agent(
            agent_name=subtask.agent_name,
            task_description=subtask.task_description,
            choice=choice,
            request=request,
            stage=stage,
            context=None
        )

        StageProcessor.close_stage_safely(stage)
        return result

    async def _execute_multi_agent(
            self,
            choice: Choice,
            request: Request,
            task_decomposition: TaskDecomposition
    ) -> List[AgentResult]:
        results: Dict[int, AgentResult] = {}
        subtasks_by_id = {st.task_id: st for st in task_decomposition.subtasks}

        dependency_graph = self._build_dependency_graph(task_decomposition.subtasks)

        if task_decomposition.execution_strategy == "parallel":
            results = await self._execute_parallel(
                choice=choice,
                request=request,
                subtasks_by_id=subtasks_by_id,
                dependency_graph=dependency_graph
            )
        else:
            results = await self._execute_sequential(
                choice=choice,
                request=request,
                subtasks_by_id=subtasks_by_id,
                dependency_graph=dependency_graph
            )

        return list(results.values())

    def _build_dependency_graph(self, subtasks: List[Subtask]) -> Dict[int, List[int]]:
        graph = {st.task_id: st.depends_on or [] for st in subtasks}
        return graph

    async def _execute_sequential(
            self,
            choice: Choice,
            request: Request,
            subtasks_by_id: Dict[int, Subtask],
            dependency_graph: Dict[int, List[int]]
    ) -> Dict[int, AgentResult]:
        results: Dict[int, AgentResult] = {}

        execution_order = self._topological_sort(dependency_graph)

        for task_id in execution_order:
            subtask = subtasks_by_id[task_id]

            context = self._gather_context(subtask, results)

            stage = StageProcessor.open_stage(
                choice,
                f"Task {task_id}: {subtask.agent_name} Agent"
            )

            try:
                message = await self._call_agent(
                    agent_name=subtask.agent_name,
                    task_description=subtask.task_description,
                    choice=choice,
                    request=request,
                    stage=stage,
                    context=context
                )

                results[task_id] = AgentResult(
                    task_id=task_id,
                    agent_name=subtask.agent_name,
                    content=message.content,
                    success=True
                )

            except Exception as e:
                logger.error(f"Error executing task {task_id}: {e}")
                results[task_id] = AgentResult(
                    task_id=task_id,
                    agent_name=subtask.agent_name,
                    content="",
                    success=False,
                    error=str(e)
                )
            finally:
                StageProcessor.close_stage_safely(stage)

        return results

    async def _execute_parallel(
            self,
            choice: Choice,
            request: Request,
            subtasks_by_id: Dict[int, Subtask],
            dependency_graph: Dict[int, List[int]]
    ) -> Dict[int, AgentResult]:
        results: Dict[int, AgentResult] = {}

        levels = self._get_dependency_levels(dependency_graph)

        for level_tasks in levels:
            tasks = []
            for task_id in level_tasks:
                subtask = subtasks_by_id[task_id]
                context = self._gather_context(subtask, results)

                tasks.append(
                    self._execute_subtask_with_stage(
                        task_id=task_id,
                        subtask=subtask,
                        choice=choice,
                        request=request,
                        context=context
                    )
                )

            level_results = await asyncio.gather(*tasks, return_exceptions=True)

            for task_result in level_results:
                if isinstance(task_result, AgentResult):
                    results[task_result.task_id] = task_result

        return results

    async def _execute_subtask_with_stage(
            self,
            task_id: int,
            subtask: Subtask,
            choice: Choice,
            request: Request,
            context: Optional[str]
    ) -> AgentResult:
        stage = StageProcessor.open_stage(
            choice,
            f"Task {task_id}: {subtask.agent_name} Agent"
        )

        try:
            message = await self._call_agent(
                agent_name=subtask.agent_name,
                task_description=subtask.task_description,
                choice=choice,
                request=request,
                stage=stage,
                context=context
            )

            return AgentResult(
                task_id=task_id,
                agent_name=subtask.agent_name,
                content=message.content,
                success=True
            )

        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            return AgentResult(
                task_id=task_id,
                agent_name=subtask.agent_name,
                content="",
                success=False,
                error=str(e)
            )
        finally:
            StageProcessor.close_stage_safely(stage)

    def _topological_sort(self, graph: Dict[int, List[int]]) -> List[int]:
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

        queue = [node for node in graph if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _get_dependency_levels(self, graph: Dict[int, List[int]]) -> List[List[int]]:
        in_degree = {node: len(deps) for node, deps in graph.items()}
        levels = []

        while in_degree:
            current_level = [node for node, degree in in_degree.items() if degree == 0]
            if not current_level:
                break

            levels.append(current_level)

            for node in current_level:
                del in_degree[node]

            for node in in_degree:
                in_degree[node] = len([dep for dep in graph[node] if dep not in current_level])

        return levels

    def _gather_context(self, subtask: Subtask, results: Dict[int, AgentResult]) -> Optional[str]:
        if not subtask.depends_on:
            return None

        context_parts = []
        for dep_id in subtask.depends_on:
            if dep_id in results and results[dep_id].success:
                context_parts.append(
                    f"## Result from Task {dep_id} ({results[dep_id].agent_name}):\n"
                    f"{results[dep_id].content}\n"
                )

        return "\n".join(context_parts) if context_parts else None

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
            )
        elif agent_name == AgentName.UMS:
            return await self.ums_gateway.response(
                choice=choice,
                request=request,
                stage=stage,
                task_description=instruction,
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    async def _aggregate_results(
            self,
            client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_results: List[AgentResult],
            stage: Stage
    ) -> Message:
        results_context = "# Agent Results:\n\n"
        for result in agent_results:
            if result.success:
                results_context += f"## Task {result.task_id} - {result.agent_name} Agent:\n"
                results_context += f"{result.content}\n\n"
            else:
                results_context += f"## Task {result.task_id} - {result.agent_name} Agent FAILED:\n"
                results_context += f"Error: {result.error}\n\n"

        msgs = self._prepare_messages(request, AGGREGATION_SYSTEM_PROMPT)
        original_user_request = msgs[-1]["content"]

        msgs[-1]["content"] = (
            f"{results_context}\n"
            f"---\n\n"
            f"# Original User Request:\n{original_user_request}\n\n"
            f"Please synthesize the agent results into a coherent response for the user."
        )

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

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
        )

    def _prepare_messages(
            self,
            request: Request,
            system_prompt: str
    ) -> list[dict[str, Any]]:
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