import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
        )

        coordination_stage = StageProcessor.open_stage(choice, "Coordination Request")
        coordination_request = await self.__prepare_coordination_request(
            client=client,
            request=request,
        )
        logger.info(f"coordination_request: {coordination_request.model_dump_json()}")
        coordination_stage.append_content(f"```json\n\r{coordination_request.model_dump_json(indent=2)}\n\r```\n\r")
        StageProcessor.close_stage_safely(coordination_stage)

        processing_stage = StageProcessor.open_stage(choice, f"Call {coordination_request.agent_name} Agent")
        agent_message = await self.__handle_coordination_request(
            coordination_request=coordination_request,
            choice=choice,
            stage=processing_stage,
            request=request,
        )
        logger.info(f"Agent response: {agent_message.json()}")
        StageProcessor.close_stage_safely(processing_stage)

        final_response = await self.__final_response(
            client=client,
            request=request,
            choice=choice,
            agent_message=agent_message,
        )

        logger.info(f"Final response: {final_response.json()}")

        return final_response

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        response = await client.chat.completions.create(
            messages=self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT),
            deployment_name=self.deployment_name,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                },
            },
            api_version="2024-08-01-preview"
        )

        dict_content = json.loads(response.choices[0].message.content)
        return CoordinationRequest.model_validate(dict_content)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
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

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        if coordination_request.agent_name is AgentName.GPA:
            return await GPAGateway(endpoint=self.endpoint).response(
                choice=choice,
                request=request,
                stage=stage,
                additional_instructions=coordination_request.additional_instructions,
            )

        elif coordination_request.agent_name is AgentName.UMS:
            return await UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint).response(
                choice=choice,
                request=request,
                stage=stage,
                additional_instructions=coordination_request.additional_instructions,
            )
        else:
            raise ValueError("Unknown Agent Name")

    async def __final_response(self, client: AsyncDial, choice: Choice, request: Request,
                               agent_message: Message) -> Message:
        msgs = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)

        updated_user_request = f"## CONTEXT:\n {agent_message.content}\n ---\n ## USER_REQUEST: \n {msgs[-1]["content"]}"
        msgs[-1]["content"] = updated_user_request

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
                    choice.append_content(delta.content)
                    content += delta.content

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
            custom_content=agent_message.custom_content
        )
