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
        #TODO:
        # 1. Create AsyncDial client
        # 2. Open stage for Coordination Request (StageProcessor will help with that)
        # 3. Prepare coordination request
        # 4. Add to the stage generated coordination request and close the stage
        # 5. Handle coordination request (don't forget that all the content that will write called agent need to provide to stage)
        # 6. Generate final response based on the message from called agent
        raise NotImplementedError()

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        #TODO:
        # 1. Make call to LLM with prepared messages and COORDINATION_REQUEST_SYSTEM_PROMPT. For GPT model we can use
        #    `response_format` https://platform.openai.com/docs/guides/structured-outputs?example=structured-data and
        #    response will be returned in JSON format. The `response_format` parameter must be provided as extra_body dict
        #    {response_format": {"type": "json_schema","json_schema": {"name": "response","schema": CoordinationRequest.model_json_schema()}}}
        #    provide api_version="2024-08-01-preview" (required)
        # 2. Get content from response -> choice -> message -> content
        # 3. Load as dict
        # 4. Create CoordinationRequest from result, since CoordinationRequest is pydentic model, you can use `model_validate` method
        raise NotImplementedError()

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        #TODO:
        # 1. Create array with messages, first message is system prompt and it is dict
        # 2. Iterate through messages from request and:
        #       - if user message that it has custom content and then add dict with user message and content (custom_content should be skipped)
        #       - otherwise append it as dict with excluded none fields (use `dict` method, despite it is deprecated since
        #         DIAL is using pydentic.v1)
        raise NotImplementedError()

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        #TODO:
        # Make appropriate coordination requests to to proper agents and return the result
        raise NotImplementedError()

    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        #TODO:
        # 1. Prepare messages with FINAL_RESPONSE_SYSTEM_PROMPT
        # 2. Make augmentation of retrieved agent response (as context) with user request (as user request)
        # 3. Update last message content with augmented prompt
        # 4. Call LLM with streaming
        # 5. Stream final response to choice
        raise NotImplementedError()
