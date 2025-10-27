import json
from typing import Optional

import httpx
from aidial_client.types.chat.legacy.chat_completion import CustomContent
from aidial_sdk.chat_completion import Role, Request, Message, Stage
from pydantic import StrictStr

from task.constants import UMS_CONVERSATION_ID


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
            self,
            stage: Stage,
            request: Request,
            task_description: Optional[str],
            ums_conversation_id: Optional[str]
    ) -> Message:
        if not ums_conversation_id:
            ums_conversation_id = self.__get_ums_conversation_id(request)
        custom_content = None
        if not ums_conversation_id:
            ums_conversation_id = await self.__create_ums_conversation()
            custom_content = CustomContent(
                state={UMS_CONVERSATION_ID: ums_conversation_id},
            )
            stage.append_content(f"_Created new UMS conversation: {ums_conversation_id}_\n\n")

        content = await self.__call_ums_agent(
            conversation_id=ums_conversation_id,
            user_message=task_description,
            stage=stage
        )

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
            custom_content=custom_content,
        )

    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""
        for msg in request.messages:
            if msg.custom_content and msg.custom_content.state:
                ums_conversation_id = msg.custom_content.state.get(UMS_CONVERSATION_ID)
                if ums_conversation_id:
                    return ums_conversation_id
        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations",
                json={"title": "UMS Agent Conversation"},
                timeout=30.0
            )
            response.raise_for_status()
            conversation_data = response.json()
            return conversation_data['id']

    async def __call_ums_agent(
            self,
            conversation_id: str,
            user_message: str,
            stage: Stage
    ) -> str:
        """Call UMS agent and stream the response"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                json={
                    "message": {
                        "role": "user",
                        "content": user_message
                    },
                    "stream": True
                },
                timeout=60.0
            )
            response.raise_for_status()

            content = ''
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data_str = line[6:]

                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)

                        if 'conversation_id' in data:
                            continue

                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if delta_content := delta.get('content'):
                                stage.append_content(delta_content)
                                content += delta_content
                    except json.JSONDecodeError:
                        continue

            return content
