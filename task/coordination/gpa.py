from copy import deepcopy
from typing import Optional, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            task_description: str
    ) -> Message:
        api_key = request.api_key
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
        )

        chunks = await client.chat.completions.create(
            stream=True,
            messages=self.__prepare_gpa_messages(request, task_description),
            deployment_name="general-purpose-agent",
            extra_headers={
                'x-conversation-id': request.headers.get('x-conversation-id'),
            }
        )

        content = ''
        result_custom_content: CustomContent = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    stage.append_content(delta.content)
                    content += delta.content
                if cc := delta.custom_content:
                    print(cc)
                    if cc.attachments:
                        result_custom_content.attachments.extend(cc.attachments)

                    if cc.state:
                        result_custom_content.state = cc.state

                    cc_dict = cc.dict(exclude_none=True)
                    if stages := cc_dict.get("stages"):
                        for stg in stages:
                            idx = stg["index"]
                            if opened_stg := stages_map.get(idx):
                                if stg_content := stg.get("content"):
                                    opened_stg.append_content(stg_content)
                                elif stg_attachments := stg.get("attachments"):
                                    for stg_attachment in stg_attachments:
                                        opened_stg.add_attachment(Attachment(**stg_attachment))
                                elif stg.get("status") and stg.get("status") == 'completed':
                                    StageProcessor.close_stage_safely(stages_map[idx])
                            else:
                                stages_map[idx] = StageProcessor.open_stage(choice, stg.get("name"))

        for stg in stages_map.values():
            StageProcessor.close_stage_safely(stg)

        for attachment in result_custom_content.attachments:
            choice.add_attachment(
                Attachment(**attachment.dict(exclude_none=True))
            )

        choice.set_state(
            {
                _IS_GPA: True,
                _GPA_MESSAGES: result_custom_content.state,
            }
        )

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
        )

    def __prepare_gpa_messages(self, request: Request, task_description: Optional[str]) -> list[dict[str, Any]]:
        res_messages = []

        for idx in range(len(request.messages)):
            msg = request.messages[idx]
            if msg.role == Role.ASSISTANT:
                if msg.custom_content and msg.custom_content.state:
                    msg_state = msg.custom_content.state
                    if msg_state.get(_IS_GPA):
                        # 1. add user request (user message is always before assistant message)
                        res_messages.append(request.messages[idx - 1].dict(exclude_none=True))
                        # 2. Copy assistant message
                        copied_msg = deepcopy(msg)
                        copied_msg.custom_content.state = msg_state.get(_GPA_MESSAGES)
                        res_messages.append(copied_msg.dict(exclude_none=True))

        last_user_msg = request.messages[-1]
        usr_msg: dict[str, Any] = {
            "role": Role.USER,
            "content": task_description,
        }
        if last_user_msg.custom_content:
            usr_msg["custom_content"] = last_user_msg.custom_content.dict(exclude_none=True)

        res_messages.append(usr_msg)

        return res_messages
