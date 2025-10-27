import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import MASCoordinator
from task.logging_config import setup_logging, get_logger

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
UMS_AGENT_ENDPOINT = os.getenv('UMS_AGENT_ENDPOINT', "http://localhost:8042")
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

setup_logging(log_level=LOG_LEVEL)
logger = get_logger(__name__)


class MASCoordinatorApplication(ChatCompletion):

    async def chat_completion(self, request: Request, response: Response) -> None:
        conversation_id = request.headers.get('x-conversation-id', 'unknown')
        logger.info(f"Received chat completion request [conversation_id={conversation_id}]")
        logger.debug(f"Request details: {len(request.messages)} messages")

        with response.create_single_choice() as choice:
            logger.debug(f"Created response choice [conversation_id={conversation_id}]")

            await MASCoordinator(
                endpoint=DIAL_ENDPOINT,
                deployment_name=DEPLOYMENT_NAME,
                ums_agent_endpoint=UMS_AGENT_ENDPOINT
            ).handle_request(
                choice=choice,
                request=request,
            )

            logger.info(f"Successfully completed chat request [conversation_id={conversation_id}]")


logger.info("Creating DIAL application")
app: DIALApp = DIALApp()
agent_app = MASCoordinatorApplication()
app.add_chat_completion(deployment_name="mas-coordinator", impl=agent_app)
logger.info("DIAL application initialized successfully")

if __name__ == "__main__":
    import sys

    if 'pydevd' in sys.modules:
        logger.info("Running in debug mode")
        config = uvicorn.Config(app, port=8055, host="0.0.0.0", log_level="info")
        server = uvicorn.Server(config)
        import asyncio
        asyncio.run(server.serve())
    else:
        logger.info("Starting uvicorn server on 0.0.0.0:8055")
        uvicorn.run(app, port=8055, host="0.0.0.0", log_level="info")
