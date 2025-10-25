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
        #TODO:
        # 1. Create single choice with context manager
        # 2. Create MASCoordinator and handle request
        raise NotImplementedError()



#TODO:
# 1. Create DIALApp
# 2. Create MASCoordinatorApplication
# 3. Add to created DIALApp chat_completion with:
#       - deployment_name="mas-coordinator"
#       - impl=agent_app
# 4. Run it with uvicorn: `uvicorn.run({CREATED_DIAL_APP}, port=8055, host="0.0.0.0")`

