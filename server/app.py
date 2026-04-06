import sys
sys.path.insert(0, "/app")

from openenv.core.env_server import create_fastapi_app
from server.environment import PEDealScreeningEnv

app = create_fastapi_app(PEDealScreeningEnv)
