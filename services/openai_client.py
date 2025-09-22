import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class MissingAPIKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY not set. Add it to your environment or .env file."
        )
    # The OpenAI SDK reads the env var internally too; setting ensures clarity.
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


