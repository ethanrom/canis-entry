import os
import openai

def set_openai_api_key(api_key):
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key


DEFAULT_TEMPERATURE = 0


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    OPENAI_API_KEY = "sk-abcd"

