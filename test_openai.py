from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)

key = os.getenv("OPENAI_API_KEY")
print("KEY HEAD:", repr(key[:20] if key else None))
print("KEY TAIL:", repr(key[-8:] if key else None))

client = OpenAI(api_key=key)

try:
    models = client.models.list()
    print("API OK")
except Exception as e:
    print("API ERROR:")
    print(e)