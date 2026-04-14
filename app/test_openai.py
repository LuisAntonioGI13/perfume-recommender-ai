from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "system", "content": "Eres un experto en perfumes."},
        {"role": "user", "content": "Recomiéndame un perfume fresco para uso diario."}
    ]
)

print(response.choices[0].message.content)