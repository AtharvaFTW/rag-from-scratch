from ollama import Client
from dotenv import load_dotenv
import os
load_dotenv()

HOST = os.environ.get("OLLAMA_NGROK_TUNNEL")

client = Client(host = HOST,  headers={'ngrok-skip-browser-warning': 'true'})

if __name__ == "__main__":
    response = client.chat(model='llama3.1:8b', messages=[
    {'role': 'user', 'content': 'Write a song like the weeknd (able tesfaye)'}
    ])
    print(response['message']['content'])