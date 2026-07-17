"""Cliente HTTP para a API de chat completions da Maritaca AI."""

import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

MARITACA_API_KEY = os.getenv("MARITACA_API_KEY")
MARITACA_BASE_URL = os.getenv("MARITACA_BASE_URL", "https://chat.maritaca.ai/api")
MARITACA_MODEL = os.getenv("MARITACA_MODEL", "sabia-4")


def chamar_chat_completions(messages, model=None, timeout=60, tools=None):
    """Chama o endpoint /chat/completions e devolve a mensagem de resposta."""
    payload = {
        "model": model or MARITACA_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resposta = requests.post(
        f"{MARITACA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {MARITACA_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resposta.raise_for_status()
    return resposta.json()["choices"][0]["message"]


def extrair_json(conteudo):
    """Extrai o primeiro objeto JSON de uma resposta em texto do modelo."""
    match = re.search(r"\{.*\}", conteudo, re.DOTALL)
    if not match:
        raise ValueError(f"Resposta do modelo não contém JSON válido:\n{conteudo}")
    return json.loads(match.group(0))
