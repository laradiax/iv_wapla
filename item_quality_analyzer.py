"""Formata os prompts e analisa um item educacional com a Maritaca via LangChain."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system_template.txt"
HUMAN_PROMPT_PATH = ROOT / "prompts" / "human_template.txt"

load_dotenv(ROOT / ".env")


def _texto(valor: Any) -> str:
    if valor is None:
        return ""
    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass
    return str(valor).strip()


def _campo(item: Mapping[str, Any], nome: str) -> Any:
    try:
        return item[nome]
    except (KeyError, TypeError):
        return ""


def _montar_enunciado_completo(item: Mapping[str, Any]) -> str:
    partes = [_texto(_campo(item, "stem"))]
    for coluna, rotulo in (
        ("statement_i", "Afirmação I"),
        ("statement_ii", "Afirmação II"),
        ("statement_iii", "Afirmação III"),
        ("statement_iv", "Afirmação IV"),
        ("assertion_i", "Asserção I"),
        ("assertion_ii", "Asserção II"),
    ):
        valor = _texto(_campo(item, coluna))
        if valor:
            partes.append(f"{rotulo}: {valor}")
    return "\n".join(parte for parte in partes if parte)


def formatar_mensagens(item: Mapping[str, Any]) -> list[SystemMessage | HumanMessage]:
    """Preenche os templates do projeto com uma linha do CSV."""
    system_template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    human_template = HUMAN_PROMPT_PATH.read_text(encoding="utf-8")
    item_id = _texto(_campo(item, "question_id"))

    # Remove vírgulas finais do exemplo JSON antes de enviá-lo ao modelo.
    system_template = re.sub(r",(\s*[}\]])", r"\1", system_template)

    # O prompt de sistema contém chaves JSON, então não usamos str.format nele.
    system_prompt = system_template.replace("{id}", item_id)
    human_prompt = human_template.format(
        id=item_id,
        tipo_questao=_texto(_campo(item, "question_type")),
        disciplina=_texto(_campo(item, "discipline")),
        bloom_declarado=_texto(_campo(item, "taxonomy_level")),
        texto_base=_texto(_campo(item, "base_text")),
        enunciado=_montar_enunciado_completo(item),
        alternativa_a=_texto(_campo(item, "option_a")),
        alternativa_b=_texto(_campo(item, "option_b")),
        alternativa_c=_texto(_campo(item, "option_c")),
        alternativa_d=_texto(_campo(item, "option_d")),
        alternativa_e=_texto(_campo(item, "option_e")),
        gabarito=_texto(_campo(item, "correct_option")),
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def criar_modelo(
    model: str | None = None,
    timeout: float = 180,
    max_retries: int = 2,
) -> ChatOpenAI:
    """Cria um ChatOpenAI configurado para a API compatível da Maritaca."""
    api_key = os.getenv("MARITACA_API_KEY")
    if not api_key:
        raise RuntimeError("MARITACA_API_KEY não configurada no arquivo .env")
    return ChatOpenAI(
        model=model or os.getenv("MARITACA_MODEL", "sabia-4"),
        api_key=api_key,
        base_url=os.getenv("MARITACA_BASE_URL", "https://chat.maritaca.ai/api"),
        temperature=0,
        timeout=timeout,
        max_retries=max_retries,
    )


def extrair_json_resposta(conteudo: Any) -> dict[str, Any]:
    """Extrai e decodifica o primeiro objeto JSON retornado pelo modelo."""
    if isinstance(conteudo, list):
        conteudo = "".join(
            bloco.get("text", "") if isinstance(bloco, dict) else str(bloco)
            for bloco in conteudo
        )
    if not isinstance(conteudo, str):
        raise ValueError(f"Conteúdo inesperado na resposta: {type(conteudo).__name__}")
    limpo = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", conteudo, flags=re.I)
    inicio = limpo.find("{")
    if inicio < 0:
        raise ValueError("A resposta do modelo não contém um objeto JSON")
    try:
        resultado, _ = json.JSONDecoder().raw_decode(limpo[inicio:])
    except json.JSONDecodeError as erro:
        raise ValueError(f"A resposta contém JSON inválido: {erro}") from erro
    if not isinstance(resultado, dict):
        raise ValueError("A resposta JSON não é um objeto")
    return resultado


class RespostaJSONInvalida(ValueError):
    """Erro que preserva a saída original quando a LLM não retorna JSON válido."""

    def __init__(self, mensagem: str, saida_bruta: Any):
        super().__init__(mensagem)
        self.saida_bruta = saida_bruta


def analisar_item(item: Mapping[str, Any], llm: ChatOpenAI) -> dict[str, Any]:
    """Executa uma análise e aceita o resultado somente quando o JSON é válido."""
    item_id = _texto(_campo(item, "question_id"))
    resposta = llm.invoke(
        formatar_mensagens(item),
        config={
            "run_name": "avaliar_item_wapla",
            "tags": ["wapla", "avaliacao-pedagogica"],
            "metadata": {
                "question_id": item_id,
                "discipline": _texto(_campo(item, "discipline")),
            },
        },
    )

    try:
        resultado = extrair_json_resposta(resposta.content)
    except ValueError as erro:
        raise RespostaJSONInvalida(str(erro), resposta.content) from erro

    resultado["id"] = item_id
    return resultado

