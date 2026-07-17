"""Classifica (LLM) se o `base_text` de uma questão é
realmente necessário para respondê-la (comando, distratores e gabarito)."""

import json

from llm_client import MARITACA_API_KEY, chamar_chat_completions, extrair_json
from math_tool import MATH_TOOL, resolver_matematica
from prompts import montar_prompt

_MAX_TOOL_ITERACOES = 8


def classificar_uso_texto_base(
    base_text,
    stem,
    options,
    correct_option=None,
    statements=None,
    assertions=None,
    question_id=None,
    model=None,
    timeout=60,
):
    """Chama a LLM para classificar se `base_text` é usado/necessário.

    Args:
        base_text: texto-base da questão.
        stem: enunciado/comando principal.
        options: dict com as alternativas, ex. {"a": "...", "b": "...", ...}.
        correct_option: letra da alternativa correta (gabarito), ex. "A".
        statements: lista opcional de afirmações (statement_i..iv) que fazem
            parte do comando em questões desse tipo.
        assertions: lista opcional de asserções (assertion_i/ii).
        question_id: identificador opcional, incluído no retorno para rastreio.
        model: sobrescreve o modelo definido em MARITACA_MODEL.
        timeout: timeout da requisição HTTP em segundos.

    Returns:
        dict com as chaves comando_depende_do_texto, alternativas_dependem_do_texto,
        gabarito_depende_do_texto, texto_base_necessario, justificativa e question_id.
    """
    if not MARITACA_API_KEY:
        raise RuntimeError(
            "MARITACA_API_KEY não configurada. Defina-a no arquivo .env"
        )

    prompt = montar_prompt(
        base_text, stem, options, correct_option, statements, assertions,
        usar_calculadora=False,
    )
    mensagem = chamar_chat_completions(
        [{"role": "user", "content": prompt}], model, timeout,
    )
    resultado = extrair_json(mensagem["content"])
    resultado["question_id"] = question_id
    return resultado


def classificar_uso_texto_base_com_calculadora(
    base_text,
    stem,
    options,
    correct_option=None,
    statements=None,
    assertions=None,
    question_id=None,
    model=None,
    timeout=60,
):
    """Igual a `classificar_uso_texto_base`, mas dá ao modelo uma ferramenta
    (`resolver_matematica`, via sympy) para conferir contas em vez de calculá-las
    mentalmente. Pensada para questões de matemática, onde LLMs erram mais.

    Retorna o mesmo formato de `classificar_uso_texto_base`, mais a chave
    `chamadas_calculadora` com a quantidade de vezes que a ferramenta foi usada.
    """
    if not MARITACA_API_KEY:
        raise RuntimeError(
            "MARITACA_API_KEY não configurada. Defina-a no arquivo .env"
        )

    prompt = montar_prompt(
        base_text, stem, options, correct_option, statements, assertions,
        usar_calculadora=True,
    )
    messages = [{"role": "user", "content": prompt}]
    chamadas_calculadora = 0

    for _ in range(_MAX_TOOL_ITERACOES):
        mensagem = chamar_chat_completions(
            messages, model, timeout, tools=[MATH_TOOL],
        )
        tool_calls = mensagem.get("tool_calls")
        if not tool_calls:
            resultado = extrair_json(mensagem["content"])
            resultado["question_id"] = question_id
            resultado["chamadas_calculadora"] = chamadas_calculadora
            return resultado

        messages.append(mensagem)
        for chamada in tool_calls:
            argumentos = json.loads(chamada["function"]["arguments"])
            saida = resolver_matematica(argumentos.get("codigo", ""))
            chamadas_calculadora += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": chamada["id"],
                    "content": saida,
                }
            )

    raise RuntimeError(
        f"Excedeu {_MAX_TOOL_ITERACOES} chamadas de ferramenta sem resposta final "
        f"(question_id={question_id})"
    )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/rejected_questions.csv")
    row = df.iloc[0]

    resultado = classificar_uso_texto_base(
        base_text=row["base_text"],
        stem=row["stem"],
        options={
            "a": row["option_a"],
            "b": row["option_b"],
            "c": row["option_c"],
            "d": row["option_d"],
            "e": row["option_e"],
        },
        correct_option=row["correct_option"],
        statements=[
            row.get("statement_i"),
            row.get("statement_ii"),
            row.get("statement_iii"),
            row.get("statement_iv"),
        ],
        assertions=[row.get("assertion_i"), row.get("assertion_ii")],
        question_id=row["question_id"],
    )
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
