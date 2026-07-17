"""Template e montagem do prompt de classificação de uso do texto-base."""

_PROMPT = """Você é um avaliador de itens de prova. Analise se o TEXTO-BASE abaixo \
é efetivamente necessário para responder à questão, ou se a questão poderia ser \
respondida sem ele (texto-base decorativo/dispensável).

Avalie separadamente se cada parte da questão depende do texto-base:
- COMANDO: o enunciado faz referência a informações que estão no texto-base?
- DISTRATORES/GABARITO: as alternativas (incluindo a correta) fazem sentido, \
ou só podem ser distinguidas entre si, com base em algo dito no texto-base?
{instrucao_calculadora}
TEXTO-BASE:
{base_text}

COMANDO (enunciado da questão):
{comando}

ALTERNATIVAS:
{alternativas}

GABARITO (alternativa correta): {gabarito}

Responda APENAS com um JSON no formato exato abaixo, sem texto adicional:
{{
  "comando_depende_do_texto": true/false,
  "alternativas_dependem_do_texto": true/false,
  "gabarito_depende_do_texto": true/false,
  "texto_base_necessario": true/false,
  "justificativa": "explicação breve (1-2 frases)"
}}"""

INSTRUCAO_CALCULADORA = """
Esta questão é de matemática. Antes de decidir se uma alternativa (inclusive o \
gabarito) pode ser distinguida das demais sem o texto-base, use a ferramenta \
`resolver_matematica` para conferir qualquer conta, derivada, integral, equação \
ou simplificação necessária em vez de calcular de cabeça.
"""


def montar_prompt(base_text, stem, options, correct_option, statements, assertions, usar_calculadora):
    partes_comando = [stem, *(statements or []), *(assertions or [])]
    comando = "\n".join(
        str(p) for p in partes_comando if p and str(p).lower() != "nan"
    )

    alternativas = "\n".join(
        f"{letra.upper()}) {texto}"
        for letra, texto in options.items()
        if texto and str(texto).lower() != "nan"
    )

    return _PROMPT.format(
        instrucao_calculadora=INSTRUCAO_CALCULADORA if usar_calculadora else "",
        base_text=base_text,
        comando=comando,
        alternativas=alternativas,
        gabarito=correct_option or "não informado",
    )
