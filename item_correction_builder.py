"""Junta diagnóstico de código + LLM por question_id e formata o prompt de correção."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

import item_diagnostic_agent as diagnostic_agent
from item_quality_analyzer import (
    RespostaJSONInvalida,
    _campo,
    _montar_enunciado_completo,
    _texto,
    criar_modelo,
    extrair_json_resposta,
)

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system_correction_template.txt"
HUMAN_PROMPT_PATH = ROOT / "prompts" / "human_correction_template.txt"
MODULAR_TEMPLATE_PATH = ROOT / "prompts" / "prompt_correction_template_modularizado.txt"

ITENS_REFINO_PATH = ROOT / "data" / "itens_para_refino_top100.csv"
MARITACA_PATH = ROOT / "data" / "analises_maritaca.csv"
REJECTED_PATH = ROOT / "data" / "rejected_questions.csv"

# critério -> nome do bloco de diagnóstico LLM no template modular que o ilustra
# (ver CRITERIO_PARA_CAMPOS_LLM em item_diagnostic_agent, mesma correspondência)
CRITERIO_PARA_BLOCO_DIAG_LLM: dict[int, str] = {
    1: "diag_llm_corretude",
    3: "diag_llm_corretude",
    4: "diag_llm_corretude",
    5: "diag_llm_paralelismo",
    6: "diag_llm_convergencia",
    7: "diag_llm_plausibilidade",
    9: "diag_llm_convergencia",
    10: "diag_llm_clareza",
    11: "diag_llm_ideia_central",
    12: "diag_llm_clareza",
    15: "diag_llm_coerencia",
    16: "diag_llm_coerencia",
    17: "diag_llm_bloom",
}

# placeholder do human_correction_template -> coluna de analises_maritaca.csv
CAMPOS_LLM = {
    "llm_alternativa_correta_observada": "resolucao_independente.alternativa_correta_observada",
    "llm_situacao_gabarito": "resolucao_independente.situacao_do_gabarito",
    "llm_bloom_nivel_observado": "bloom.nivel_observado",
    "llm_bloom_alinhamento": "bloom.alinhamento",
    "llm_bloom_evidencia": "bloom.evidencia",
    "llm_paralelismo_status": "criterios.paralelismo.status",
    "llm_paralelismo_evidencia": "criterios.paralelismo.evidencia",
    "llm_paralelismo_justificativa": "criterios.paralelismo.justificativa",
    "llm_coerencia_status": "criterios.coerencia_conteudo.status",
    "llm_coerencia_evidencia": "criterios.coerencia_conteudo.evidencia",
    "llm_coerencia_justificativa": "criterios.coerencia_conteudo.justificativa",
    "llm_corretude_status": "criterios.corretude_unicidade_gabarito.status",
    "llm_corretude_evidencia": "criterios.corretude_unicidade_gabarito.evidencia",
    "llm_corretude_justificativa": "criterios.corretude_unicidade_gabarito.justificativa",
    "llm_convergencia_status": "criterios.convergencia_pistas.status",
    "llm_convergencia_alternativa_favorecida": "criterios.convergencia_pistas.alternativa_favorecida",
    "llm_convergencia_evidencia": "criterios.convergencia_pistas.evidencia",
    "llm_plausibilidade_status_geral": "criterios.plausibilidade_distratores.status_geral",
    "llm_plausibilidade_sugestao_correcao": "criterios.plausibilidade_distratores.sugestao_correcao",
    "llm_ideia_central_status": "criterios.ideia_central_enunciado.status",
    "llm_ideia_central_evidencia": "criterios.ideia_central_enunciado.evidencia",
    "llm_clareza_status": "criterios.clareza_linguistica.status",
    "llm_clareza_evidencia": "criterios.clareza_linguistica.evidencia",
    "llm_clareza_justificativa": "criterios.clareza_linguistica.justificativa",
}

# critérios fora dos 19 (camada subjetiva) — viram só contexto no template, não são corrigidos
CAMPOS_SUBJETIVOS_EXTRA = [
    ("Opinião/subjetividade",
     "criterios.opiniao_subjetividade.status",
     "criterios.opiniao_subjetividade.local_do_problema",
     "criterios.opiniao_subjetividade.justificativa"),
    ("Adequação ao nível de ensino",
     "criterios.adequacao_nivel_ensino.status",
     None,
     "criterios.adequacao_nivel_ensino.justificativa"),
    ("Termos absolutos/generalizações",
     "criterios.termos_absolutos_generalizacoes.status",
     "criterios.termos_absolutos_generalizacoes.local_do_problema",
     "criterios.termos_absolutos_generalizacoes.justificativa"),
]


def carregar_itens_refino(caminho: Path = ITENS_REFINO_PATH) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    df["question_id"] = df["question_id"].astype(int)
    return df.set_index("question_id", drop=False)


def carregar_avaliacoes_llm(caminho: Path = MARITACA_PATH) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding="utf-8-sig").rename(columns={"id": "question_id"})
    df["question_id"] = df["question_id"].astype(int)
    return df.set_index("question_id", drop=False)


def calcular_letra_dominante(caminho: Path = REJECTED_PATH) -> str:
    """Letra de gabarito sobre-representada no conjunto completo (célula 4.1 do notebook)."""
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    return diagnostic_agent.calcular_letra_dominante(df)


def _valor_llm(linha_llm: Mapping[str, Any] | None, coluna: str | None) -> str:
    if linha_llm is None or coluna is None:
        return "NAO_DISPONIVEL"
    return _texto(linha_llm.get(coluna)) or "NAO_DISPONIVEL"


def _montar_observacoes_subjetivas_extra(linha_llm: Mapping[str, Any] | None) -> str:
    if linha_llm is None:
        return "Avaliação LLM indisponível para este item."
    blocos = []
    for rotulo, col_status, col_local, col_justificativa in CAMPOS_SUBJETIVOS_EXTRA:
        if _valor_llm(linha_llm, col_status) != "PROBLEMA":
            continue
        local = f" ({_valor_llm(linha_llm, col_local)})" if col_local else ""
        blocos.append(f"- {rotulo}{local}: {_valor_llm(linha_llm, col_justificativa)}")
    return "\n".join(blocos) if blocos else "Nenhuma observação subjetiva adicional registrada."


def formatar_mensagens_correcao(
    item: Mapping[str, Any],
    linha_llm: Mapping[str, Any] | None,
    letra_dominante: str,
) -> list[SystemMessage | HumanMessage]:
    """Preenche os templates de correção com uma linha de itens_para_refino_top100.csv
    e a linha correspondente (por question_id) de analises_maritaca.csv."""
    system_template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    human_template = HUMAN_PROMPT_PATH.read_text(encoding="utf-8")
    item_id = _texto(_campo(item, "question_id"))

    # o prompt de sistema contém chaves JSON de exemplo, então só substituímos {id} nele
    system_prompt = system_template.replace("{id}", item_id)

    campos_llm = {chave: _valor_llm(linha_llm, coluna) for chave, coluna in CAMPOS_LLM.items()}

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
        o_que_melhorar=_texto(_campo(item, "o_que_melhorar")) or "Nenhum problema objetivo listado.",
        letra_dominante_conjunto=letra_dominante,
        observacoes_subjetivas_extra=_montar_observacoes_subjetivas_extra(linha_llm),
        **campos_llm,
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def montar_prompt_por_question_id(
    question_id: int,
    itens: pd.DataFrame | None = None,
    avaliacoes_llm: pd.DataFrame | None = None,
    letra_dominante: str | None = None,
) -> list[SystemMessage | HumanMessage]:
    """Ponto de entrada único: dado um question_id, junta os dois CSVs e monta as mensagens."""
    itens = itens if itens is not None else carregar_itens_refino()
    avaliacoes_llm = avaliacoes_llm if avaliacoes_llm is not None else carregar_avaliacoes_llm()
    letra_dominante = letra_dominante or calcular_letra_dominante()

    if question_id not in itens.index:
        raise KeyError(f"question_id {question_id} não está em {ITENS_REFINO_PATH.name}")
    item = itens.loc[question_id]
    linha_llm = avaliacoes_llm.loc[question_id] if question_id in avaliacoes_llm.index else None
    return formatar_mensagens_correcao(item, linha_llm, letra_dominante)


def corrigir_item(
    question_id: int,
    llm,
    itens: pd.DataFrame | None = None,
    avaliacoes_llm: pd.DataFrame | None = None,
    letra_dominante: str | None = None,
) -> dict[str, Any]:
    """Executa a correção de um item e aceita o resultado somente quando o JSON é válido."""
    mensagens = montar_prompt_por_question_id(question_id, itens, avaliacoes_llm, letra_dominante)
    resposta = llm.invoke(
        mensagens,
        config={
            "run_name": "corrigir_item_wapla",
            "tags": ["wapla", "correcao-item"],
            "metadata": {"question_id": str(question_id)},
        },
    )
    try:
        resultado = extrair_json_resposta(resposta.content)
    except ValueError as erro:
        raise RespostaJSONInvalida(str(erro), resposta.content) from erro

    resultado["id"] = str(question_id)
    return resultado


# ---------------------------------------------------------------------------
# Prompt modular (prompts/prompt_correction_template_modularizado.txt): monta um
# prompt de correção único por item, incluindo só os blocos de critério (e de
# diagnóstico LLM) relevantes para os problemas detectados NESTE item — em vez de
# sempre despejar as 19 explicações de critério e todas as seções de diagnóstico
# pedagógico, como faz `formatar_mensagens_correcao`. Usado pelo loop iterativo de
# correção (loop_correcao_avaliacao.py), onde o item muda a cada rodada e o
# diagnóstico precisa ser recalculado a cada vez.
# ---------------------------------------------------------------------------
_BLOCO_RE = re.compile(r"### BLOCO: (\w+) ###\n(.*?)\n### FIM_BLOCO: \1 ###", re.DOTALL)


@lru_cache(maxsize=1)
def _carregar_blocos_modulares() -> dict[str, str]:
    conteudo = MODULAR_TEMPLATE_PATH.read_text(encoding="utf-8")
    return {nome: texto.strip() for nome, texto in _BLOCO_RE.findall(conteudo)}


def montar_mensagens_correcao_modular(
    item: Mapping[str, Any],
    criterios_ativos: list[int],
    o_que_melhorar: str,
    linha_llm: Mapping[str, Any] | None = None,
    letra_dominante: str | None = None,
) -> list[SystemMessage | HumanMessage]:
    """Monta o par (system, human) do template modular, incluindo somente os
    critérios em `criterios_ativos` (números 1-19) — tipicamente
    `item_diagnostic_agent.criterios_flagrados(flags, linha_llm)` para este item."""
    blocos = _carregar_blocos_modulares()
    criterios_ativos = sorted(set(criterios_ativos))
    item_id = _texto(_campo(item, "question_id"))

    partes_sistema = [blocos["contexto"], blocos["procedimento"], blocos["criterios_header"]]
    partes_sistema += [blocos[f"criterio_{n}"] for n in criterios_ativos if f"criterio_{n}" in blocos]
    partes_sistema += [blocos["criterios_footer"], blocos["regras"]]
    partes_sistema.append(blocos["formato_saida"].replace("{id}", item_id))
    system_prompt = "\n\n".join(partes_sistema)

    partes_humanas = [
        blocos["human_intro"],
        blocos["item_original"].format(
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
        ),
        blocos["diagnostico_objetivo"].format(o_que_melhorar=o_que_melhorar or "Nenhum problema objetivo listado."),
    ]

    if 19 in criterios_ativos:
        partes_humanas.append(
            blocos["diagnostico_letra_dominante"].format(
                letra_dominante_conjunto=letra_dominante or "NAO_DISPONIVEL"
            )
        )

    blocos_llm_incluidos: set[str] = set()
    campos_llm = {chave: _valor_llm(linha_llm, coluna) for chave, coluna in CAMPOS_LLM.items()}
    for criterio in criterios_ativos:
        nome_bloco = CRITERIO_PARA_BLOCO_DIAG_LLM.get(criterio)
        if not nome_bloco or nome_bloco in blocos_llm_incluidos:
            continue
        if not blocos_llm_incluidos:
            partes_humanas.append(blocos["diagnostico_pedagogico_intro"])
        blocos_llm_incluidos.add(nome_bloco)
        partes_humanas.append(blocos[nome_bloco].format(**campos_llm))

    partes_humanas.append(
        blocos["observacoes_fora_escopo"].format(
            observacoes_subjetivas_extra=_montar_observacoes_subjetivas_extra(linha_llm)
        )
    )
    human_prompt = "\n\n".join(partes_humanas)

    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def corrigir_item_modular(
    item: Mapping[str, Any],
    criterios_ativos: list[int],
    o_que_melhorar: str,
    llm,
    linha_llm: Mapping[str, Any] | None = None,
    letra_dominante: str | None = None,
) -> dict[str, Any]:
    """Como `corrigir_item`, mas a partir de um item e diagnóstico já em memória
    (não faz lookup em CSV por question_id) — necessário no loop iterativo, onde o
    item corrigido de uma rodada vira o item de entrada da próxima."""
    item_id = _texto(_campo(item, "question_id"))
    mensagens = montar_mensagens_correcao_modular(item, criterios_ativos, o_que_melhorar, linha_llm, letra_dominante)
    resposta = llm.invoke(
        mensagens,
        config={
            "run_name": "corrigir_item_wapla_modular",
            "tags": ["wapla", "correcao-item", "modular"],
            "metadata": {"question_id": item_id, "criterios_ativos": str(criterios_ativos)},
        },
    )
    try:
        resultado = extrair_json_resposta(resposta.content)
    except ValueError as erro:
        raise RespostaJSONInvalida(str(erro), resposta.content) from erro

    resultado["id"] = item_id
    return resultado


if __name__ == "__main__":
    import sys

    itens = carregar_itens_refino()
    qid = int(sys.argv[1]) if len(sys.argv) > 1 else int(itens["question_id"].iloc[0])

    mensagens = montar_prompt_por_question_id(qid, itens=itens)
    for mensagem in mensagens:
        print(f"=== {mensagem.type.upper()} ===")
        print(mensagem.content)
        print()