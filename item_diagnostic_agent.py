"""Agente de diagnóstico de itens (avaliação por código, reexecutável).

Reimplementa como funções importáveis — chamáveis para um item isolado ou para um
DataFrame inteiro — as análises que hoje só existiam espalhadas em notebooks e
scripts:

- `diagnostico_e_selecao_itens.ipynb` (fonte principal, já uma consolidação):
  células 2-3 (normalização de texto), 9/11/13/15/20 (eixo Formulação e Estrutura,
  `fe_*`), 23/25-26 (eixo Alinhamento Pedagógico, `ap_*`), 29 (eixo Aplicação,
  `apl_*`, viés de posição do gabarito), 34 (camada subjetiva via Maritaca), 37-46
  (catálogo de melhorias, score de priorização, elegibilidade e seleção para
  refino).
- `revisao_questoes.ipynb`: versão anterior das mesmas checagens estruturais — já
  superada pela implementação do notebook acima (mesma lógica, regex mais precisos),
  não é portada de novo.
- `base_text_classifier.py`: classificação (LLM) de uso do texto-base, usada aqui
  com o mesmo fallback heurístico (`textbase_dispensavel_proxy`) da célula 16.

Uso típico:
    from item_diagnostic_agent import diagnosticar_item, diagnosticar_dataframe

    # um item isolado (ex.: revalidar um item já corrigido, dentro de um loop)
    resultado = diagnosticar_item(item_dict, letra_dominante="A")

    # o dataset inteiro, reproduzindo o pipeline do notebook, com progresso salvo em disco
    df = diagnosticar_dataframe()

Nota sobre `fe_gabarito_matematico_impossivel`: no notebook original essa checagem foi
feita item a item, à mão, com `sympy` (célula 18) — são fórmulas específicas para 10
`question_id` conhecidos do conjunto `rejected_questions.csv`. Isso não generaliza para
um item novo ou reescrito (não há como derivar automaticamente "qual é a fórmula certa
para este enunciado"). Por isso essa verificação só é aplicada aos `question_id`
conhecidos (`VERIFICACAO_MATEMATICA_CONHECIDA`) e apenas quando
`usar_verificacao_matematica_conhecida=True` (padrão ao diagnosticar o item original).
Para itens corrigidos, a correção factual do gabarito passa a depender da resolução
independente feita pela LLM (ver `item_quality_analyzer.analisar_item`).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REJECTED_PATH = DATA_DIR / "rejected_questions.csv"
TEXTBASE_CACHE_PATH = DATA_DIR / "text_base_classification_results.csv"
MARITACA_PATH = DATA_DIR / "analises_maritaca.csv"
VEREDITO_PATH = DATA_DIR / "veredito_por_questao.csv"
REFINO_PATH = DATA_DIR / "itens_para_refino_top100.csv"

# ---------------------------------------------------------------------------
# Constantes e helpers de normalização (portados de diagnostico_e_selecao_itens.ipynb)
# ---------------------------------------------------------------------------
OPCOES = ["option_a", "option_b", "option_c", "option_d", "option_e"]
LETRAS_VALIDAS = {"a", "b", "c", "d", "e"}
COLS_STATEMENT = ["statement_i", "statement_ii", "statement_iii", "statement_iv"]
COLS_ASSERTION = ["assertion_i", "assertion_ii"]
TIPOS_VALIDOS = {
    "Afirmação Incompleta", "Múltipla Escolha Simples",
    "Múltipla Escolha Complexa", "Asserção-Razão",
}

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "sem", "sobre", "entre", "e", "ou", "que", "se", "ao", "aos",
    "essa", "esse", "isso", "isto", "sua", "seu", "suas", "seus", "mais", "menos", "como", "quando",
    "onde", "qual", "quais", "ser", "sao", "esta", "estao", "foi", "foram", "deve", "devem",
    "correta", "corretas", "alternativa", "opcao", "questao", "contexto", "apresentado",
    "descrito", "considerando", "assinale", "identifique", "afirma", "afirmam", "apenas",
}

TERMOS_ABSOLUTOS = [
    "sempre", "nunca", "jamais", "apenas", "somente", "exclusivamente", "todos", "todas",
    "nenhum", "nenhuma", "impossivel", "impossibilidade", "dispensa", "dispensavel",
    "obrigatoriamente", "totalmente", "unicamente",
]

TERMOS_NEGATIVOS_ENUNCIADO = ["exceto", "incorreta", "incorreto", "falsa", "falso", "errado", "menos"]

COMANDOS_EXPLICITOS = ["assinale", "identifique", "indique", "selecione", "marque",
                       "corresponde", "determine", "calcule", "classifique", "avalie", "analise"]


def texto_valido(valor: Any) -> bool:
    return pd.notna(valor) and str(valor).strip() != "" and str(valor).strip().lower() != "nan"


def remover_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(texto)) if not unicodedata.combining(c))


def normalizar_lexico(texto: str) -> str:
    texto = remover_acentos(texto).lower()
    texto = re.sub(r"[^a-z0-9ivxlcdm]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_igualdade(texto: str) -> str:
    texto = remover_acentos(texto).lower()
    texto = texto.replace("“", '"').replace("”", '"').replace("’", "'")
    texto = re.sub(r"\s+", " ", texto).strip()
    return re.sub(r"[.;:,]+$", "", texto)


def tokens(texto: str) -> set:
    return {t for t in normalizar_lexico(texto).split() if len(t) > 2 and t not in STOPWORDS}


def contem_matematica(texto: str) -> bool:
    return bool(re.search(r"[$=+\-*/^{}\\]|\bfrac\b|\bsqrt\b", str(texto)))


def campos_presentes(row: Mapping[str, Any], campos: list) -> list:
    return [c for c in campos if texto_valido(row.get(c))]


def opcoes_validas(row: Mapping[str, Any]) -> dict:
    return {c[-1]: str(row[c]).strip() for c in OPCOES if texto_valido(row.get(c))}


def similaridade_textual(a: str, b: str) -> float:
    return SequenceMatcher(None, normalizar_lexico(a), normalizar_lexico(b)).ratio()


def parse_romanos(texto: str) -> tuple:
    encontrados = re.findall(r"\b(?:i|ii|iii|iv)\b", normalizar_lexico(texto))
    ordem = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    return tuple(sorted(set(encontrados), key=lambda x: ordem[x]))


def contem_termo(texto: str, termos: list) -> bool:
    tn = normalizar_lexico(texto)
    return any(re.search(rf"\b{re.escape(t)}\b", tn) for t in termos)


def overlap_lexico(texto_a: str, texto_b: str) -> float:
    ta, tb = tokens(texto_a), tokens(texto_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def normalizar_disciplina(texto: str) -> str:
    return remover_acentos(str(texto).strip().lower())


# ---------------------------------------------------------------------------
# Eixo FE — Formulação e Estrutura (diagnostico_e_selecao_itens.ipynb, células 9/11/13/15)
# ---------------------------------------------------------------------------
def fomulacao_estrutura_gabarito(row: Mapping[str, Any]) -> dict:
    """Integridade do gabarito, duplicatas e equivalências entre alternativas."""
    flags: dict[str, Any] = {}
    opcoes = opcoes_validas(row)
    gabarito = normalizar_lexico(row.get("correct_option", "")).lower()
    tipo = str(row.get("question_type", "")).strip()

    flags["fe_alternativas_ausentes"] = len(opcoes) < len(OPCOES)
    flags["fe_gabarito_invalido"] = gabarito not in LETRAS_VALIDAS
    flags["fe_gabarito_vazio"] = (gabarito in LETRAS_VALIDAS) and (gabarito not in opcoes)
    flags["fe_enunciado_ausente"] = not texto_valido(row.get("stem"))

    dup_alt = dup_gab = equiv_gab = similares = False
    letras = list(opcoes)
    norm = {l: normalizar_igualdade(t) for l, t in opcoes.items()}
    for i, li in enumerate(letras):
        for lj in letras[i + 1:]:
            textos = (opcoes[li], opcoes[lj])
            tem_math = any(contem_matematica(t) for t in textos)
            por_template = tipo in {"Múltipla Escolha Complexa", "Asserção-Razão"}
            if norm[li] and norm[li] == norm[lj]:
                if gabarito in {li, lj}:
                    dup_gab = True
                else:
                    dup_alt = True
            elif not tem_math and not por_template:
                sim = similaridade_textual(*textos)
                if sim >= 0.93 and min(len(t) for t in textos) >= 60:
                    if gabarito in {li, lj}:
                        equiv_gab = True
                    else:
                        similares = True

    flags["fe_alternativas_duplicadas"] = dup_alt
    flags["fe_gabarito_duplicado"] = dup_gab
    flags["fe_gabarito_equivale_distrator"] = equiv_gab
    flags["fe_alternativas_similares"] = similares
    return flags


def fomulacao_estrutura_paralelismo(row: Mapping[str, Any]) -> dict:
    """Viés de tamanho, extensão desigual, metalinguística e termos absolutos."""
    flags = {k: False for k in ["fe_vies_tamanho_maior", "fe_vies_tamanho_menor",
                                "fe_extensao_desigual", "fe_metalinguistica",
                                "fe_distratores_termos_absolutos"]}
    opcoes = opcoes_validas(row)
    gabarito = normalizar_lexico(row.get("correct_option", "")).lower()

    if gabarito in opcoes:
        len_correta = len(opcoes[gabarito])
        outras = [len(t) for l, t in opcoes.items() if l != gabarito]
        if outras:
            media = sum(outras) / len(outras)
            flags["fe_vies_tamanho_maior"] = len_correta > media * 1.3
            flags["fe_vies_tamanho_menor"] = len_correta < media * 0.55

    tamanhos = [len(t) for t in opcoes.values()]
    if len(tamanhos) >= 4 and min(tamanhos) > 0:
        flags["fe_extensao_desigual"] = (max(tamanhos) / min(tamanhos)) > 2.7

    for t in opcoes.values():
        if re.search(r"\b(todas|nenhuma)\b.*\b(anteriores|alternativas|opcoes|acima)\b", normalizar_lexico(t)):
            flags["fe_metalinguistica"] = True
            break

    if gabarito in opcoes:
        absolutas_erradas = [l for l, t in opcoes.items()
                             if l != gabarito and contem_termo(t, TERMOS_ABSOLUTOS)]
        correta_absoluta = contem_termo(opcoes[gabarito], TERMOS_ABSOLUTOS)
        flags["fe_distratores_termos_absolutos"] = (len(absolutas_erradas) >= 2 and not correta_absoluta)
    return flags


def fomulacao_estrutura_pistas(row: Mapping[str, Any]) -> dict:
    """Lacuna, V/F disfarçado, sequência numérica, enunciado negativo, comando, word-repeats."""
    flags: dict[str, Any] = {}
    opcoes = opcoes_validas(row)
    stem = str(row.get("stem", ""))
    stem_norm = normalizar_lexico(stem)
    tipo = str(row.get("question_type", "")).strip()

    flags["fe_lacuna_enunciado"] = ("__" in stem) or ("preencha a lacuna" in stem_norm)

    flags["fe_verdadeiro_falso"] = any(
        normalizar_lexico(t) in ("verdadeiro", "falso", "sim", "nao", "certo", "errado")
        for t in opcoes.values())

    valores, aplicavel = [], len(opcoes) >= 3
    if aplicavel:
        for t in opcoes.values():
            nums = re.findall(r"-?\d+(?:[.,]\d+)?", t)
            if len(nums) != 1:
                aplicavel = False
                break
            valores.append(float(nums[0].replace(",", ".")))
    flags["fe_sequencia_fora_de_ordem"] = aplicavel and valores != sorted(valores)

    flags["fe_enunciado_negativo"] = contem_termo(stem_norm, TERMOS_NEGATIVOS_ENUNCIADO)

    flags["fe_comando_pouco_explicito"] = (
        tipo == "Múltipla Escolha Simples"
        and not any(re.search(rf"\b{c}\b", stem_norm) for c in COMANDOS_EXPLICITOS))

    gabarito = normalizar_lexico(row.get("correct_option", "")).lower()
    pista = False
    if gabarito in opcoes:
        stem_tok = {t for t in tokens(stem) if len(t) > 4}
        gab_tok = tokens(opcoes[gabarito])
        outros = set()
        for l, t in opcoes.items():
            if l != gabarito:
                outros |= tokens(t)
        pista = len((stem_tok & gab_tok) - outros) >= 1
    flags["fe_pista_palavra_repetida"] = pista
    return flags


def fomulacao_estrutura_conteudo(row: Mapping[str, Any]) -> dict:
    """Alternativas desconexas e texto-base decorativo (regra determinística)."""
    flags = {"fe_alternativas_desconexas": False, "fe_textbase_decorativo_regra": False}
    opcoes = opcoes_validas(row)
    tipo = str(row.get("question_type", "")).strip()
    stem_norm = normalizar_lexico(row.get("stem", ""))

    contexto = " ".join(
        str(row.get(c, "")) for c in ["base_text", "stem", *COLS_STATEMENT, *COLS_ASSERTION]
        if texto_valido(row.get(c)))

    if tipo in {"Afirmação Incompleta", "Múltipla Escolha Simples"}:
        desconexas = [l for l, t in opcoes.items()
                      if len(tokens(t)) >= 6 and overlap_lexico(t, contexto) < 0.06]
        flags["fe_alternativas_desconexas"] = len(desconexas) >= 3

    if texto_valido(row.get("base_text")) and texto_valido(row.get("stem")):
        base_tok = tokens(row.get("base_text"))
        item_tok = tokens(" ".join([str(row.get("stem", "")), " ".join(opcoes.values())]))
        refere_contexto = re.search(r"\b(contexto|situacao|texto|descrit[ao]|apresentad[ao])\b", stem_norm)
        if (refere_contexto and len(base_tok) >= 12 and len(item_tok) >= 12
                and len(base_tok & item_tok) / len(base_tok) < 0.04):
            flags["fe_textbase_decorativo_regra"] = True
    return flags


def textbase_dispensavel_proxy(row: Mapping[str, Any]):
    """Fallback heurístico (sem LLM) para saber se o texto-base é dispensável."""
    if not texto_valido(row.get("base_text")):
        return pd.NA
    base_tok = tokens(row.get("base_text"))
    item_tok = tokens(" ".join([str(row.get("stem", ""))] + [str(row.get(c, "")) for c in OPCOES]))
    if not base_tok or not item_tok:
        return pd.NA
    return (len(base_tok & item_tok) / len(base_tok)) < 0.08


def classificar_texto_base_llm(row: Mapping[str, Any], model: str | None = None, timeout: float = 60) -> bool | None:
    """Classifica (LLM, base_text_classifier) se o texto-base é necessário para este item.
    Cai para o proxy léxico se a chamada à API falhar. Retorna None se não houver texto-base."""
    if not texto_valido(row.get("base_text")):
        return None
    try:
        from base_text_classifier import classificar_uso_texto_base

        resultado = classificar_uso_texto_base(
            base_text=row.get("base_text"),
            stem=row.get("stem"),
            options={letra: row.get(f"option_{letra}") for letra in "abcde"},
            correct_option=row.get("correct_option"),
            statements=[row.get(c) for c in COLS_STATEMENT],
            assertions=[row.get(c) for c in COLS_ASSERTION],
            question_id=row.get("question_id"),
            model=model,
            timeout=timeout,
        )
        return bool(resultado.get("texto_base_necessario"))
    except Exception:
        proxy = textbase_dispensavel_proxy(row)
        return None if pd.isna(proxy) else not bool(proxy)


# Verificação simbólica (sympy) feita à mão no notebook (diagnostico_e_selecao_itens.ipynb,
# célula 18 / exploracao_dados.ipynb, seção 12) para os itens de matemática do conjunto
# ORIGINAL. `True` = a conta bate (gabarito ok); `False` = gabarito matematicamente
# impossível. Não generaliza para itens fora deste conjunto (ver docstring do módulo).
VERIFICACAO_MATEMATICA_CONHECIDA: dict[int, bool] = {
    18: False, 39: True, 62: True, 90: True, 184: True, 246: True,
    353: True, 400: True, 483: True, 516: True, 567: True, 808: True, 653: True, 704: True,
}


def gabarito_matematico_impossivel(question_id: Any) -> bool:
    try:
        qid = int(question_id)
    except (TypeError, ValueError):
        return False
    return VERIFICACAO_MATEMATICA_CONHECIDA.get(qid) is False


def aplicar_formulacao_estrutura(
    row: Mapping[str, Any],
    texto_base_necessario: bool | None = None,
    gabarito_matematico_ok: bool | None = None,
    usar_verificacao_matematica_conhecida: bool = True,
) -> dict:
    """Roda os quatro blocos do eixo FE e agrega os textbase/matemática, que dependem
    de fontes externas (LLM / verificação numérica) em vez de regex sobre o próprio item."""
    flags: dict[str, Any] = {}
    flags.update(fomulacao_estrutura_gabarito(row))
    flags.update(fomulacao_estrutura_paralelismo(row))
    flags.update(fomulacao_estrutura_pistas(row))
    flags.update(fomulacao_estrutura_conteudo(row))

    if texto_base_necessario is None:
        proxy = textbase_dispensavel_proxy(row)
        flags["fe_textbase_dispensavel"] = bool(proxy) if not pd.isna(proxy) else False
    else:
        flags["fe_textbase_dispensavel"] = not bool(texto_base_necessario)

    if gabarito_matematico_ok is not None:
        flags["fe_gabarito_matematico_impossivel"] = gabarito_matematico_ok is False
    elif usar_verificacao_matematica_conhecida:
        flags["fe_gabarito_matematico_impossivel"] = gabarito_matematico_impossivel(row.get("question_id"))
    else:
        flags["fe_gabarito_matematico_impossivel"] = False

    return flags


COLS_INTRINSECOS_PREFIXO = "fe_"

# ---------------------------------------------------------------------------
# Eixo AP — Alinhamento Pedagógico (células 23, 25-26)
# ---------------------------------------------------------------------------
BLOOM_PISTAS = {
    "Lembrar": ["identifique", "indique", "cite", "liste", "defina", "nomeie", "aponte", "enumere", "reconheca"],
    "Compreender": ["explique", "descreva", "interprete", "resuma", "caracterize", "exemplifique", "classifique"],
    "Aplicar": ["aplique", "calcule", "resolva", "utilize", "determine", "empregue", "execute", "compute", "obtenha"],
    "Analisar": ["analise", "diferencie", "compare", "relacione", "examine", "distinga", "categorize", "infira", "contraste"],
    "Avaliar": ["avalie", "julgue", "critique", "justifique", "decida", "argumente", "valide", "verifique", "mais adequada", "mais adequado"],
    "Criar": ["elabore", "proponha", "crie", "produza", "construa", "planeje", "formule", "desenvolva", "projete", "sintetize"],
}


def detectar_bloom(stem: Any):
    if pd.isna(stem):
        return pd.NA
    s = normalizar_lexico(stem)
    for nivel, pistas in BLOOM_PISTAS.items():
        for pista in pistas:
            if re.search(rf"\b{re.escape(normalizar_lexico(pista))}\b", s):
                return nivel
    return pd.NA


def bloom_desalinhado(row: Mapping[str, Any]) -> tuple:
    """Retorna (desalinhado: bool, bloom_detectado: str|NA)."""
    detectado = detectar_bloom(row.get("stem"))
    desalinhado = (not pd.isna(detectado)) and detectado != row.get("taxonomy_level")
    return bool(desalinhado), detectado


def diagnosticar_formato(row: Mapping[str, Any]) -> list:
    """Inconsistências entre o tipo DECLARADO e o item efetivamente construído."""
    tipo = str(row.get("question_type", "")).strip()
    erros: list = []
    n_stmt = len(campos_presentes(row, COLS_STATEMENT))
    n_asrt = len(campos_presentes(row, COLS_ASSERTION))
    opcoes = opcoes_validas(row)
    stem_norm = normalizar_lexico(row.get("stem", ""))
    opcoes_txt = " ".join(normalizar_lexico(t) for t in opcoes.values())

    if tipo not in TIPOS_VALIDOS:
        erros.append(f"tipo declarado inválido: {tipo!r}")
        return erros

    if tipo == "Múltipla Escolha Complexa":
        if n_stmt < 4:
            erros.append(f"declarada Complexa mas tem {n_stmt} afirmativas (esperado 4)")
        if n_asrt > 0:
            erros.append("declarada Complexa mas tem campos de asserção")
        if not re.search(r"\b(i|ii|iii|iv)\b", opcoes_txt):
            erros.append("declarada Complexa mas alternativas não combinam I-IV")
        combos: dict = {}
        for l, t in opcoes.items():
            c = parse_romanos(t)
            if c and c in combos:
                erros.append(f"combinação I-IV duplicada ({combos[c].upper()} e {l.upper()})")
                break
            if c:
                combos[c] = l

    elif tipo == "Asserção-Razão":
        if n_asrt < 2:
            erros.append(f"declarada Asserção-Razão mas tem {n_asrt} asserções (esperado 2)")
        if n_stmt > 0:
            erros.append("declarada Asserção-Razão mas tem afirmativas")
        if not any(p in opcoes_txt for p in ["verdadeira", "falsa", "justifica", "proposicao"]):
            erros.append("declarada Asserção-Razão mas alternativas fora do padrão V/F+justifica")

    elif tipo in ("Afirmação Incompleta", "Múltipla Escolha Simples"):
        if n_stmt > 0 or n_asrt > 0:
            erros.append("declarada simples mas tem campos de Complexa/Asserção-Razão")
        if re.search(r"\b(afirmativas|assertivas)\b", stem_norm):
            erros.append("declarada simples mas o enunciado se comporta como Complexa")
    return erros


# ---------------------------------------------------------------------------
# Eixo APL — Aplicação (célula 29): viés de posição do gabarito no CONJUNTO
# ---------------------------------------------------------------------------
def calcular_letra_dominante(df: pd.DataFrame) -> str:
    """Letra de gabarito sobre-representada no conjunto (célula 29 do notebook)."""
    return df["correct_option"].astype(str).str.upper().value_counts().idxmax()


# ---------------------------------------------------------------------------
# Catálogo de melhorias acionáveis (célula 37) + mapeamento para os critérios
# numerados usados no prompt de correção (system_correction_template.txt /
# prompt_correction_template_modularizado.txt)
# ---------------------------------------------------------------------------
MELHORIAS: dict[str, tuple] = {
    "fe_alternativas_ausentes": ("completar as 5 alternativas", "FE"),
    "fe_enunciado_ausente": ("escrever o enunciado", "FE"),
    "fe_gabarito_invalido": ("corrigir gabarito para o intervalo A-E", "FE"),
    "fe_gabarito_vazio": ("apontar gabarito para alternativa preenchida", "FE"),
    "fe_alternativas_duplicadas": ("eliminar alternativas equivalentes", "FE"),
    "fe_gabarito_duplicado": ("eliminar duplicata do gabarito", "FE"),
    "fe_gabarito_equivale_distrator": ("diferenciar gabarito do distrator equivalente", "FE"),
    "fe_alternativas_similares": ("diferenciar alternativas quase idênticas", "FE"),
    "fe_gabarito_matematico_impossivel": ("corrigir gabarito matematicamente impossível", "FE"),
    "fe_vies_tamanho_maior": ("encurtar o gabarito (mais longo que os distratores)", "FE"),
    "fe_vies_tamanho_menor": ("alongar o gabarito (mais curto que os distratores)", "FE"),
    "fe_extensao_desigual": ("homogeneizar a extensão das alternativas (paralelismo)", "FE"),
    "fe_metalinguistica": ("remover alternativa \"todas/nenhuma das anteriores\"", "FE"),
    "fe_distratores_termos_absolutos": ("suavizar termos absolutos nos distratores", "FE"),
    "fe_pista_palavra_repetida": ("eliminar pista: palavra do enunciado só no gabarito", "FE"),
    "fe_enunciado_negativo": ("reescrever enunciado em forma afirmativa", "FE"),
    "fe_comando_pouco_explicito": ("explicitar o comando do enunciado", "FE"),
    "fe_lacuna_enunciado": ("remover lacuna (__) do enunciado", "FE"),
    "fe_verdadeiro_falso": ("reescrever alternativas V/F disfarçadas", "FE"),
    "fe_sequencia_fora_de_ordem": ("ordenar alternativas numéricas", "FE"),
    "fe_textbase_dispensavel": ("tornar o texto-base necessário ou removê-lo", "FE"),
    "fe_textbase_decorativo_regra": ("conectar o texto-base ao enunciado", "FE"),
    "fe_alternativas_desconexas": ("conectar as alternativas ao conteúdo do item", "FE"),
    "ap_bloom_desalinhado": ("alinhar o verbo do enunciado ao Bloom declarado", "AP"),
    "ap_formato_incompativel": ("ajustar o formato ao tipo declarado", "AP"),
    "apl_gabarito_letra_dominante": ("reposicionar o gabarito (viés de letra no conjunto)", "APL"),
}

# flag de código -> nº do critério (1-19) no prompt de correção. Flags que indicam
# item inutilizável (enunciado/alternativas ausentes) ficam de fora: esses itens são
# descartados antes de chegar à correção (ver `descartar_auto` em `diagnosticar_item`).
FLAG_PARA_CRITERIO: dict[str, int] = {
    "fe_gabarito_invalido": 1,
    "fe_gabarito_vazio": 1,
    "fe_alternativas_duplicadas": 2,
    "fe_alternativas_similares": 2,
    "fe_gabarito_equivale_distrator": 3,
    "fe_gabarito_duplicado": 3,
    "fe_gabarito_matematico_impossivel": 4,
    "fe_extensao_desigual": 5,
    "fe_vies_tamanho_maior": 6,
    "fe_vies_tamanho_menor": 6,
    "fe_distratores_termos_absolutos": 7,
    "fe_metalinguistica": 8,
    "fe_pista_palavra_repetida": 9,
    "fe_enunciado_negativo": 10,
    "fe_comando_pouco_explicito": 11,
    "fe_lacuna_enunciado": 12,
    "fe_verdadeiro_falso": 13,
    "fe_sequencia_fora_de_ordem": 14,
    "fe_textbase_dispensavel": 15,
    "fe_textbase_decorativo_regra": 15,
    "fe_alternativas_desconexas": 16,
    "ap_bloom_desalinhado": 17,
    "ap_formato_incompativel": 18,
    "apl_gabarito_letra_dominante": 19,
}

# critério -> campos de status de data/analises_maritaca.csv que, se PROBLEMA (ou,
# no caso do Bloom, "NAO"), também acionam aquele critério a partir da camada LLM.
CRITERIO_PARA_CAMPOS_LLM: dict[int, list] = {
    1: ["criterios.corretude_unicidade_gabarito.status"],
    3: ["criterios.corretude_unicidade_gabarito.status"],
    4: ["criterios.corretude_unicidade_gabarito.status"],
    5: ["criterios.paralelismo.status"],
    6: ["criterios.convergencia_pistas.status"],
    7: ["criterios.plausibilidade_distratores.status_geral", "criterios.termos_absolutos_generalizacoes.status"],
    9: ["criterios.convergencia_pistas.status"],
    10: ["criterios.clareza_linguistica.status"],
    11: ["criterios.ideia_central_enunciado.status"],
    12: ["criterios.clareza_linguistica.status"],
    15: ["criterios.coerencia_conteudo.status"],
    16: ["criterios.coerencia_conteudo.status"],
    17: ["bloom.alinhamento"],
}


def montar_melhorias(
    flags: Mapping[str, Any],
    bloom_detectado: Any = None,
    taxonomy_level: Any = None,
    erros_formato: list | None = None,
) -> list:
    """Traduz as flags de código em uma lista de melhorias acionáveis (célula 37)."""
    erros_formato = erros_formato or []
    itens = []
    for flag, (rotulo_m, eixo) in MELHORIAS.items():
        if not bool(flags.get(flag)):
            continue
        detalhe = ""
        if flag == "ap_bloom_desalinhado":
            detalhe = f" (detectado {bloom_detectado} vs declarado {taxonomy_level})"
        elif flag == "ap_formato_incompativel" and erros_formato:
            detalhe = f" ({erros_formato[0]})"
        itens.append(f"[{eixo}] {rotulo_m}{detalhe}")
    return itens


def criterios_flagrados_por_codigo(flags: Mapping[str, Any]) -> list:
    """Nºs (1-19) dos critérios do prompt de correção com problema detectado por CÓDIGO."""
    return sorted({FLAG_PARA_CRITERIO[flag] for flag, valor in flags.items()
                   if bool(valor) and flag in FLAG_PARA_CRITERIO})


def _eh_problema_llm(campo: str, valor: Any) -> bool:
    v = str(valor).strip().upper()
    if campo == "bloom.alinhamento":
        return v == "NAO"
    return v in {"PROBLEMA", "INADEQUADO", "NAO", "GABARITO_INCORRETO", "GABARITO_DUPLICADO"}


def criterios_flagrados_por_llm(linha_llm: Mapping[str, Any] | None) -> list:
    """Nºs (1-19) dos critérios com problema apontado pela avaliação pedagógica (LLM)."""
    if not linha_llm:
        return []
    flagrados = set()
    for criterio, campos in CRITERIO_PARA_CAMPOS_LLM.items():
        for campo in campos:
            if campo in linha_llm and _eh_problema_llm(campo, linha_llm.get(campo)):
                flagrados.add(criterio)
                break
    return sorted(flagrados)


def criterios_flagrados(flags: Mapping[str, Any], linha_llm: Mapping[str, Any] | None = None) -> list:
    """União dos critérios acionados por código e (opcionalmente) por LLM."""
    return sorted(set(criterios_flagrados_por_codigo(flags)) | set(criterios_flagrados_por_llm(linha_llm)))


# ---------------------------------------------------------------------------
# Função de entrada única: diagnóstico completo de UM item
# ---------------------------------------------------------------------------
def diagnosticar_item(
    row: Mapping[str, Any],
    letra_dominante: str | None = None,
    texto_base_necessario: bool | None = None,
    gabarito_matematico_ok: bool | None = None,
    usar_verificacao_matematica_conhecida: bool = True,
    linha_llm: Mapping[str, Any] | None = None,
) -> dict:
    """Roda o diagnóstico de código (eixos FE + AP + APL) para um único item.

    Não depende do resto do dataset, exceto por `letra_dominante` (estatística do
    conjunto — calcule uma vez com `calcular_letra_dominante(df)` e passe aqui).
    `linha_llm`, se informada, é combinada com os flags de código para produzir a
    lista final de critérios acionados (`criterios_flagrados`), usada para montar o
    prompt de correção modular.
    """
    flags: dict[str, Any] = {}
    flags.update(aplicar_formulacao_estrutura(
        row, texto_base_necessario, gabarito_matematico_ok, usar_verificacao_matematica_conhecida,
    ))

    bloom_flag, bloom_detectado = bloom_desalinhado(row)
    flags["ap_bloom_desalinhado"] = bloom_flag

    erros_formato = diagnosticar_formato(row)
    flags["ap_formato_incompativel"] = len(erros_formato) > 0

    gabarito_atual = str(row.get("correct_option", "")).strip().upper()
    flags["apl_gabarito_letra_dominante"] = bool(letra_dominante) and gabarito_atual == str(letra_dominante).upper()

    o_que_melhorar = montar_melhorias(flags, bloom_detectado, row.get("taxonomy_level"), erros_formato)
    n_fe = sum(1 for m in o_que_melhorar if m.startswith("[FE]"))
    n_ap = sum(1 for m in o_que_melhorar if m.startswith("[AP]"))
    n_apl = sum(1 for m in o_que_melhorar if m.startswith("[APL]"))

    # fe_gabarito_matematico_impossivel NÃO entra aqui: ao contrário de enunciado
    # ausente/formato quebrado, esse problema tem correção prevista (critério 4,
    # "corrigir gabarito matematicamente impossível") e deve seguir para o loop de
    # correção como qualquer outro item, em vez de ser descartado antes de tentar.
    descartar_auto = bool(
        flags.get("fe_enunciado_ausente")
        or len(erros_formato) >= 3
    )

    return {
        "question_id": row.get("question_id"),
        "flags": flags,
        "bloom_detectado": bloom_detectado,
        "erros_formato": erros_formato,
        "o_que_melhorar": o_que_melhorar,
        "n_fe": n_fe,
        "n_ap": n_ap,
        "n_apl": n_apl,
        "n_melhorias": len(o_que_melhorar),
        "criterios_flagrados": criterios_flagrados(flags, linha_llm),
        "descartar_auto": descartar_auto,
    }


# ---------------------------------------------------------------------------
# Camada subjetiva (Maritaca) — apenas contexto, não entra em o_que_melhorar
# (célula 34)
# ---------------------------------------------------------------------------
def eh_problema_maritaca(v: Any) -> bool:
    return str(v).strip().upper() in {"PROBLEMA", "INADEQUADO", "NAO", "GABARITO_INCORRETO", "GABARITO_DUPLICADO"}


def observacoes_subjetivas_da_linha(linha_llm: Mapping[str, Any] | None) -> list:
    if not linha_llm:
        return []
    cols_status = [c for c in linha_llm.keys() if str(c).startswith("criterios.") and str(c).endswith(".status")]
    return [
        str(c).replace("criterios.", "").replace(".status", "").replace("_", " ").title()
        for c in cols_status if eh_problema_maritaca(linha_llm.get(c))
    ]


# ---------------------------------------------------------------------------
# Diagnóstico em lote — reproduz o pipeline completo do notebook (células 4-46),
# salvando progresso parcial em disco e imprimindo o andamento.
# ---------------------------------------------------------------------------
def _classificar_texto_base_lote(
    df: pd.DataFrame,
    caminho_cache: Path,
    usar_llm: bool,
    mostrar_progresso: bool,
) -> dict:
    """Devolve {question_id: texto_base_necessario(bool)}, reaproveitando cache em disco
    e chamando a LLM (com fallback heurístico) apenas para o que falta. Salva a cada item."""
    colunas = ["question_id", "comando_depende_do_texto", "alternativas_dependem_do_texto",
               "gabarito_depende_do_texto", "texto_base_necessario", "justificativa"]
    if caminho_cache.exists():
        cache = pd.read_csv(caminho_cache, encoding="utf-8-sig")
    else:
        cache = pd.DataFrame(columns=colunas)

    resultado = dict(zip(cache.get("question_id", []), cache.get("texto_base_necessario", [])))
    resultado = {int(k): (str(v).strip().lower() == "true") for k, v in resultado.items() if pd.notna(k)}

    com_texto_base = df[df["base_text"].apply(texto_valido)]
    faltantes = com_texto_base[~com_texto_base["question_id"].isin(resultado.keys())]
    if mostrar_progresso:
        print(f"  [texto-base] {len(resultado)} em cache | {len(faltantes)} novas a classificar"
              f" ({'LLM' if usar_llm else 'proxy léxico'})")

    for posicao, (_, row) in enumerate(faltantes.iterrows(), 1):
        qid = int(row["question_id"])
        if usar_llm:
            necessario = classificar_texto_base_llm(row)
        else:
            proxy = textbase_dispensavel_proxy(row)
            necessario = None if pd.isna(proxy) else not bool(proxy)
        if necessario is None:
            continue
        resultado[qid] = necessario
        nova_linha = {
            "question_id": qid, "comando_depende_do_texto": None,
            "alternativas_dependem_do_texto": None, "gabarito_depende_do_texto": None,
            "texto_base_necessario": necessario, "justificativa": "",
        }
        # salva a cada item processado, para retomar de onde parou em caso de erro/interrupção
        cache = pd.concat([cache, pd.DataFrame([nova_linha])], ignore_index=True)
        caminho_cache.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(caminho_cache, index=False, encoding="utf-8-sig")
        if mostrar_progresso:
            print(f"    [{posicao}/{len(faltantes)}] question_id={qid}: "
                  f"texto_base_necessario={necessario}")
    return resultado


def diagnosticar_dataframe(
    caminho_entrada: Path = REJECTED_PATH,
    caminho_textbase_cache: Path = TEXTBASE_CACHE_PATH,
    caminho_maritaca: Path | None = MARITACA_PATH,
    caminho_veredito_saida: Path | None = VEREDITO_PATH,
    caminho_refino_saida: Path | None = REFINO_PATH,
    usar_llm_textbase: bool = True,
    min_prob: int = 2,
    max_prob: int = 6,
    n_alvo: int = 100,
    mostrar_progresso: bool = True,
) -> pd.DataFrame:
    """Reproduz o pipeline de `diagnostico_e_selecao_itens.ipynb` de ponta a ponta e
    devolve o DataFrame diagnosticado. Salva progresso parcial a cada item processado
    (cache de texto-base) e, ao final, `veredito_por_questao.csv` e o pacote de itens
    selecionados para refino — mesmos artefatos gerados pelo notebook.

    Pode ser chamada tanto para o dataset original (`data/rejected_questions.csv`)
    quanto para qualquer outro CSV com o mesmo formato (ex.: uma leva de itens já
    corrigidos, para conferir se as correções realmente zeraram os problemas).
    """
    df = pd.read_csv(caminho_entrada)
    df["discipline"] = df["discipline"].map(normalizar_disciplina)
    total = len(df)
    if mostrar_progresso:
        print(f"[diagnostico] {total} itens carregados de {caminho_entrada}")

    letra_dominante = calcular_letra_dominante(df)

    textbase_necessario = _classificar_texto_base_lote(
        df, caminho_textbase_cache, usar_llm_textbase, mostrar_progresso,
    )

    df_mar = None
    if caminho_maritaca is not None and Path(caminho_maritaca).exists():
        df_mar = pd.read_csv(caminho_maritaca, encoding="utf-8-sig")
        id_col = "id" if "id" in df_mar.columns else "question_id"
        df_mar = df_mar.rename(columns={id_col: "question_id"})
        df_mar["question_id"] = df_mar["question_id"].astype(int)
        df_mar = df_mar.set_index("question_id", drop=False)

    linhas_resultado = []
    for posicao, (_, row) in enumerate(df.iterrows(), 1):
        qid = int(row["question_id"])
        linha_llm = df_mar.loc[qid].to_dict() if (df_mar is not None and qid in df_mar.index) else None
        diagnostico = diagnosticar_item(
            row,
            letra_dominante=letra_dominante,
            texto_base_necessario=textbase_necessario.get(qid),
            linha_llm=linha_llm,
        )
        n_subjetivos_lista = observacoes_subjetivas_da_linha(linha_llm)
        linhas_resultado.append({
            "question_id": qid,
            **diagnostico["flags"],
            "bloom_detectado": diagnostico["bloom_detectado"],
            "erros_formato": diagnostico["erros_formato"],
            "n_erros_formato": len(diagnostico["erros_formato"]),
            "o_que_melhorar": diagnostico["o_que_melhorar"],
            "n_fe": diagnostico["n_fe"],
            "n_ap": diagnostico["n_ap"],
            "n_apl": diagnostico["n_apl"],
            "n_melhorias": diagnostico["n_melhorias"],
            "criterios_flagrados": diagnostico["criterios_flagrados"],
            "descartar_auto": diagnostico["descartar_auto"],
            "observacoes_subjetivas": n_subjetivos_lista,
            "n_subjetivos": len(n_subjetivos_lista),
        })
        if mostrar_progresso and (posicao % 25 == 0 or posicao == total):
            print(f"  [{posicao}/{total}] diagnosticados (último: question_id={qid}, "
                  f"n_melhorias={diagnostico['n_melhorias']})")

    diag_df = pd.DataFrame(linhas_resultado)
    df = df.merge(diag_df, on="question_id", how="left")

    df["score_refino"] = (
        df["n_melhorias"] * 1.0
        + df["n_fe"] * 0.5
        + df["n_ap"] * 0.3
        - df["n_subjetivos"] * 0.4
    )
    df["elegivel_refino"] = (
        (~df["descartar_auto"])
        & (df["n_melhorias"] >= min_prob)
        & (df["n_melhorias"] <= max_prob)
    )

    n_formulacao_estrutura_cols = [c for c in df.columns if c.startswith("fe_")]
    df["n_formulacao_estrutura"] = df[n_formulacao_estrutura_cols].sum(axis=1)

    selecionados = (
        df[df["elegivel_refino"]]
        .sort_values(["score_refino", "n_formulacao_estrutura"], ascending=False)
        .head(n_alvo)
    )
    ids_selecionados = set(selecionados["question_id"])

    def _decidir(row):
        if row["descartar_auto"]:
            if row["fe_gabarito_matematico_impossivel"]:
                return "DESCARTAR (gabarito matematicamente impossível)"
            if row["fe_enunciado_ausente"]:
                return "DESCARTAR (enunciado ausente)"
            return "DESCARTAR (tipo declarado irreconciliável)"
        if row["question_id"] in ids_selecionados:
            return "REFINAR_SELECIONADO"
        if row["n_melhorias"] == 0:
            return "INVESTIGAR (rejeitado, mas sem problema objetivo detectado)"
        if row["n_melhorias"] < min_prob:
            return "INVESTIGAR (diagnóstico insuficiente para guiar o refino)"
        if row["n_melhorias"] > max_prob:
            return "DESCARTAR (problemas demais para refino viável)"
        return "REFINAR (elegível, fora do top {})".format(n_alvo)

    df["decisao"] = df.apply(_decidir, axis=1)
    df["categoria"] = df["decisao"].str.split(" ").str[0]

    if mostrar_progresso:
        print("[diagnostico] decisões:")
        print(df["categoria"].value_counts().to_string())

    if caminho_veredito_saida is not None:
        veredito = df[[
            "question_id", "question_type", "discipline", "taxonomy_level", "correct_option",
            "n_fe", "n_ap", "n_apl", "n_melhorias", "n_subjetivos", "score_refino",
            "elegivel_refino", "decisao", "categoria", "o_que_melhorar", "observacoes_subjetivas",
        ]].copy()
        veredito["o_que_melhorar"] = veredito["o_que_melhorar"].apply(lambda x: " | ".join(x or []))
        veredito["observacoes_subjetivas"] = veredito["observacoes_subjetivas"].apply(lambda x: " | ".join(x or []))
        Path(caminho_veredito_saida).parent.mkdir(parents=True, exist_ok=True)
        veredito.to_csv(caminho_veredito_saida, index=False, encoding="utf-8-sig")
        if mostrar_progresso:
            print(f"[diagnostico] salvo: {caminho_veredito_saida}")

    if caminho_refino_saida is not None:
        cols_item = ["question_id", "question_type", "discipline", "taxonomy_level", "base_text", "stem",
                     *COLS_STATEMENT, *COLS_ASSERTION, *OPCOES, "correct_option"]
        cols_item = [c for c in cols_item if c in df.columns]
        pacote = df[df["question_id"].isin(ids_selecionados)][cols_item].merge(
            df[["question_id", "o_que_melhorar", "n_melhorias", "n_fe", "n_ap", "n_apl"]],
            on="question_id", how="left")
        pacote["o_que_melhorar"] = pacote["o_que_melhorar"].apply(lambda x: " | ".join(x or []))
        Path(caminho_refino_saida).parent.mkdir(parents=True, exist_ok=True)
        pacote.to_csv(caminho_refino_saida, index=False, encoding="utf-8-sig")
        if mostrar_progresso:
            print(f"[diagnostico] salvo: {caminho_refino_saida} ({len(pacote)} itens)")

    return df


if __name__ == "__main__":
    diagnosticar_dataframe()
