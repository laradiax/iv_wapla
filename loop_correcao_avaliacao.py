"""Loop multi-agente de correção e avaliação de itens.

Para cada item, repete até `n_iteracoes` vezes (padrão 3) o ciclo:

  1. AVALIAR  — `item_quality_analyzer.analisar_item` (agente pedagógico, LLM) lê o
     item ATUAL (original na 1ª rodada, corrigido nas seguintes) e produz uma nova
     avaliação qualitativa.
  2. DIAGNOSTICAR — `item_diagnostic_agent.diagnosticar_item` (agente de código)
     roda as checagens determinísticas sobre o mesmo item ATUAL e as combina com a
     avaliação LLM fresca, produzindo a lista de critérios (1-19) ainda com problema.
  3. Se não sobrou nenhum critério ativo (nem por código, nem por LLM), o item está
     aprovado e o ciclo encerra antes da hora.
  4. Caso contrário, CORRIGIR — `item_correction_builder.corrigir_item_modular`
     (agente corretor, LLM) usa o prompt modular — só com os critérios ainda
     ativos — para produzir uma nova versão do item, que vira a entrada da próxima
     rodada.

Cada rodada de cada item é salva incrementalmente em
`data/loop_correcao/<question_id>.json`, e um resumo do lote em
`data/loop_correcao/_resumo_lote.csv` — dá para acompanhar o progresso e retomar a
leitura sem esperar o processamento inteiro terminar.

Uso:
    from loop_correcao_avaliacao import executar_ciclo_item, executar_lote

    resultado = executar_ciclo_item(2)                  # um item, até 3 rodadas
    resumo    = executar_lote([2, 3, 5], n_iteracoes=3)  # vários itens em sequência
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import item_diagnostic_agent as diagnostic_agent
from item_correction_builder import (
    calcular_letra_dominante,
    carregar_avaliacoes_llm,
    carregar_itens_refino,
    corrigir_item_modular,
)
from item_quality_analyzer import (
    RespostaJSONInvalida,
    _campo,
    analisar_item,
    criar_modelo,
)

ROOT = Path(__file__).resolve().parent
SAIDA_DIR = ROOT / "data" / "loop_correcao"

N_ITERACOES_PADRAO = 3

CAMPOS_ITEM = [
    "question_id", "question_type", "discipline", "taxonomy_level", "base_text", "stem",
    "statement_i", "statement_ii", "statement_iii", "statement_iv",
    "assertion_i", "assertion_ii",
    "option_a", "option_b", "option_c", "option_d", "option_e", "correct_option",
]


def _campos_item(dado: Mapping[str, Any]) -> dict:
    """Normaliza um item (Series, dict ou item_corrigido da LLM) para um dict plano."""
    return {c: _campo(dado, c) for c in CAMPOS_ITEM}


def _achatar_avaliacao(avaliacao: Mapping[str, Any] | None) -> dict | None:
    """Achata o JSON aninhado de `analisar_item` (`criterios.paralelismo.status`, ...)
    no mesmo formato de `data/analises_maritaca.csv`, usado pelos agentes de
    diagnóstico e correção para ler os campos por chave pontilhada."""
    if not avaliacao:
        return None
    return pd.json_normalize([avaliacao], sep=".").iloc[0].to_dict()


def _decisao_aprovada(linha_llm_flat: Mapping[str, Any] | None) -> bool:
    """Lê `sintese.decisao_recomendada` no formato achatado (mesmo de
    data/analises_maritaca.csv e do retorno de `_achatar_avaliacao`)."""
    if not linha_llm_flat:
        return False
    return str(linha_llm_flat.get("sintese.decisao_recomendada", "")).strip().upper() == "APROVAR"


def _salvar_progresso(caminho: Path, rodadas: list) -> None:
    """Grava a lista de rodadas já concluídas (tmp + replace, para nunca deixar um
    arquivo de saída pela metade caso o processo seja interrompido no meio)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rodadas, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(caminho)


def executar_ciclo_item(
    question_id: int,
    n_iteracoes: int = N_ITERACOES_PADRAO,
    item_inicial: Mapping[str, Any] | None = None,
    avaliacao_inicial: Mapping[str, Any] | None = None,
    avaliacoes_llm: pd.DataFrame | None = None,
    reaproveitar_avaliacao_inicial: bool = True,
    letra_dominante: str | None = None,
    llm_avaliador=None,
    llm_corretor=None,
    saida_dir: Path = SAIDA_DIR,
    mostrar_progresso: bool = True,
    _print_lock: threading.Lock | None = None,
) -> dict:
    """Roda o ciclo avaliar -> diagnosticar -> corrigir até `n_iteracoes` vezes (ou
    até o item não ter mais problema) para UM item — ou seja,
    avalia -> corrige -> avalia -> corrige -> ... -> avalia, com uma avaliação a
    mais que o nº de correções (a última avaliação apenas confirma o resultado da
    última correção, sem corrigir de novo).

    `item_inicial`, se informado, sobrepõe a leitura de
    `data/itens_para_refino_top100.csv` — use para encadear o resultado de uma
    corrida anterior ou para rodar o loop sobre um item fora do top 100.

    Com `reaproveitar_avaliacao_inicial=True` (padrão), a 1ª avaliação NÃO é refeita
    do zero: reaproveita a que já existe em `data/analises_maritaca.csv` (ou
    `avaliacao_inicial`, se informada) — afinal o item ainda é o original, então a
    avaliação já feita continua valendo. Use `False` para forçar uma avaliação nova
    pela LLM já na 1ª rodada (ex.: depois de alterar os prompts de avaliação/correção
    e querer um resultado limpo, sem reaproveitar nada de uma rodada anterior). A
    partir da 2ª avaliação (item já corrigido pelo menos uma vez), a avaliação é
    sempre refeita pela LLM, porque o texto mudou.

    `_print_lock`, se informado (uso interno de `executar_lote` com `max_workers>1`),
    serializa as mensagens de progresso para não intercalar a saída de itens
    processados em paralelo.

    Retorna um dict com `question_id`, a lista `rodadas` (uma entrada por rodada
    executada, já salva em disco a cada passo) e `item_final`.
    """

    def _print(mensagem: str) -> None:
        if not mostrar_progresso:
            return
        if _print_lock is not None:
            with _print_lock:
                print(mensagem)
        else:
            print(mensagem)

    letra_dominante = letra_dominante or calcular_letra_dominante()
    llm_avaliador = llm_avaliador or criar_modelo()
    llm_corretor = llm_corretor or llm_avaliador

    if item_inicial is not None:
        item = _campos_item(item_inicial)
    else:
        itens = carregar_itens_refino()
        if question_id not in itens.index:
            raise KeyError(
                f"question_id {question_id} não está em itens_para_refino_top100.csv "
                "e nenhum `item_inicial` foi informado."
            )
        item = _campos_item(itens.loc[question_id])

    if avaliacao_inicial is None and reaproveitar_avaliacao_inicial:
        avaliacoes_llm = avaliacoes_llm if avaliacoes_llm is not None else carregar_avaliacoes_llm()
        if question_id in avaliacoes_llm.index:
            avaliacao_inicial = avaliacoes_llm.loc[question_id].to_dict()

    caminho_saida = saida_dir / f"{question_id}.json"
    rodadas: list = []

    # n_iteracoes correções, cada uma precedida de uma avaliação -> n_iteracoes+1
    # avaliações no total (a última é só para confirmar o resultado da última correção)
    total_passos = n_iteracoes + 1
    origem_inicial = "cache (data/analises_maritaca.csv)" if avaliacao_inicial is not None else "nenhuma (será avaliado pela LLM)"
    _print(f"[loop] question_id={question_id}: iniciando ciclo de até {n_iteracoes} correção(ões) "
           f"({total_passos} avaliação(ões)) | avaliação inicial: {origem_inicial}")

    for rodada in range(1, total_passos + 1):
        if rodada == 1 and avaliacao_inicial is not None:
            linha_llm_flat = avaliacao_inicial
            avaliacao_origem = "cache"
            erro_avaliacao = None
        else:
            try:
                avaliacao_bruta = analisar_item(item, llm_avaliador)
                linha_llm_flat = _achatar_avaliacao(avaliacao_bruta)
                avaliacao_origem = "llm"
                erro_avaliacao = None
            except RespostaJSONInvalida as erro:
                linha_llm_flat = None
                avaliacao_origem = None
                erro_avaliacao = str(erro)

        diagnostico = diagnostic_agent.diagnosticar_item(
            item,
            letra_dominante=letra_dominante,
            # a verificação sympy hardcoded (VERIFICACAO_MATEMATICA_CONHECIDA) só vale
            # para o enunciado ORIGINAL; a partir da 1ª correção, o texto muda e a
            # factualidade do gabarito passa a depender só da resolução independente da LLM
            usar_verificacao_matematica_conhecida=(rodada == 1),
            linha_llm=linha_llm_flat,
        )
        criterios_ativos = diagnostico["criterios_flagrados"]

        registro = {
            "question_id": question_id,
            "rodada": rodada,
            "timestamp": datetime.now().astimezone().isoformat(),
            "item": dict(item),
            "diagnostico_codigo": {
                "o_que_melhorar": diagnostico["o_que_melhorar"],
                "n_melhorias": diagnostico["n_melhorias"],
                "descartar_auto": diagnostico["descartar_auto"],
            },
            "avaliacao_llm": linha_llm_flat,
            "avaliacao_origem": avaliacao_origem,
            "erro_avaliacao_llm": erro_avaliacao,
            "criterios_ativos": criterios_ativos,
        }

        decisao_llm = (linha_llm_flat or {}).get("sintese.decisao_recomendada", "N/D")
        _print(f"  [{question_id}] avaliação {rodada}/{total_passos} (origem: {avaliacao_origem or 'erro'}): "
               f"{diagnostico['n_melhorias']} problema(s) de código | "
               f"critérios ativos={criterios_ativos} | decisão LLM={decisao_llm}")

        if diagnostico["descartar_auto"]:
            registro["status"] = "DESCARTADO"
            rodadas.append(registro)
            _salvar_progresso(caminho_saida, rodadas)
            _print(f"  [{question_id}] descartado automaticamente (problema insanável) — ciclo encerrado")
            break

        sem_problema = (not criterios_ativos) and (_decisao_aprovada(linha_llm_flat) or linha_llm_flat is None)
        if sem_problema:
            registro["status"] = "APROVADO"
            rodadas.append(registro)
            _salvar_progresso(caminho_saida, rodadas)
            _print(f"  [{question_id}] sem problemas remanescentes — ciclo encerrado na avaliação {rodada}")
            break

        if rodada == total_passos:
            registro["status"] = "LIMITE_RODADAS_ATINGIDO"
            rodadas.append(registro)
            _salvar_progresso(caminho_saida, rodadas)
            _print(f"  [{question_id}] limite de {n_iteracoes} correção(ões) atingido, ainda com problema(s)")
            break

        try:
            correcao = corrigir_item_modular(
                item,
                criterios_ativos,
                " | ".join(diagnostico["o_que_melhorar"]) or "Nenhum problema objetivo listado.",
                llm_corretor,
                linha_llm=linha_llm_flat,
                letra_dominante=letra_dominante,
            )
            registro["correcao"] = correcao
            registro["status"] = "CORRIGIDO"
            item = _campos_item({**item, **correcao.get("item_corrigido", {})})
        except RespostaJSONInvalida as erro:
            registro["status"] = "ERRO_CORRECAO"
            registro["erro_correcao"] = str(erro)
            rodadas.append(registro)
            _salvar_progresso(caminho_saida, rodadas)
            _print(f"  [{question_id}] ERRO ao corrigir na rodada {rodada}: {erro} — ciclo encerrado")
            break

        rodadas.append(registro)
        _salvar_progresso(caminho_saida, rodadas)

    _print(f"[loop] question_id={question_id}: histórico salvo em {caminho_saida}")

    return {"question_id": question_id, "rodadas": rodadas, "item_final": item}


STATUS_TERMINAIS = {"APROVADO", "DESCARTADO", "LIMITE_RODADAS_ATINGIDO", "ERRO_CORRECAO"}


def executar_lote(
    question_ids: list,
    n_iteracoes: int = N_ITERACOES_PADRAO,
    saida_dir: Path = SAIDA_DIR,
    pular_concluidos: bool = True,
    reaproveitar_avaliacao_inicial: bool = True,
    max_workers: int = 1,
    mostrar_progresso: bool = True,
) -> pd.DataFrame:
    """Roda `executar_ciclo_item` para vários itens, salvando um resumo incremental
    (`data/loop_correcao/_resumo_lote.csv`) a cada item concluído — assim dá pra
    acompanhar o progresso do lote sem esperar todos terminarem.

    Com `pular_concluidos=True` (padrão), itens que já têm `data/loop_correcao/<id>.json`
    terminado (status final em `STATUS_TERMINAIS`) são pulados — permite retomar um
    lote grande interrompido no meio sem refazer (e pagar de novo por) o que já rodou.

    `max_workers>1` processa vários itens em paralelo (threads — as chamadas são
    todas I/O de rede, então o GIL não é gargalo). Cada item grava seu próprio
    arquivo (`<question_id>.json`), então não há disputa entre threads por arquivo;
    as mensagens de progresso usam um lock para não intercalar linhas de itens
    diferentes.
    """
    letra_dominante = calcular_letra_dominante()
    llm = criar_modelo()
    avaliacoes_llm = carregar_avaliacoes_llm()

    resumo_path = saida_dir / "_resumo_lote.csv"
    linhas: list = []
    print_lock = threading.Lock() if max_workers > 1 else None

    def _salvar_resumo() -> None:
        resumo_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(linhas).to_csv(resumo_path, index=False, encoding="utf-8-sig")

    total = len(question_ids)
    pendentes = []
    for posicao, qid in enumerate(question_ids, 1):
        caminho_existente = saida_dir / f"{qid}.json"
        if pular_concluidos and caminho_existente.exists():
            try:
                rodadas_existentes = json.loads(caminho_existente.read_text(encoding="utf-8"))
                status_existente = rodadas_existentes[-1]["status"] if rodadas_existentes else None
            except Exception:
                rodadas_existentes, status_existente = [], None
            if status_existente in STATUS_TERMINAIS:
                if mostrar_progresso:
                    print(f"=== [{posicao}/{total}] question_id={qid}: já concluído "
                          f"({status_existente}), pulando ===")
                linhas.append({
                    "question_id": qid,
                    "rodadas_executadas": len(rodadas_existentes),
                    "status_final": status_existente,
                })
                continue
        pendentes.append(qid)

    _salvar_resumo()
    if mostrar_progresso:
        print(f"[loop] {len(pendentes)} item(ns) pendente(s) de {total} | max_workers={max_workers}")

    def _rodar(qid: int) -> dict:
        return executar_ciclo_item(
            qid,
            n_iteracoes=n_iteracoes,
            avaliacoes_llm=avaliacoes_llm,
            reaproveitar_avaliacao_inicial=reaproveitar_avaliacao_inicial,
            letra_dominante=letra_dominante,
            llm_avaliador=llm,
            llm_corretor=llm,
            saida_dir=saida_dir,
            mostrar_progresso=mostrar_progresso,
            _print_lock=print_lock,
        )

    def _registrar_resultado(qid: int, resultado: dict | None, erro: Exception | None) -> None:
        if erro is not None:
            linhas.append({"question_id": qid, "rodadas_executadas": 0, "status_final": f"ERRO: {erro}"})
            if mostrar_progresso:
                print(f"  [{qid}] ERRO inesperado: {erro}")
        else:
            ultima_rodada = resultado["rodadas"][-1] if resultado["rodadas"] else {}
            linhas.append({
                "question_id": qid,
                "rodadas_executadas": len(resultado["rodadas"]),
                "status_final": ultima_rodada.get("status"),
            })
        _salvar_resumo()

    if max_workers <= 1:
        for posicao, qid in enumerate(pendentes, 1):
            if mostrar_progresso:
                print(f"=== [{posicao}/{len(pendentes)}] question_id={qid} ===")
            try:
                resultado = _rodar(qid)
                _registrar_resultado(qid, resultado, None)
            except Exception as erro:  # falhas inesperadas não devem interromper o lote
                _registrar_resultado(qid, None, erro)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futuros = {executor.submit(_rodar, qid): qid for qid in pendentes}
            for futuro in as_completed(futuros):
                qid = futuros[futuro]
                try:
                    resultado = futuro.result()
                    _registrar_resultado(qid, resultado, None)
                except Exception as erro:  # falhas inesperadas não devem interromper o lote
                    _registrar_resultado(qid, None, erro)

    if mostrar_progresso:
        print(f"[loop] resumo do lote salvo em {resumo_path}")
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ids = [int(x) for x in sys.argv[1:]]
    else:
        itens = carregar_itens_refino()
        ids = [int(itens["question_id"].iloc[0])]
    executar_lote(ids)
