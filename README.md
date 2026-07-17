# Competição 2 - IV_WAPLA

Pipeline de diagnóstico, avaliação e correção automatizada de itens de múltipla escolha, desenvolvido para a **Competição 2 - Qualidade de questões geradas com IA generativa** do IV Workshop de Aplicações Práticas de Learning Analytics em Instituições de Ensino no Brasil (WAPLA 2026).

Parte de uma base de itens rejeitados em uma avaliação anterior e combina verificações determinísticas com avaliações de um modelo de linguagem para diagnosticar problemas, selecionar os itens com maior potencial de refino e corrigi-los de forma iterativa.

## Pipeline

```text
base de itens → diagnóstico → seleção → correção iterativa (loop) → análise dos resultados
```

1. **Diagnóstico** - 19 critérios organizados em três eixos (`item_diagnostic_agent.py`):
  - **FE** (Formulação e Estrutura, 16 critérios) - clareza, consistência, pistas indevidas.
  - **AP** (Alinhamento Pedagógico, 2 critérios) - Taxonomia de Bloom e compatibilidade de formato.
  - **APL** (Aplicação, 1 critério) - concentração da letra do gabarito no conjunto.
2. **Seleção** - regras de elegibilidade e `score_refino` priorizam os itens com problemas identificáveis e correção viável.
3. **Correção iterativa** - para cada item, até N rodadas de **avaliar (LLM) → diagnosticar (código) → corrigir (LLM)**, encerrando antes se o item deixar de ter critérios ativos.
4. **Análise dos resultados** - compara diagnóstico inicial e final, mede efetividade das correções por critério e identifica regressões.

## Estrutura do repositório

| Arquivo | Responsabilidade |
| --- | --- |
| `analise_e_correcao.ipynb` | Notebook principal e reprodutível: reúne todas as etapas acima, com análises exploratórias, métodos, visualizações e evidências. |
| `item_diagnostic_agent.py` | Diagnóstico determinístico (regras/regex) pelos 19 critérios dos eixos FE/AP/APL; monta a seleção para refino. |
| `item_quality_analyzer.py` | Avaliação pedagógica de um item via LLM (resolução independente, coerência, plausibilidade dos distratores, alinhamento de Bloom). |
| `item_correction_builder.py` | Junta diagnóstico de código + avaliação LLM por item e monta o prompt de correção modular (só com os critérios ainda ativos). |
| `loop_correcao_avaliacao.py` | Orquestra o ciclo avaliar → diagnosticar → corrigir por até `n_iteracoes` rodadas, item a item ou em lote; salva o histórico incrementalmente. |
| `base_text_classifier.py` | Classifica via LLM se o texto-base de uma questão é de fato necessário para respondê-la. |
| `math_tool.py` | Ferramenta de cálculo simbólico (`sympy`) exposta ao LLM via function calling, para verificar gabaritos matemáticos. |
| `llm_client.py` | Cliente HTTP para a API de chat completions da Maritaca AI. |
| `prompts.py` / `prompts/` | Templates e montagem dos prompts (classificação de texto-base, avaliação, correção). |

## Dados

O diretório `data/` contém a base de entrada (`rejected_questions.csv`), os caches de classificação/avaliação (`text_base_classification_results.csv`, `analises_maritaca.csv`) e a seleção para refino (`itens_para_refino_top100.csv`) usada pelo CLI.

Como o processo de correção pode ser executado com diferentes modelos, cada corrida grava seu diagnóstico e histórico de correção numa subpasta nomeada pelo modelo/versão usado:

- `data/sabia-4-thinking/` - modelo `sabia-4-thinking` (raciocínio estendido); é a corrida referenciada por `PASTA_DIAGNOSTICO`/`PASTA_CORRECAO` no notebook, cujos resultados são discutidos nas Partes 4-7.
  
- `data/sabia-4/` - duas corridas com o modelo `sabia-4` (`loop_correcao_v1`, `loop_correcao_v2`),  usadas na Seção 6.8 para comparar aprovação item a item entre modelos/versões.

Cada subpasta de correção tem um arquivo `<question_id>.json` por item (histórico rodada a rodada) e relatórios agregados (`relatorio_*.csv`, `_resumo_lote.csv`).

## Configuração

Crie um arquivo `.env` na raiz do projeto com:

```env
MARITACA_API_KEY=<sua chave>
MARITACA_MODEL=sabia-4    # opcional; default definido em item_quality_analyzer.criar_modelo
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Para rodar o notebook (`analise_e_correcao.ipynb`), também são necessários `matplotlib`, `scipy`, `sympy`, `langchain-openai` e `jupyter`.

## Como rodar

- **Notebook**: abra `analise_e_correcao.ipynb` e execute as células em ordem - cada parte é independente e autoexplicativa.
- **Linha de comando**: `python loop_correcao_avaliacao.py <question_id> [question_id...]` roda o loop de correção para um ou mais itens (sem argumentos, usa o primeiro item de `data/itens_para_refino_top100.csv`).
