# CLAUDE.md

## Objetivo

Medir o **nível de cada questão** e o **nível da prova** a partir das respostas dos alunos, com a Teoria Clássica dos Testes (dificuldade, discriminação D, ponto-bisserial, KR-20). Fontes de dados: prova simulada (modelo TRI 3PL), microdados do ENEM e CSV de professor.

Fases 1 e 2 (ingestão + teoria clássica) estão prontas. Próximas: TRI (3PL) validada contra os parâmetros oficiais do INEP e app Streamlit.

## Estrutura

```
src/
├── ingestao.py    # ENEM e CSV → matriz 0/1 (corrigir, carregar_enem, carregar_csv)
├── simulacao.py   # gerador de provas pelo 3PL, com parâmetros verdadeiros conhecidos
├── classica.py    # métricas por questão + KR-20 (entrada: matriz 0/1)
├── relatorio.py   # faixas (nível, qualidade, KR-20), alertas, resumo, saída
└── cli.py         # python -m src.cli <simular|enem|csv>
tests/test_analise.py   # pytest, agrupado por módulo (ingestão, métricas, classificação, simulação)
exemplos/          # prova de 12 alunos × 6 questões; a q6 é ruim de propósito
data/              # microdados do ENEM (fora do git, montado no Docker)
saida/             # itens.csv, prova.json, validacao_simulacao.csv (fora do git)
```

Fluxo: `ingestao`/`simulacao` → matriz 0/1 → `relatorio.classificar_itens` (usa `classica`) → `resumir_prova` → `imprimir`/`salvar`.

## Como rodar

### Docker

```bash
docker compose build                       # obrigatório após mudar src/, tests/ ou exemplos/ (são copiados na imagem)
docker compose run --rm testes             # pytest -v
docker compose run --rm analise simular
docker compose run --rm analise csv exemplos/respostas_exemplo.csv --gabarito exemplos/gabarito_exemplo.csv
docker compose run --rm analise enem data/MICRODADOS_ENEM_2023.csv --area MT --alunos 10000
```

Só `data/` e `saida/` são volumes. O resto vem da imagem, então sem `build` o container roda o código antigo.

### Sem Docker

```bash
python -m venv .venv && .venv\Scripts\activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python -m src.cli simular
```

No terminal do Windows, defina `PYTHONIOENCODING=utf-8`, senão o `imprimir` quebra em caracteres como `→`.

## Convenções

- **Português** em tudo: nomes de funções, variáveis, colunas, docstrings, comentários, mensagens de erro e saída da CLI (`dificuldade`, `respostas`, `"Questões sem gabarito: ..."`).
- **Formato padrão = matriz 0/1 alunos × questões**: `DataFrame` com uma linha por aluno e uma coluna por questão, valores `0` (erro/branco/dupla marcação) ou `1` (acerto), `dtype int8`. Toda fonte de dados é convertida para esse formato em `ingestao.py` ou `simulacao.py`, e todo o resto (`classica`, `relatorio`) recebe só a matriz. Colunas geradas seguem `q01, q02, ...`. Métricas por questão devolvem `pd.Series` indexada pela questão e com `name` definido.
- **Toda função nova tem teste em pytest** em `tests/`, na seção do módulo correspondente. Use dados pequenos montados à mão, com resultado conhecido, e `simulacao.simular_prova(seed=...)` quando o teste for estatístico.
- Estilo: `from __future__ import annotations`, type hints, docstring curta explicando *o que a métrica significa*, constantes com referência (ex.: `FRACAO_GRUPO = 0.27  # Kelley, 1939`).
- Erros de entrada viram `ValueError` com mensagem clara em português.

## Cuidados

- **pandas 3**: o `requirements.txt` aceita `pandas>=2.2`, e a imagem instala a 3.x. Evite `DataFrame.eq(Series)` e outras operações que alinham por rótulo quando os índices têm nomes diferentes (isso já quebrou `carregar_csv`). Compare por posição com `.to_numpy()`.
- **Microdados do ENEM**: `;` como separador, `latin-1`, vários GB. Leia em pedaços (`chunksize`) e só com as colunas necessárias. Cada cor de caderno (`CO_PROVA_*`) tem outra ordem de questões, então a análise usa um caderno só. Linguagens (LC) fica de fora (língua estrangeira).
- `classica.discriminacao_grupos` e `ponto_bisserial` usam o total da própria prova. Bisserial e KR-20 retornam `NaN` quando não há variância. Trate `NaN` nas funções de `relatorio`.
