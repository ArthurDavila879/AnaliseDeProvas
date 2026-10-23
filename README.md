# Análise de Provas

Mede o **nível de cada questão** e o **nível da prova** a partir das respostas dos alunos, usando a Teoria Clássica dos Testes. Funciona com uma prova simulada, com os microdados do ENEM ou com a prova de um professor em CSV.

> Fases 1 a 4 do projeto (ingestão, teoria clássica, TRI 3PL e app Streamlit). A validação contra os parâmetros oficiais do INEP já está no comando `tri --itens`.

## O que é calculado

**Por questão**

| Métrica | O que mede | Leitura |
|---|---|---|
| Dificuldade (p) | % de alunos que acertou | ≥ 70% fácil · 30–70% média · < 30% difícil |
| Discriminação (D) | acerto dos 27% melhores − acerto dos 27% piores | < 0 problemática · < 0,2 fraca · 0,2–0,3 regular · 0,3–0,4 boa · ≥ 0,4 ótima |
| Ponto-bisserial | correlação entre a questão e a nota no resto da prova | confirma a discriminação |
| KR-20 sem o item | confiabilidade se a questão fosse removida | se sobe, a questão atrapalha |

**Pela prova:** acerto médio (nível da prova), média e desvio das notas, confiabilidade KR-20 e lista de questões com alerta.

Alertas automáticos: discriminação negativa (provável gabarito errado ou enunciado ambíguo), questão que não separa alunos, quase todos acertam/erram, e questão que derruba a confiabilidade.

## Como rodar (Docker)

```bash
docker compose build

# Prova simulada: gera dados com parâmetros conhecidos e confere se o sistema os recupera
docker compose run --rm analise simular

# Prova de exemplo em CSV (12 alunos, 6 questões, a q6 tem problema de propósito)
docker compose run --rm analise csv exemplos/respostas_exemplo.csv --gabarito exemplos/gabarito_exemplo.csv

# TRI (3PL): estima a, b, c de cada questão e θ de cada aluno
docker compose run --rm analise tri                       # prova simulada, compara com os parâmetros verdadeiros
docker compose run --rm analise tri --csv exemplos/respostas_exemplo.csv --gabarito exemplos/gabarito_exemplo.csv
docker compose run --rm analise tri --enem data/MICRODADOS_ENEM_2023.csv --area MT --alunos 10000

# TRI no ENEM + comparação com os parâmetros oficiais do INEP (ITENS_PROVA_AAAA.csv, também em DADOS/ do zip)
docker compose run --rm analise tri --enem data/MICRODADOS_ENEM_2023.csv --itens data/ITENS_PROVA_2023.csv --area MT

# App web (upload de CSV ou prova simulada, gráficos e relatório para baixar)
docker compose up -d app       # abra http://localhost:8501 · para parar: docker compose down

# Testes
docker compose run --rm testes
```

Os resultados vão para `saida/`: `itens.csv` (métricas por questão), `prova.json` (resumo) e, no comando `tri`, `itens_tri.csv` (a, b, c).

### Com os microdados do ENEM

1. Baixe os microdados de um ano em <https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem>.
2. Extraia o `MICRODADOS_ENEM_AAAA.csv` (fica dentro da pasta `DADOS/` do zip) para `data/`.
3. Rode:

```bash
docker compose run --rm analise enem data/MICRODADOS_ENEM_2023.csv --area MT --alunos 10000
```

- `--area`: `MT` (Matemática), `CN` (Natureza) ou `CH` (Humanas). Linguagens fica de fora porque as 5 primeiras questões dependem da língua estrangeira escolhida.
- Cada cor de caderno tem as questões em ordem diferente, então a análise usa um caderno só (por padrão o mais comum; troque com `--caderno`).
- O arquivo tem alguns GB. A leitura é feita em pedaços, mas leva alguns minutos.
- No fim, o programa mostra a correlação entre nº de acertos e a nota oficial TRI.

### Sem Docker

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.cli simular
python -m pytest
```

## Formato da prova em CSV

`respostas.csv`: coluna `aluno` + uma coluna por questão, com a letra marcada.

```csv
aluno,q1,q2,q3
a01,A,C,B
a02,A,B,D
```

`gabarito.csv`:

```csv
questao,gabarito
q1,A
q2,C
q3,B
```

Se as respostas já estiverem corrigidas (0/1), omita `--gabarito`.

## Validação com dados simulados

O comando `simular` gera respostas pelo modelo TRI de 3 parâmetros, com habilidades e parâmetros conhecidos, e inclui questões "defeituosas" (discriminação negativa). Resultado com a seed padrão:

- Correlação entre dificuldade TRI (b) e % de acerto: **−0,97**
- Questões defeituosas detectadas: **2 de 2**
- A correlação entre discriminação TRI (a) e o índice D é mais fraca (~0,4), porque o D também depende da dificuldade: questões muito fáceis ou muito difíceis têm D baixo mesmo discriminando bem. Essa é uma das limitações da teoria clássica que a TRI resolve.

## TRI (3PL)

`src/tri.py` estima o modelo de 3 parâmetros (D = 1,7) por máxima verossimilhança marginal, o mesmo método do BILOG usado pelo INEP: θ ~ N(0, 1) integrado por quadratura, otimização L-BFGS-B com gradiente analítico e priors fracos em b (N(0, 2)) e c (Beta(5, 17)). A habilidade dos alunos sai por EAP, com erro padrão.

Recuperação na prova simulada (2000 alunos, 30 questões, seed padrão): correlação **0,91** em a, **0,997** em b, **0,69** em c e **0,94** em θ, e as 2 questões defeituosas saem com a < 0. O c é sempre o mais difícil de estimar; com 10000 alunos e 45 questões as correlações sobem para 0,98 / 0,998 / 0,84. Com poucos alunos (como na prova de exemplo), o 3PL fica instável e o programa avisa.

### Validação contra o INEP

Com `--itens`, os parâmetros estimados são comparados com os oficiais do `ITENS_PROVA_AAAA.csv`: o item na posição k do caderno (ordem de `CO_POSICAO`) é a questão `qk`. Itens anulados ficam de fora. Como o INEP calibra numa escala de referência própria, o que importa é a **correlação** (Pearson e Spearman). A conversão para a escala do INEP é feita pelo método média-desvio (`b_inep ≈ A·b + B`, `a_inep ≈ a / A`). Saída: `saida/validacao_inep.csv` e `saida/b_vs_inep.png`.

A lib `girth` foi testada, mas não funciona com o numpy 2 / scipy atuais (último release em 2021).

## Estrutura

```
src/
├── ingestao.py    # ENEM e CSV → matriz 0/1 (alunos × questões)
├── simulacao.py   # gerador de provas pelo modelo 3PL
├── classica.py    # dificuldade, discriminação, bisserial, KR-20
├── tri.py         # 3PL: a, b, c por MML e θ por EAP
├── relatorio.py   # classificação, alertas, saída
└── cli.py         # comandos simular | enem | csv | tri
tests/             # pytest
exemplos/          # prova pequena de exemplo
```
