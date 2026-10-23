"""Leitura de dados e conversão para a matriz padrão de acertos.

Formato padrão usado no resto do projeto:
    DataFrame com uma linha por aluno e uma coluna por questão,
    valores 0 (errou/branco) ou 1 (acertou).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Linguagens (LC) ficou de fora: as 5 primeiras questões dependem da língua
# estrangeira escolhida, o que exige um tratamento à parte.
AREAS_ENEM = {"MT": "Matemática", "CN": "Ciências da Natureza", "CH": "Ciências Humanas"}


def corrigir(respostas: pd.Series, gabarito: str | pd.Series) -> pd.DataFrame:
    """Compara strings de respostas com o gabarito e devolve a matriz 0/1.

    `respostas`: uma string por aluno (ex.: "ABCDE.*A..."). Branco ('.') e
    dupla marcação ('*') contam como erro.
    `gabarito`: string única ou uma por aluno.
    """
    respostas = respostas.astype(str).reset_index(drop=True)
    if isinstance(gabarito, str):
        gabarito = pd.Series([gabarito] * len(respostas))
    gabarito = gabarito.astype(str).reset_index(drop=True)

    tamanho = gabarito.str.len().mode().iloc[0]
    validos = (respostas.str.len() == tamanho) & (gabarito.str.len() == tamanho)
    if not validos.all():
        respostas, gabarito = respostas[validos], gabarito[validos]

    r = np.array([list(s) for s in respostas])
    g = np.array([list(s) for s in gabarito])
    acertos = (r == g).astype(np.int8)

    colunas = [f"q{i + 1:02d}" for i in range(tamanho)]
    return pd.DataFrame(acertos, columns=colunas, index=respostas.index)


def carregar_enem(
    caminho: str | Path,
    area: str = "MT",
    n_alunos: int | None = 10_000,
    co_prova: int | None = None,
    seed: int = 42,
    chunksize: int = 250_000,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Lê o CSV de microdados do ENEM em pedaços e devolve uma amostra corrigida.

    Cada cor de caderno (CO_PROVA) tem as questões em ordem diferente, então a
    análise usa um caderno só. Se `co_prova` não for informado, usa o mais comum.

    Retorna (matriz de acertos, nota oficial TRI de cada aluno, co_prova usado).
    """
    area = area.upper()
    if area not in AREAS_ENEM:
        raise ValueError(f"Área '{area}' não suportada. Use uma de: {', '.join(AREAS_ENEM)}")

    col_prova = f"CO_PROVA_{area}"
    col_presenca = f"TP_PRESENCA_{area}"
    col_resp = f"TX_RESPOSTAS_{area}"
    col_gab = f"TX_GABARITO_{area}"
    col_nota = f"NU_NOTA_{area}"

    leitura = dict(sep=";", encoding="latin-1", chunksize=chunksize, dtype=str)

    # 1ª passada (barata): descobrir o caderno mais comum entre os presentes.
    if co_prova is None:
        contagem: pd.Series | None = None
        for chunk in pd.read_csv(caminho, usecols=[col_prova, col_presenca], **leitura):
            presentes = chunk[pd.to_numeric(chunk[col_presenca], errors="coerce") == 1]
            c = pd.to_numeric(presentes[col_prova], errors="coerce").value_counts()
            contagem = c if contagem is None else contagem.add(c, fill_value=0)
        if contagem is None or contagem.empty:
            raise ValueError(f"Nenhum participante presente encontrado para {area}.")
        co_prova = int(contagem.idxmax())

    # 2ª passada: só as linhas desse caderno.
    colunas = [col_prova, col_presenca, col_resp, col_gab, col_nota]
    pedacos = []
    for chunk in pd.read_csv(caminho, usecols=colunas, **leitura):
        filtro = (pd.to_numeric(chunk[col_presenca], errors="coerce") == 1) & (
            pd.to_numeric(chunk[col_prova], errors="coerce") == co_prova
        )
        pedacos.append(chunk.loc[filtro, [col_resp, col_gab, col_nota]].dropna(subset=[col_resp, col_gab]))
    dados = pd.concat(pedacos, ignore_index=True)

    if dados.empty:
        raise ValueError(f"Nenhum aluno encontrado para o caderno {co_prova} em {area}.")
    if n_alunos and len(dados) > n_alunos:
        dados = dados.sample(n=n_alunos, random_state=seed).reset_index(drop=True)

    matriz = corrigir(dados[col_resp], dados[col_gab])
    nota = pd.to_numeric(dados.loc[matriz.index, col_nota], errors="coerce").rename("nota_oficial")
    return matriz.reset_index(drop=True), nota.reset_index(drop=True), co_prova


def carregar_csv(caminho_respostas: str | Path, caminho_gabarito: str | Path | None = None) -> pd.DataFrame:
    """Lê uma prova no formato simples (para uso de professor).

    respostas.csv: coluna `aluno` + uma coluna por questão.
        - Com gabarito: as células são letras (A, B, C...).
        - Sem gabarito: as células já são 0/1.
    gabarito.csv: colunas `questao,gabarito` (nomes iguais às colunas de respostas).
    """
    resp = pd.read_csv(caminho_respostas, dtype=str)
    if "aluno" in resp.columns:
        resp = resp.set_index("aluno")

    if caminho_gabarito is None:
        matriz = resp.apply(pd.to_numeric, errors="coerce")
        if not matriz.isin([0, 1]).all().all():
            raise ValueError("Sem gabarito, todas as células precisam ser 0 ou 1.")
        return matriz.astype(np.int8)

    gab = pd.read_csv(caminho_gabarito, dtype=str).set_index("questao")["gabarito"]
    faltando = set(resp.columns) - set(gab.index)
    if faltando:
        raise ValueError(f"Questões sem gabarito: {sorted(faltando)}")

    normalizado = resp.apply(lambda col: col.str.strip().str.upper())
    esperado = gab.str.strip().str.upper()
    # Compara por posição: o alinhamento do pandas 3 quebra com índices de nomes diferentes.
    acertos = normalizado.to_numpy() == esperado[resp.columns].to_numpy()
    return pd.DataFrame(acertos, index=resp.index, columns=resp.columns).astype(np.int8)
