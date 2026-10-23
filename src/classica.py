"""Métricas da Teoria Clássica dos Testes (TCT).

Entrada: matriz 0/1 (linhas = alunos, colunas = questões).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRACAO_GRUPO = 0.27  # 27% superior vs 27% inferior (Kelley, 1939)


def dificuldade(respostas: pd.DataFrame) -> pd.Series:
    """Índice de dificuldade (p): proporção de acertos. Quanto MAIOR, mais FÁCIL."""
    return respostas.mean().rename("dificuldade")


def discriminacao_grupos(respostas: pd.DataFrame, fracao: float = FRACAO_GRUPO) -> pd.Series:
    """Índice D: acerto do grupo superior menos acerto do grupo inferior.

    Os grupos são definidos pela nota total na própria prova. É exatamente a
    ideia de "se só os bons acertam, a questão separa bem quem sabe".
    """
    total = respostas.sum(axis=1)
    n = max(1, int(round(len(respostas) * fracao)))
    ordem = total.sort_values(kind="mergesort").index
    inferior = respostas.loc[ordem[:n]]
    superior = respostas.loc[ordem[-n:]]
    return (superior.mean() - inferior.mean()).rename("discriminacao")


def ponto_bisserial(respostas: pd.DataFrame) -> pd.Series:
    """Correlação ponto-bisserial corrigida: item vs. total SEM o próprio item.

    Tirar o item do total evita que ele "se correlacione consigo mesmo".
    """
    total = respostas.sum(axis=1)
    valores = {}
    for col in respostas.columns:
        item = respostas[col]
        resto = total - item
        if item.std() == 0 or resto.std() == 0:
            valores[col] = np.nan  # todo mundo acertou/errou: não dá para correlacionar
        else:
            valores[col] = float(np.corrcoef(item, resto)[0, 1])
    return pd.Series(valores, name="bisserial")


def kr20(respostas: pd.DataFrame) -> float:
    """Confiabilidade KR-20 (alfa de Cronbach para itens 0/1). Vai de 0 a 1."""
    k = respostas.shape[1]
    if k < 2:
        return float("nan")
    p = respostas.mean()
    var_total = respostas.sum(axis=1).var(ddof=0)
    if var_total == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / var_total))


def kr20_sem_item(respostas: pd.DataFrame) -> pd.Series:
    """KR-20 da prova se cada questão fosse removida.

    Se a confiabilidade SOBE ao remover a questão, ela está atrapalhando.
    """
    return pd.Series(
        {col: kr20(respostas.drop(columns=col)) for col in respostas.columns},
        name="kr20_sem_item",
    )


def analisar_itens(respostas: pd.DataFrame) -> pd.DataFrame:
    """Tabela com todas as métricas por questão."""
    tabela = pd.concat(
        [
            dificuldade(respostas),
            discriminacao_grupos(respostas),
            ponto_bisserial(respostas),
            kr20_sem_item(respostas),
        ],
        axis=1,
    )
    tabela.index.name = "questao"
    return tabela
