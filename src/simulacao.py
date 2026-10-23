"""Geração de provas simuladas com o modelo TRI de 3 parâmetros (3PL).

Serve para testar o sistema: como você define a "verdade" (habilidade dos
alunos e parâmetros das questões), dá para conferir se as métricas recuperam ela.

    P(acerto) = c + (1 - c) / (1 + exp(-D * a * (θ - b)))

    θ: habilidade do aluno     a: discriminação
    b: dificuldade da questão  c: chance de acerto no chute
    D = 1.7 (constante que aproxima a curva logística da normal)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

D = 1.7


def prob_acerto(theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Matriz (alunos x questões) com a probabilidade de acerto."""
    z = D * a[None, :] * (theta[:, None] - b[None, :])
    return c[None, :] + (1 - c[None, :]) / (1 + np.exp(-z))


def simular_prova(
    n_alunos: int = 2_000,
    n_itens: int = 30,
    n_problematicas: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Simula respostas de uma prova de múltipla escolha.

    `n_problematicas`: questões com discriminação negativa (quem sabe mais
    erra mais) — imitam gabarito errado ou enunciado ambíguo.

    Retorna (matriz de acertos, parâmetros verdadeiros, habilidade verdadeira).
    """
    rng = np.random.default_rng(seed)

    theta = rng.normal(0, 1, n_alunos)
    a = rng.lognormal(mean=0.0, sigma=0.3, size=n_itens)      # ~0.5 a 2
    b = rng.uniform(-2.0, 2.0, size=n_itens)                  # fácil → difícil
    c = rng.uniform(0.10, 0.25, size=n_itens)                 # 5 alternativas ≈ 0.2

    problematicas = rng.choice(n_itens, size=min(n_problematicas, n_itens), replace=False)
    a[problematicas] = -rng.uniform(0.6, 1.0, size=len(problematicas))
    b[problematicas] = rng.uniform(-0.5, 0.5, size=len(problematicas))

    p = prob_acerto(theta, a, b, c)
    acertos = (rng.random(p.shape) < p).astype(np.int8)

    colunas = [f"q{i + 1:02d}" for i in range(n_itens)]
    respostas = pd.DataFrame(acertos, columns=colunas)
    respostas.index.name = "aluno"

    parametros = pd.DataFrame(
        {"a": a, "b": b, "c": c, "problematica": np.isin(np.arange(n_itens), problematicas)},
        index=pd.Index(colunas, name="questao"),
    )
    return respostas, parametros, pd.Series(theta, name="theta")
