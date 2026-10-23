"""Teoria de Resposta ao Item: estimação do modelo de 3 parâmetros (3PL).

Entrada: matriz 0/1 (linhas = alunos, colunas = questões).

    P(acerto) = c + (1 - c) / (1 + exp(-D * a * (θ - b))),  D = 1.7

Parâmetros das questões: máxima verossimilhança marginal (MML). A habilidade θ
é "integrada" numa grade de pontos (quadratura) com θ ~ N(0, 1), o que também
fixa a escala: média 0 e desvio 1 na população. É o mesmo método do BILOG,
que o INEP usa no ENEM. O gradiente vem da posterior de cada aluno na grade,
o que equivale ao algoritmo EM de Bock-Aitkin, mas otimizado direto com L-BFGS-B.

O 3PL puro diverge com facilidade (c e a sem limite), então são usados priors
fracos, como no BILOG: b ~ N(0, 2) e c ~ Beta(5, 17), média ≈ 0,2 (5 alternativas).
O `a` pode ficar negativo: é assim que a TRI sinaliza uma questão em que
quem sabe mais erra mais.

Habilidade dos alunos: EAP (média da posterior), que existe mesmo para quem
acertou ou errou tudo, ao contrário da máxima verossimilhança.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .simulacao import D, prob_acerto

N_PONTOS = 41  # pontos da quadratura em θ
LIMITES_THETA = (-4.0, 4.0)
LIMITES_A = (-4.0, 4.0)
LIMITES_B = (-5.0, 5.0)
LIMITES_C = (1e-4, 0.5)
PRIOR_B_DESVIO = 2.0
PRIOR_C = (5.0, 17.0)  # Beta(alfa, beta)
EPS = 1e-9


def _quadratura(n_pontos: int = N_PONTOS) -> tuple[np.ndarray, np.ndarray]:
    """Grade de θ e pesos da N(0, 1) normalizados para somar 1."""
    nos = np.linspace(*LIMITES_THETA, n_pontos)
    pesos = stats.norm.pdf(nos)
    return nos, pesos / pesos.sum()


def _posterior(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, nos: np.ndarray, pesos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Posterior de cada aluno na grade de θ.

    Retorna (posterior alunos × pontos, P pontos × questões, log-verossimilhança marginal).
    """
    p = np.clip(prob_acerto(nos, a, b, c), EPS, 1 - EPS)
    log_l = x @ np.log(p).T + (1 - x) @ np.log(1 - p).T + np.log(pesos)[None, :]
    maximo = log_l.max(axis=1, keepdims=True)
    soma = np.exp(log_l - maximo).sum(axis=1, keepdims=True)
    post = np.exp(log_l - maximo) / soma
    return post, p, float((maximo + np.log(soma)).sum())


def _objetivo(params: np.ndarray, x: np.ndarray, nos: np.ndarray, pesos: np.ndarray) -> tuple[float, np.ndarray]:
    """−(log-verossimilhança marginal + log-prior) e seu gradiente."""
    a, b, c = np.split(params, 3)
    post, p, log_vero = _posterior(x, a, b, c, nos, pesos)

    # Contagens esperadas em cada ponto da grade: alunos (n) e acertos (r) por questão.
    n = post.sum(axis=0)[:, None]  # pontos × 1
    r = post.T @ x  # pontos × questões
    dl_dp = r / p - (n - r) / (1 - p)

    sig = (p - c) / (1 - c)
    base = (1 - c) * sig * (1 - sig) * D
    dist = nos[:, None] - b
    grad_a = (dl_dp * base * dist).sum(axis=0)
    grad_b = (dl_dp * -base * a).sum(axis=0) - b / PRIOR_B_DESVIO**2
    grad_c = (dl_dp * (1 - sig)).sum(axis=0) + (PRIOR_C[0] - 1) / c - (PRIOR_C[1] - 1) / (1 - c)

    log_prior = stats.norm.logpdf(b, 0, PRIOR_B_DESVIO).sum() + stats.beta.logpdf(c, *PRIOR_C).sum()
    return -(log_vero + log_prior), -np.concatenate([grad_a, grad_b, grad_c])


def _valores_iniciais(x: np.ndarray) -> np.ndarray:
    """Chute inicial a partir da teoria clássica (sinal da correlação e % de acerto)."""
    p = x.mean(axis=0).clip(0.05, 0.95)
    total = x.sum(axis=1)
    sinal = np.array(
        [np.sign(np.corrcoef(x[:, j], total - x[:, j])[0, 1]) if x[:, j].std() > 0 else 1.0 for j in range(x.shape[1])]
    )
    sinal[sinal == 0] = 1.0
    c = np.full(x.shape[1], 0.2)
    p_sem_chute = ((p - c) / (1 - c)).clip(0.05, 0.95)
    b = (-np.log(p_sem_chute / (1 - p_sem_chute)) / D * sinal).clip(*LIMITES_B)
    return np.concatenate([sinal, b, c])


def estimar_itens(respostas: pd.DataFrame, max_iter: int = 500) -> pd.DataFrame:
    """Estima a, b e c de cada questão por máxima verossimilhança marginal.

    Retorna DataFrame indexado pela questão com colunas a, b, c.
    """
    x = respostas.to_numpy(dtype=float)
    if x.shape[1] < 2:
        raise ValueError("A TRI precisa de pelo menos 2 questões.")
    nos, pesos = _quadratura()
    k = x.shape[1]
    limites = [LIMITES_A] * k + [LIMITES_B] * k + [LIMITES_C] * k
    resultado = optimize.minimize(
        _objetivo,
        _valores_iniciais(x),
        args=(x, nos, pesos),
        jac=True,
        method="L-BFGS-B",
        bounds=limites,
        options={"maxiter": max_iter},
    )
    a, b, c = np.split(resultado.x, 3)
    tabela = pd.DataFrame({"a": a, "b": b, "c": c}, index=pd.Index(respostas.columns, name="questao"))
    tabela.attrs["convergiu"] = bool(resultado.success)
    return tabela


def estimar_theta(respostas: pd.DataFrame, itens: pd.DataFrame) -> pd.DataFrame:
    """Habilidade de cada aluno por EAP, com o erro padrão (desvio da posterior)."""
    x = respostas.to_numpy(dtype=float)
    nos, pesos = _quadratura()
    post, _, _ = _posterior(x, itens["a"].to_numpy(), itens["b"].to_numpy(), itens["c"].to_numpy(), nos, pesos)
    theta = post @ nos
    erro = np.sqrt(post @ nos**2 - theta**2)
    return pd.DataFrame({"theta": theta, "erro_padrao": erro}, index=respostas.index)


def informacao(theta: np.ndarray, itens: pd.DataFrame) -> pd.DataFrame:
    """Informação de Fisher de cada questão em cada θ (3PL; Birnbaum, 1968).

    Diz o quanto a questão "mede bem" alunos daquela habilidade: é alta perto
    de b e cresce com a². O chute (c) reduz a informação. A informação da
    prova é a soma das colunas, e o erro padrão de θ ≈ 1/√informação.

    Retorna DataFrame θ × questões.
    """
    theta = np.asarray(theta, dtype=float)
    a, c = itens["a"].to_numpy(), itens["c"].to_numpy()
    p = np.clip(prob_acerto(theta, a, itens["b"].to_numpy(), c), EPS, 1 - EPS)
    info = (D * a) ** 2 * (1 - p) / p * ((p - c) / (1 - c)) ** 2
    return pd.DataFrame(info, index=pd.Index(theta, name="theta"), columns=itens.index)


def curva_empirica(acertos: pd.Series, theta: pd.Series, n_grupos: int = 10) -> pd.DataFrame:
    """Proporção de acerto observada por faixa de habilidade (quantis de θ).

    Serve para conferir o ajuste: os pontos devem cair perto da curva
    característica do item. Retorna DataFrame com theta (média da faixa),
    proporcao e n por faixa.
    """
    grupos = pd.qcut(theta.rank(method="first"), q=min(n_grupos, len(theta)), labels=False)
    dados = pd.DataFrame({"theta": theta.to_numpy(), "acerto": acertos.to_numpy(), "grupo": grupos.to_numpy()})
    tabela = dados.groupby("grupo").agg(theta=("theta", "mean"), proporcao=("acerto", "mean"), n=("acerto", "size"))
    return tabela.reset_index(drop=True)


def comparar_com_inep(estimados: pd.DataFrame, oficiais: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compara os parâmetros estimados com os oficiais do INEP (ITENS_PROVA).

    O INEP calibra numa escala de referência própria (não a N(0, 1) da nossa
    amostra), então b e a só batem a menos de uma transformação linear:
    b_inep ≈ A·b + B e a_inep ≈ a / A. A e B saem pelo método média-desvio
    (média e desvio dos b iguais nas duas escalas); c não muda com a escala.

    Retorna (tabela por questão, resumo com correlações e constantes A, B).
    """
    tabela = estimados.join(oficiais, how="inner")
    validos = ~tabela["anulado"] & tabela[["a_inep", "b_inep", "c_inep"]].notna().all(axis=1)
    v = tabela[validos]
    if len(v) < 3:
        raise ValueError("Menos de 3 itens com parâmetros oficiais: não dá para comparar.")

    escala_a = float(v["b_inep"].std() / v["b"].std())
    escala_b = float(v["b_inep"].mean() - escala_a * v["b"].mean())
    tabela["b_convertido"] = escala_a * tabela["b"] + escala_b
    tabela["a_convertido"] = tabela["a"] / escala_a
    tabela["dif_b"] = tabela["b_convertido"] - tabela["b_inep"]

    resumo = {
        "n_itens": len(tabela),
        "n_comparados": int(validos.sum()),
        "anulados": tabela.index[tabela["anulado"]].tolist(),
        "escala_A": round(escala_a, 4),
        "escala_B": round(escala_b, 4),
    }
    for p in ("a", "b", "c"):
        resumo[f"pearson_{p}"] = round(float(np.corrcoef(v[p], v[f"{p}_inep"])[0, 1]), 4)
        resumo[f"spearman_{p}"] = round(float(stats.spearmanr(v[p], v[f"{p}_inep"]).statistic), 4)
    resumo["rmse_b_convertido"] = round(float(np.sqrt((tabela.loc[validos, "dif_b"] ** 2).mean())), 4)
    return tabela, resumo


def ajustar_3pl(respostas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atalho: (parâmetros das questões, habilidade dos alunos)."""
    itens = estimar_itens(respostas)
    return itens, estimar_theta(respostas, itens)
