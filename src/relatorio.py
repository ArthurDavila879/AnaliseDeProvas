"""Classificação das questões e da prova + saída dos resultados."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import classica


def nivel_questao(p: float) -> str:
    """Faixas usuais de dificuldade pela proporção de acerto."""
    if np.isnan(p):
        return "indefinido"
    if p >= 0.70:
        return "fácil"
    if p >= 0.30:
        return "média"
    return "difícil"


def qualidade_discriminacao(d: float) -> str:
    """Faixas de Ebel (1965) para o índice D."""
    if np.isnan(d):
        return "indefinida"
    if d < 0:
        return "problemática"
    if d < 0.20:
        return "fraca"
    if d < 0.30:
        return "regular"
    if d < 0.40:
        return "boa"
    return "ótima"


def alertas(linha: pd.Series, kr20_prova: float) -> str:
    msgs = []
    if linha["discriminacao"] < 0:
        msgs.append("alunos fracos acertam mais que os fortes: revisar gabarito/enunciado")
    elif linha["discriminacao"] < 0.20:
        msgs.append("não separa quem sabe de quem não sabe")
    if linha["dificuldade"] >= 0.95:
        msgs.append("quase todos acertaram")
    elif linha["dificuldade"] <= 0.10:
        msgs.append("quase ninguém acertou")
    if not np.isnan(linha["kr20_sem_item"]) and linha["kr20_sem_item"] > kr20_prova + 0.005:
        msgs.append("remover aumenta a confiabilidade da prova")
    return "; ".join(msgs)


def classificar_itens(respostas: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    itens = classica.analisar_itens(respostas)
    confiab = classica.kr20(respostas)
    itens["nivel"] = itens["dificuldade"].apply(nivel_questao)
    itens["qualidade"] = itens["discriminacao"].apply(qualidade_discriminacao)
    itens["alertas"] = itens.apply(alertas, axis=1, kr20_prova=confiab)
    return itens, confiab


def interpretar_kr20(valor: float) -> str:
    if np.isnan(valor):
        return "indefinida"
    if valor >= 0.90:
        return "excelente"
    if valor >= 0.80:
        return "boa"
    if valor >= 0.70:
        return "aceitável"
    if valor >= 0.60:
        return "questionável"
    return "baixa"


def resumir_prova(respostas: pd.DataFrame, itens: pd.DataFrame, confiab: float) -> dict:
    total = respostas.sum(axis=1)
    p_media = float(itens["dificuldade"].mean())
    return {
        "n_alunos": int(len(respostas)),
        "n_questoes": int(respostas.shape[1]),
        "acerto_medio": round(p_media, 4),
        "nivel_prova": nivel_questao(p_media),
        "nota_media": round(float(total.mean()), 2),
        "nota_desvio": round(float(total.std(ddof=0)), 2),
        "kr20": round(confiab, 4),
        "confiabilidade": interpretar_kr20(confiab),
        "distribuicao_niveis": itens["nivel"].value_counts().to_dict(),
        "distribuicao_qualidade": itens["qualidade"].value_counts().to_dict(),
        "questoes_com_alerta": itens.index[itens["alertas"] != ""].tolist(),
    }


def salvar(itens: pd.DataFrame, resumo: dict, pasta: str | Path) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    itens.round(4).to_csv(pasta / "itens.csv", encoding="utf-8")
    (pasta / "prova.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return pasta


def grafico_b_inep(tabela: pd.DataFrame, resumo: dict, caminho: str | Path, n_rotulos: int = 4) -> Path:
    """Dispersão do b estimado vs. b oficial do INEP, com a reta média-desvio.

    Rotula só os itens mais distantes da reta (e os de discriminação negativa).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tinta, tinta_2, fundo, serie = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
    v = tabela[~tabela["anulado"] & tabela["b_inep"].notna()]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150, facecolor=fundo)
    ax.set_facecolor(fundo)
    ax.scatter(v["b"], v["b_inep"], s=55, color=serie, edgecolors=fundo, linewidths=1.5, zorder=3)

    x = np.linspace(v["b"].min(), v["b"].max(), 2)
    ax.plot(x, resumo["escala_A"] * x + resumo["escala_B"], color=tinta_2, lw=1.5, ls="--", zorder=2)
    ax.annotate(f"b_inep = {resumo['escala_A']:.2f}·b {resumo['escala_B']:+.2f}", (x[1], resumo["escala_A"] * x[1] + resumo["escala_B"]),
                textcoords="offset points", xytext=(-4, 8), ha="right", color=tinta_2, fontsize=9)

    destaque = set(v["dif_b"].abs().nlargest(n_rotulos).index) | set(v.index[v["a"] < 0])
    for q in destaque:
        ax.annotate(q, (v.at[q, "b"], v.at[q, "b_inep"]), textcoords="offset points", xytext=(6, -3), color=tinta, fontsize=9)

    ax.set_title(f"Dificuldade (b): estimada vs. oficial do INEP\nPearson {resumo['pearson_b']:.3f} · "
                 f"Spearman {resumo['spearman_b']:.3f} · {resumo['n_comparados']} itens",
                 loc="left", color=tinta, fontsize=11)
    ax.set_xlabel("b estimado (src/tri.py, escala da amostra)", color=tinta_2)
    ax.set_ylabel("b oficial (ITENS_PROVA)", color=tinta_2)
    ax.grid(color="#e4e3df", lw=0.8, zorder=0)
    ax.tick_params(colors=tinta_2)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c8c2")

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=fundo)
    plt.close(fig)
    return caminho


def imprimir(itens: pd.DataFrame, resumo: dict) -> None:
    print("\n=== PROVA ===")
    print(f"Alunos: {resumo['n_alunos']} | Questões: {resumo['n_questoes']}")
    print(f"Acerto médio: {resumo['acerto_medio']:.1%} → prova {resumo['nivel_prova'].upper()}")
    print(f"Nota média: {resumo['nota_media']} ± {resumo['nota_desvio']}")
    print(f"Confiabilidade (KR-20): {resumo['kr20']:.3f} ({resumo['confiabilidade']})")
    print(f"Níveis: {resumo['distribuicao_niveis']}")
    print(f"Discriminação: {resumo['distribuicao_qualidade']}")

    print("\n=== QUESTÕES ===")
    tabela = itens[["dificuldade", "discriminacao", "bisserial", "nivel", "qualidade"]].copy()
    tabela["dificuldade"] = tabela["dificuldade"].map("{:.0%}".format)
    tabela[["discriminacao", "bisserial"]] = tabela[["discriminacao", "bisserial"]].round(2)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(tabela.to_string())

    com_alerta = itens[itens["alertas"] != ""]
    if not com_alerta.empty:
        print("\n=== ALERTAS ===")
        for q, linha in com_alerta.iterrows():
            print(f"{q}: {linha['alertas']}")
