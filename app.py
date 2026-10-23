"""App web (Streamlit): nível de cada questão e da prova, pela Teoria Clássica e pela TRI.

    streamlit run app.py          # ou: docker compose up app → http://localhost:8501

Toda a análise vem de src/; aqui ficam só a interface e os gráficos. O tema (claro/escuro)
vem de .streamlit/config.toml; os gráficos e o cartão da prova leem o tema ativo e usam a
paleta correspondente.
"""

from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ingestao, relatorio, simulacao, tri

EXEMPLOS = Path(__file__).resolve().parent / "exemplos"
FONTE = '"Atkinson Hyperlegible Next", "Atkinson Hyperlegible", sans-serif'  # a mesma do config.toml
# Contorno dos marcadores: cinza que separa pontos sobrepostos nos dois temas. Não usa o fundo do
# tema porque o Streamlit só roda o script de novo na próxima interação depois de trocar o tema.
CONTORNO = "rgba(128, 128, 128, 0.55)"


@dataclass(frozen=True)
class Paleta:
    """Cores de um tema. Identidade: caneta azul nos dados, caneta vermelha só no que precisa de revisão."""

    serie: str  # dados (caneta azul)
    serie_2: str  # segunda série (acerto observado)
    alerta: str  # caneta vermelha: sempre junto de ícone ou texto, nunca só a cor
    referencia: str  # linhas de referência e rótulos auxiliares
    contexto: str  # curvas das outras questões
    preenchimento: str  # área sob a curva de informação
    fundo_alerta: str  # linha da tabela com alerta
    bolha_vazia: str  # 0% de acerto: bolha em branco no cartão
    bolha_cheia: str  # 100% de acerto: bolha toda preenchida
    bolha_borda: str


PALETAS = {
    "light": Paleta(
        serie="#2a78d6",
        serie_2="#eb6834",
        alerta="#d03b3b",
        referencia="#6b7280",
        contexto="rgba(91, 98, 112, 0.25)",
        preenchimento="rgba(42, 120, 214, 0.12)",
        fundo_alerta="background-color: rgba(208, 59, 59, 0.12)",
        bolha_vazia="#ffffff",
        bolha_cheia="#184f95",
        bolha_borda="#c3cad5",
    ),
    "dark": Paleta(
        serie="#3987e5",
        serie_2="#d95926",
        alerta="#e66767",
        referencia="#9aa3b2",
        contexto="rgba(154, 163, 178, 0.22)",
        preenchimento="rgba(57, 135, 229, 0.18)",
        fundo_alerta="background-color: rgba(230, 103, 103, 0.18)",
        bolha_vazia="#1b2129",
        bolha_cheia="#86b6ef",
        bolha_borda="#3a434f",
    ),
}

ORDEM_NIVEIS = ["fácil", "média", "difícil"]
MIN_ALUNOS_TRI = 500  # abaixo disso o 3PL fica instável (mesmo aviso da CLI)
MIN_ALUNOS_CURVA_EMPIRICA = 200
MAX_ALUNOS_SIMULACAO = 10_000  # o 3PL leva ~10 s nisso; mais que isso trava o deploy gratuito
GRADE_THETA = np.linspace(-4, 4, 161)

NOMES_COLUNAS = {
    "dificuldade": "Acerto (p)",
    "nivel": "Nível",
    "discriminacao": "Discriminação (D)",
    "qualidade": "Qualidade",
    "bisserial": "Ponto-bisserial",
    "kr20_sem_item": "KR-20 sem a questão",
    "a": "a (TRI)",
    "b": "b (TRI)",
    "c": "c (TRI)",
    "alertas": "Alertas",
}


def paleta() -> Paleta:
    """Paleta do tema ativo (claro quando o tema é desconhecido, como nos testes)."""
    return PALETAS.get(st.context.theme.type or "light", PALETAS["light"])


# ---------- dados (com cache: só recalcula quando a prova muda) ----------


@st.cache_data(show_spinner="Simulando a prova...")
def carregar_simulada(n_alunos: int, n_questoes: int, n_problematicas: int, seed: int) -> pd.DataFrame:
    respostas, _, _ = simulacao.simular_prova(n_alunos, n_questoes, n_problematicas, seed)
    return respostas


@st.cache_data(show_spinner="Lendo os arquivos...")
def carregar_upload(respostas: bytes, gabarito: bytes | None) -> pd.DataFrame:
    return ingestao.carregar_csv(io.BytesIO(respostas), io.BytesIO(gabarito) if gabarito else None)


@st.cache_data(show_spinner="Calculando as métricas clássicas...")
def analisar_classica(respostas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    itens, confiab = relatorio.classificar_itens(respostas)
    return itens, relatorio.resumir_prova(respostas, itens, confiab)


@st.cache_data(show_spinner="Estimando a TRI (3PL)...")
def analisar_tri(respostas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    itens, alunos = tri.ajustar_3pl(respostas)
    return itens, alunos, itens.attrs["convergiu"]


def tabela_relatorio(itens: pd.DataFrame, itens_tri: pd.DataFrame) -> pd.DataFrame:
    """Métricas clássicas + parâmetros TRI por questão (é o que vai para o CSV)."""
    return itens.join(itens_tri)[list(NOMES_COLUNAS)]


# ---------- barra lateral ----------


def _usar(fonte: str) -> None:
    st.session_state["fonte"] = fonte


def barra_lateral() -> tuple[pd.DataFrame | None, str]:
    """Escolha da prova. Retorna (matriz 0/1 ou None, descrição da origem)."""
    st.sidebar.header("Sua prova")
    arq_resp = st.sidebar.file_uploader(
        "Respostas dos alunos (CSV)",
        type="csv",
        on_change=_usar,
        args=("csv",),
        help="Coluna `aluno` + uma coluna por questão. Células com letras (A–E) se enviar o gabarito; 0/1 se não.",
    )
    arq_gab = st.sidebar.file_uploader(
        "Gabarito (CSV, opcional)",
        type="csv",
        on_change=_usar,
        args=("csv",),
        help="Colunas `questao,gabarito`, com os mesmos nomes de questão das respostas.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Sem um arquivo à mão?")
    with st.sidebar.expander("Ajustar a simulação"):
        n_alunos = st.number_input("Alunos", 100, MAX_ALUNOS_SIMULACAO, 2_000, step=100, key="sim_alunos")
        n_questoes = st.number_input("Questões", 2, 90, 30, key="sim_questoes")
        n_problematicas = st.number_input(
            "Questões com problema",
            0,
            10,
            2,
            key="sim_problematicas",
            help="Discriminação negativa: imitam gabarito errado ou enunciado ambíguo.",
        )
        seed = st.number_input("Semente", 0, 10_000, 42, key="sim_seed")
    st.sidebar.button(
        "Usar prova simulada", type="primary", on_click=_usar, args=("simulada",), key="usar_simulada", width="stretch"
    )
    st.sidebar.caption("Tema claro ou escuro: no menu ⋮, no canto superior direito.")

    fonte = st.session_state.get("fonte")
    if fonte == "simulada":
        respostas = carregar_simulada(int(n_alunos), int(n_questoes), int(n_problematicas), int(seed))
        return respostas, f"prova simulada pelo modelo 3PL (semente {seed})"
    if fonte == "csv" and arq_resp is not None:
        try:
            respostas = carregar_upload(arq_resp.getvalue(), arq_gab.getvalue() if arq_gab else None)
        except (ValueError, KeyError) as erro:
            st.error(f"Não consegui ler a prova: {erro}")
            return None, ""
        origem = arq_resp.name + (f" com o gabarito {arq_gab.name}" if arq_gab else " (já corrigida, em 0/1)")
        return respostas, origem
    return None, ""


# ---------- cartão da prova ----------


def _misturar(cor_0: str, cor_1: str, t: float) -> str:
    """Cor entre cor_0 (t = 0) e cor_1 (t = 1)."""
    a = [int(cor_0[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(cor_1[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b, strict=True))


def _luminancia(cor: str) -> float:
    """Luminância relativa (WCAG), para escolher texto claro ou escuro sobre a bolha."""
    canais = [int(cor[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lineares = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in canais]
    return 0.2126 * lineares[0] + 0.7152 * lineares[1] + 0.0722 * lineares[2]


def _contraste(cor_1: str, cor_2: str) -> float:
    """Razão de contraste WCAG (1 a 21)."""
    claro, escuro = sorted([_luminancia(cor_1), _luminancia(cor_2)], reverse=True)
    return (claro + 0.05) / (escuro + 0.05)


def _tinta_sobre(fundo: str) -> str:
    """Texto branco ou preto, o que tiver mais contraste com o fundo.

    Preto puro (e não o grafite do tema): é o único par com o branco que garante 4,5:1
    em toda a rampa da bolha (pior caso 4,6:1, nos tons do meio).
    """
    return max(("#ffffff", "#000000"), key=lambda tinta: _contraste(tinta, fundo))


def _rotulo(questao: str) -> str:
    """'q07' → '7' (como no cartão-resposta); nomes sem número ficam como estão."""
    numero = re.search(r"(\d+)$", questao)
    return str(int(numero.group(1))) if numero else questao


def _cor_da_bolha(acerto: float, p: Paleta) -> tuple[str, str]:
    """(fundo, texto) da bolha: quanto maior o acerto, mais preenchida."""
    fundo = _misturar(p.bolha_vazia, p.bolha_cheia, 0.0 if np.isnan(acerto) else float(acerto))
    return fundo, _tinta_sobre(fundo)


def _claro_escuro(campo: str) -> str:
    """CSS light-dark() com o valor do campo nas duas paletas."""
    return f"light-dark({getattr(PALETAS['light'], campo)}, {getattr(PALETAS['dark'], campo)})"


def cartao_prova(tabela: pd.DataFrame) -> str:
    """HTML do cartão da prova: uma bolha por questão, na ordem da prova.

    Quanto mais alunos acertaram, mais preenchida a bolha (como a marcação no
    cartão-resposta). Questões com alerta ganham o círculo da caneta vermelha.

    As cores das duas paletas vão no CSS com light-dark(), que segue o
    color-scheme que o Streamlit põe no .stApp: o cartão troca de tema na hora,
    sem esperar o script rodar de novo (o Streamlit não roda ao trocar o tema).
    """
    claro, escuro = PALETAS["light"], PALETAS["dark"]
    bolhas = []
    for questao, linha in tabela.iterrows():
        acerto = linha["dificuldade"]
        (fundo_c, tinta_c), (fundo_e, tinta_e) = _cor_da_bolha(acerto, claro), _cor_da_bolha(acerto, escuro)
        revisar = bool(linha["alertas"])
        acerto_txt = "sem dados" if np.isnan(acerto) else f"{acerto:.0%} de acerto"
        dica = f"{questao}: {acerto_txt}, {linha['nivel']}" + (f". Revisar: {linha['alertas']}" if revisar else "")
        bolhas.append(
            f'<li class="{"revisar" if revisar else ""}" '
            f'style="background:light-dark({fundo_c}, {fundo_e});color:light-dark({tinta_c}, {tinta_e})" '
            f'title="{html.escape(dica)}" aria-label="{html.escape(dica)}">{html.escape(_rotulo(str(questao)))}</li>'
        )
    amostra = "".join(
        f'<span class="amostra" style="background:light-dark('
        f'{_cor_da_bolha(t, claro)[0]}, {_cor_da_bolha(t, escuro)[0]})"></span>'
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    return f"""
<style>
  .cartao-prova {{ font-family: {FONTE}; }}
  .cartao-prova ol {{
    list-style: none; margin: 0.25rem 0 0.75rem; padding: 0;
    display: flex; flex-wrap: wrap; gap: 0.6rem;
  }}
  .cartao-prova li {{
    box-sizing: border-box; min-width: 2.5rem; height: 2.5rem; padding: 0 0.5rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 999px; border: 1.5px solid {_claro_escuro("bolha_borda")};
    font-weight: 700; font-size: 0.95rem; font-variant-numeric: tabular-nums; cursor: default;
  }}
  .cartao-prova li.revisar {{ outline: 2.5px solid {_claro_escuro("alerta")}; outline-offset: 3px; }}
  .cartao-prova .legenda {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1.5rem;
    font-size: 0.875rem; color: {_claro_escuro("referencia")};
  }}
  .cartao-prova .legenda > span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
  .cartao-prova .amostra {{
    display: inline-block; width: 0.9rem; height: 0.9rem; border-radius: 999px;
    border: 1.5px solid {_claro_escuro("bolha_borda")};
  }}
  .cartao-prova .amostra.revisar {{
    background: transparent; outline: 2px solid {_claro_escuro("alerta")}; outline-offset: 2px;
  }}
</style>
<div class="cartao-prova">
  <ol aria-label="Questões da prova, na ordem">{"".join(bolhas)}</ol>
  <div class="legenda">
    <span>ninguém acertou {amostra} todos acertaram</span>
    <span><span class="amostra revisar"></span> precisa de revisão</span>
  </div>
</div>
"""


# ---------- gráficos ----------


def _layout(fig: go.Figure, eixo_x: str, eixo_y: str, altura: int = 420) -> go.Figure:
    fig.update_layout(
        height=altura,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(family=FONTE),
        hoverlabel=dict(font_family=FONTE),
    )
    fig.update_xaxes(title=eixo_x, zeroline=False)
    fig.update_yaxes(title=eixo_y, zeroline=False)
    return fig


def grafico_niveis(itens: pd.DataFrame, p: Paleta) -> go.Figure:
    contagem = itens["nivel"].value_counts().reindex(ORDEM_NIVEIS, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=contagem.index,
            y=contagem.to_numpy(),
            text=contagem.to_numpy(),
            textposition="outside",
            marker=dict(color=p.serie, cornerradius=4),
            cliponaxis=False,
            hovertemplate="%{x}: %{y} questões<extra></extra>",
        )
    )
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, contagem.max() * 1.25 + 0.5])
    _layout(fig, "", "", altura=200)
    return fig.update_layout(margin=dict(t=24))  # sem legenda: não precisa da faixa de cima


def grafico_dificuldade_discriminacao(tabela: pd.DataFrame, p: Paleta) -> go.Figure:
    """Cada questão é um ponto; as faixas mostram onde está a questão ideal (média e D ≥ 0,3)."""
    fig = go.Figure()
    fig.add_vrect(x0=0.30, x1=0.70, line_width=0, fillcolor=p.serie, opacity=0.07)
    fig.add_hline(
        y=0.20,
        line=dict(color=p.referencia, dash="dot", width=1),
        annotation_text="D = 0,2: abaixo disso não separa",
        annotation_position="bottom right",
        annotation_font_color=p.referencia,
    )
    fig.add_hline(y=0, line=dict(color=p.referencia, width=1))
    # Rótulos das faixas dentro do gráfico (a legenda ocupa a faixa acima dele).
    for x, texto in [(0.15, "difícil"), (0.50, "média"), (0.85, "fácil")]:
        fig.add_annotation(
            x=x, y=1, yref="paper", text=texto, showarrow=False, yanchor="top", font=dict(color=p.referencia)
        )

    com_alerta = tabela["alertas"] != ""
    for mascara, nome, cor, simbolo, modo in [
        (~com_alerta, "sem alerta", p.serie, "circle", "markers"),
        (com_alerta, "⚠ precisa de revisão", p.alerta, "diamond", "markers+text"),
    ]:
        d = tabela[mascara]
        if d.empty:
            continue
        fig.add_scatter(
            x=d["dificuldade"],
            y=d["discriminacao"],
            mode=modo,
            name=nome,
            text=d.index,
            textposition="top center",
            textfont=dict(size=12),
            marker=dict(color=cor, size=11, symbol=simbolo, line=dict(width=1.5, color=CONTORNO)),
            customdata=np.stack([d["nivel"], d["qualidade"], d["alertas"].replace("", "—")], axis=1),
            hovertemplate=(
                "<b>%{text}</b><br>Acerto: %{x:.0%} (%{customdata[0]})<br>"
                "D: %{y:.2f} (%{customdata[1]})<br>%{customdata[2]}<extra></extra>"
            ),
        )
    fig.update_xaxes(range=[-0.02, 1.02], tickformat=".0%")
    # Folga no topo para os rótulos das faixas não cobrirem pontos.
    d_min, d_max = min(tabela["discriminacao"].min(), 0), max(tabela["discriminacao"].max(), 0.2)
    fig.update_yaxes(range=[d_min - 0.1, d_max + 0.15])
    return _layout(fig, "Acerto (p): mais à direita = mais fácil", "Discriminação (D)", altura=460)


def grafico_curva_item(itens_tri: pd.DataFrame, questao: str, empirica: pd.DataFrame | None, p: Paleta) -> go.Figure:
    """CCI da questão escolhida sobre as demais (em cinza), com os pontos observados."""
    prob = pd.DataFrame(
        simulacao.prob_acerto(GRADE_THETA, *(itens_tri[par].to_numpy() for par in "abc")),
        index=GRADE_THETA,
        columns=itens_tri.index,
    )
    fig = go.Figure()
    for i, q in enumerate(prob.columns.drop(questao)):
        fig.add_scatter(
            x=GRADE_THETA,
            y=prob[q],
            mode="lines",
            line=dict(color=p.contexto, width=1),
            name="outras questões",
            legendgroup="outras",
            showlegend=i == 0,
            hoverinfo="skip",
        )
    b, c = itens_tri.loc[questao, ["b", "c"]]
    fig.add_hline(
        y=c,
        line=dict(color=p.referencia, dash="dot", width=1),
        annotation_text=f"chute: c = {c:.2f}",
        annotation_position="top left",
        annotation_font_color=p.referencia,
    )
    fig.add_vline(
        x=b,
        line=dict(color=p.referencia, dash="dot", width=1),
        annotation_text=f"b = {b:.2f}",
        annotation_position="bottom right",
        annotation_font_color=p.referencia,
    )
    fig.add_scatter(
        x=GRADE_THETA,
        y=prob[questao],
        mode="lines",
        name=f"{questao} (modelo 3PL)",
        line=dict(color=p.serie, width=2.5),
        hovertemplate="θ = %{x:.2f}<br>P(acerto) = %{y:.0%}<extra></extra>",
    )
    if empirica is not None:
        fig.add_scatter(
            x=empirica["theta"],
            y=empirica["proporcao"],
            mode="markers",
            name=f"{questao} (observado)",
            marker=dict(color=p.serie_2, size=9, line=dict(width=1.5, color=CONTORNO)),
            customdata=empirica["n"],
            hovertemplate="θ médio = %{x:.2f}<br>acertaram: %{y:.0%} de %{customdata} alunos<extra></extra>",
        )
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return _layout(fig, "Habilidade (θ): 0 = média dos alunos", "Probabilidade de acerto")


def grafico_informacao(info: pd.Series, p: Paleta) -> go.Figure:
    pico = info.idxmax()
    fig = go.Figure(
        go.Scatter(
            x=info.index,
            y=info.to_numpy(),
            mode="lines",
            fill="tozeroy",
            name="informação da prova",
            line=dict(color=p.serie, width=2.5),
            fillcolor=p.preenchimento,
            customdata=1 / np.sqrt(info.to_numpy()),
            hovertemplate="θ = %{x:.2f}<br>informação = %{y:.1f}<br>erro padrão ≈ %{customdata:.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=pico,
        line=dict(color=p.referencia, dash="dot", width=1),
        annotation_text=f"pico: θ = {pico:.2f}",
        annotation_position="top right",
        annotation_font_color=p.referencia,
    )
    return _layout(fig, "Habilidade (θ): 0 = média dos alunos", "Informação")


# ---------- página ----------


def painel(resumo: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alunos × questões", f"{resumo['n_alunos']} × {resumo['n_questoes']}")
    c2.metric(
        "Acerto médio",
        f"{resumo['acerto_medio']:.0%}",
        f"prova {resumo['nivel_prova']}",
        delta_color="off",
        delta_arrow="off",
    )
    kr20 = "—" if np.isnan(resumo["kr20"]) else f"{resumo['kr20']:.2f}"
    c3.metric(
        "Confiabilidade (KR-20)",
        kr20,
        resumo["confiabilidade"],
        delta_color="off",
        delta_arrow="off",
        help="Consistência da prova: ≥ 0,7 aceitável, ≥ 0,8 boa, ≥ 0,9 excelente.",
    )
    n_alerta = len(resumo["questoes_com_alerta"])
    c4.metric(
        "Para revisar",
        n_alerta,
        ", ".join(resumo["questoes_com_alerta"][:6]) or "nenhuma questão",
        delta_color="off",
        delta_arrow="off",
    )


def aba_questoes(tabela: pd.DataFrame, p: Paleta) -> None:
    so_alertas = st.toggle("Mostrar só as questões para revisar")
    exibida = tabela[tabela["alertas"] != ""] if so_alertas else tabela
    exibida = exibida.assign(alertas=exibida["alertas"].map(lambda t: f"⚠ {t}" if t else ""))

    def destacar(linha: pd.Series) -> list[str]:
        return [p.fundo_alerta if linha["Alertas"] else ""] * len(linha)

    estilo = (
        exibida.rename(columns=NOMES_COLUNAS)
        .rename_axis("Questão")
        .style.apply(destacar, axis=1)
        .format(
            {
                "Acerto (p)": "{:.0%}",
                "Discriminação (D)": "{:.2f}",
                "Ponto-bisserial": "{:.2f}",
                "KR-20 sem a questão": "{:.3f}",
                "a (TRI)": "{:.2f}",
                "b (TRI)": "{:.2f}",
                "c (TRI)": "{:.2f}",
            },
            na_rep="—",
        )
    )
    st.dataframe(estilo, width="stretch", height=min(38 + 35 * len(exibida), 640))
    st.caption(
        "Nível pelo % de acerto: 70% ou mais é fácil, de 30% a 70% é média, abaixo de 30% é difícil. "
        "Discriminação D = acerto dos 27% melhores menos o dos 27% piores (Ebel: abaixo de 0,2 é fraca, "
        "0,4 ou mais é ótima)."
    )


def main() -> None:
    st.set_page_config(page_title="Análise de Provas", page_icon="📝", layout="wide")
    p = paleta()
    st.title("Análise de Provas")
    st.markdown("Veja como cada questão da sua prova funcionou e quais precisam de revisão.")

    respostas, origem = barra_lateral()
    if respostas is None:
        st.info("Para começar, envie as respostas dos alunos na barra lateral ou clique em **Usar prova simulada**.")
        with st.expander("Formato dos arquivos"):
            st.markdown("**respostas.csv**")
            st.code("aluno,q1,q2,q3\na01,A,C,B\na02,A,D,B", language="text")
            st.markdown("**gabarito.csv** (opcional; sem ele, as respostas precisam ser 0/1)")
            st.code("questao,gabarito\nq1,A\nq2,C\nq3,B", language="text")
            st.markdown("Para testar o envio, baixe a prova de exemplo (12 alunos; a q6 tem problema de propósito):")
            c1, c2 = st.columns(2)
            for coluna, nome in [(c1, "respostas_exemplo.csv"), (c2, "gabarito_exemplo.csv")]:
                coluna.download_button(nome, (EXEMPLOS / nome).read_bytes(), file_name=nome, mime="text/csv")
        st.stop()

    itens, resumo = analisar_classica(respostas)
    itens_tri, alunos, convergiu = analisar_tri(respostas)
    tabela = tabela_relatorio(itens, itens_tri)

    st.caption(f"Analisando: {origem}")
    st.subheader("Como cada questão se saiu")
    st.html(cartao_prova(tabela))
    revisar = resumo["questoes_com_alerta"]
    if revisar:
        st.markdown(f"Revise primeiro: **{', '.join(revisar)}**. O motivo de cada uma está na tabela abaixo.")

    painel(resumo)
    if len(respostas) < MIN_ALUNOS_TRI:
        st.warning(
            f"Com {len(respostas)} alunos, os parâmetros da TRI (a, b, c e as curvas) ficam instáveis; "
            f"o ideal é 1000 ou mais. As métricas clássicas continuam valendo.",
            icon="⚠️",
        )
    if not convergiu:
        st.warning("A estimação da TRI não convergiu: trate a, b e c com cautela.", icon="⚠️")

    col_niveis, col_baixar = st.columns([3, 1], vertical_alignment="center")
    with col_niveis:
        st.markdown("**Quantas questões de cada nível**")
        st.plotly_chart(grafico_niveis(itens, p), width="stretch")
    with col_baixar:
        st.download_button(
            "Baixar relatório (CSV)",
            tabela.round(4).to_csv(encoding="utf-8"),
            file_name="relatorio_prova.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )
        st.caption("Uma linha por questão: métricas clássicas, a, b, c e alertas.")

    questoes, dispersao, curva, informacao = st.tabs(
        ["Questões", "Dificuldade × discriminação", "Curva característica", "Informação da prova"]
    )

    with questoes:
        aba_questoes(tabela, p)

    with dispersao:
        st.plotly_chart(grafico_dificuldade_discriminacao(tabela, p), width="stretch")
        st.caption(
            "A faixa azul é a dificuldade média (30% a 70% de acerto). Questões boas ficam nela e acima de "
            "D = 0,2. Abaixo de 0, quem sabe mais erra mais: gabarito errado ou enunciado ambíguo."
        )

    with curva:
        questao = st.selectbox(
            "Questão", itens_tri.index, format_func=lambda q: f"{q} ⚠" if tabela.at[q, "alertas"] else q
        )
        empirica = None
        if len(respostas) >= MIN_ALUNOS_CURVA_EMPIRICA:
            empirica = tri.curva_empirica(respostas[questao], alunos["theta"])
        st.plotly_chart(grafico_curva_item(itens_tri, questao, empirica, p), width="stretch")
        a, b, c = itens_tri.loc[questao, ["a", "b", "c"]]
        if a < 0:
            st.error(
                f"{questao}: a = {a:.2f} < 0, a curva desce: quem sabe mais erra mais. Revise gabarito e enunciado.",
                icon="⚠️",
            )
        else:
            st.caption(
                f"{questao}: a = {a:.2f} é a inclinação (quanto a questão separa alunos perto de b); "
                f"b = {b:.2f} é a habilidade em que a chance de acerto fica no meio do caminho entre o chute "
                f"e 100%; c = {c:.2f} é a chance de acerto no chute."
            )
        if empirica is None:
            st.caption(f"Os pontos observados aparecem com {MIN_ALUNOS_CURVA_EMPIRICA} alunos ou mais.")
        else:
            st.caption(
                "Pontos laranja: % de acerto real dos alunos em 10 faixas de habilidade. "
                "Se ficam longe da curva azul, o modelo não descreve bem a questão."
            )

    with informacao:
        info = tri.informacao(GRADE_THETA, itens_tri).sum(axis=1)
        st.plotly_chart(grafico_informacao(info, p), width="stretch")
        pico = info.idxmax()
        perto = float(((alunos["theta"] - pico).abs() <= 1).mean())
        st.caption(
            f"A prova mede com mais precisão alunos com θ perto de {pico:.2f} "
            f"({perto:.0%} dos alunos estão a até 1 desvio desse ponto). Onde a informação é baixa, "
            "o erro padrão da habilidade (≈ 1/√informação, no tooltip) é alto. "
            "Para medir melhor uma faixa, inclua questões com b nela."
        )


if __name__ == "__main__":
    main()
