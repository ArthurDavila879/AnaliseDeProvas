"""App web (Streamlit): nível de cada questão e da prova, pela Teoria Clássica e pela TRI.

    streamlit run app.py          # ou: docker compose up app → http://localhost:8501

Toda a análise vem de src/; aqui ficam só a interface e os gráficos.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ingestao, relatorio, simulacao, tri

# Cores: série principal, segunda série e alerta (sempre junto de ícone/texto, nunca só a cor).
COR_SERIE = "#2a78d6"
COR_SERIE_2 = "#eb6834"
COR_ALERTA = "#d03b3b"
COR_CONTEXTO = "rgba(138, 137, 132, 0.35)"
FUNDO_ALERTA = "background-color: rgba(208, 59, 59, 0.14)"

ORDEM_NIVEIS = ["fácil", "média", "difícil"]
MIN_ALUNOS_TRI = 500           # abaixo disso o 3PL fica instável (mesmo aviso da CLI)
MIN_ALUNOS_CURVA_EMPIRICA = 200
GRADE_THETA = np.linspace(-4, 4, 161)

NOMES_COLUNAS = {
    "dificuldade": "Acerto (p)", "nivel": "Nível", "discriminacao": "Discriminação (D)",
    "qualidade": "Qualidade", "bisserial": "Ponto-bisserial", "kr20_sem_item": "KR-20 sem a questão",
    "a": "a (TRI)", "b": "b (TRI)", "c": "c (TRI)", "alertas": "Alertas",
}


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
    st.sidebar.header("Dados da prova")
    arq_resp = st.sidebar.file_uploader(
        "Respostas (CSV)", type="csv", on_change=_usar, args=("csv",),
        help="Coluna `aluno` + uma coluna por questão. Células com letras (A–E) se enviar o gabarito; 0/1 se não.",
    )
    arq_gab = st.sidebar.file_uploader(
        "Gabarito (CSV, opcional)", type="csv", on_change=_usar, args=("csv",),
        help="Colunas `questao,gabarito`, com os mesmos nomes de questão das respostas.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Ou use uma prova simulada")
    with st.sidebar.expander("Parâmetros da simulação"):
        n_alunos = st.number_input("Alunos", 100, 20_000, 2_000, step=100, key="sim_alunos")
        n_questoes = st.number_input("Questões", 2, 90, 30, key="sim_questoes")
        n_problematicas = st.number_input("Questões com problema", 0, 10, 2, key="sim_problematicas",
                                          help="Discriminação negativa: imitam gabarito errado ou enunciado ambíguo.")
        seed = st.number_input("Semente", 0, 10_000, 42, key="sim_seed")
    st.sidebar.button("Usar prova simulada", type="primary", on_click=_usar, args=("simulada",),
                      key="usar_simulada", width="stretch")

    fonte = st.session_state.get("fonte")
    if fonte == "simulada":
        respostas = carregar_simulada(int(n_alunos), int(n_questoes), int(n_problematicas), int(seed))
        return respostas, f"Prova simulada (3PL, semente {seed})"
    if fonte == "csv" and arq_resp is not None:
        try:
            respostas = carregar_upload(arq_resp.getvalue(), arq_gab.getvalue() if arq_gab else None)
        except (ValueError, KeyError) as erro:
            st.error(f"Não consegui ler a prova: {erro}")
            return None, ""
        origem = arq_resp.name + (f" + {arq_gab.name}" if arq_gab else " (já corrigida, 0/1)")
        return respostas, origem
    return None, ""


# ---------- gráficos ----------

def _layout(fig: go.Figure, eixo_x: str, eixo_y: str, altura: int = 420) -> go.Figure:
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(title=eixo_x, zeroline=False)
    fig.update_yaxes(title=eixo_y, zeroline=False)
    return fig


def grafico_niveis(itens: pd.DataFrame) -> go.Figure:
    contagem = itens["nivel"].value_counts().reindex(ORDEM_NIVEIS, fill_value=0)
    fig = go.Figure(go.Bar(
        x=contagem.index, y=contagem.to_numpy(), text=contagem.to_numpy(), textposition="outside",
        marker=dict(color=COR_SERIE, cornerradius=4), cliponaxis=False,
        hovertemplate="%{x}: %{y} questões<extra></extra>",
    ))
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, contagem.max() * 1.25 + 0.5])
    return _layout(fig, "", "", altura=230)


def grafico_dificuldade_discriminacao(tabela: pd.DataFrame) -> go.Figure:
    """Cada questão é um ponto; as faixas mostram onde está a questão ideal (média e D ≥ 0,3)."""
    fig = go.Figure()
    fig.add_vrect(x0=0.30, x1=0.70, line_width=0, fillcolor=COR_SERIE, opacity=0.06)
    fig.add_hline(y=0.20, line=dict(color="gray", dash="dot", width=1),
                  annotation_text="D = 0,2: abaixo disso não separa", annotation_position="bottom right")
    fig.add_hline(y=0, line=dict(color="gray", width=1))
    for x, texto in [(0.15, "difícil"), (0.50, "média"), (0.85, "fácil")]:
        fig.add_annotation(x=x, y=1, yref="paper", text=texto, showarrow=False, yanchor="bottom", font=dict(color="gray"))

    com_alerta = tabela["alertas"] != ""
    for mascara, nome, cor, simbolo, modo in [
        (~com_alerta, "sem alerta", COR_SERIE, "circle", "markers"),
        (com_alerta, "⚠ com alerta", COR_ALERTA, "diamond", "markers+text"),
    ]:
        d = tabela[mascara]
        if d.empty:
            continue
        fig.add_scatter(
            x=d["dificuldade"], y=d["discriminacao"], mode=modo, name=nome, text=d.index,
            textposition="top center", textfont=dict(size=11),
            marker=dict(color=cor, size=11, symbol=simbolo, line=dict(width=1, color="white")),
            customdata=np.stack([d["nivel"], d["qualidade"], d["alertas"].replace("", "—")], axis=1),
            hovertemplate=("<b>%{text}</b><br>Acerto: %{x:.0%} (%{customdata[0]})<br>"
                           "D: %{y:.2f} (%{customdata[1]})<br>%{customdata[2]}<extra></extra>"),
        )
    fig.update_xaxes(range=[-0.02, 1.02], tickformat=".0%")
    return _layout(fig, "Acerto (p): mais à direita = mais fácil", "Discriminação (D)", altura=460)


def grafico_curva_item(itens_tri: pd.DataFrame, questao: str, empirica: pd.DataFrame | None) -> go.Figure:
    """CCI da questão escolhida sobre as demais (em cinza), com os pontos observados."""
    prob = pd.DataFrame(
        simulacao.prob_acerto(GRADE_THETA, *(itens_tri[p].to_numpy() for p in "abc")),
        index=GRADE_THETA, columns=itens_tri.index,
    )
    fig = go.Figure()
    for i, q in enumerate(prob.columns.drop(questao)):
        fig.add_scatter(x=GRADE_THETA, y=prob[q], mode="lines", line=dict(color=COR_CONTEXTO, width=1),
                        name="outras questões", legendgroup="outras", showlegend=i == 0, hoverinfo="skip")
    a, b, c = itens_tri.loc[questao, ["a", "b", "c"]]
    fig.add_hline(y=c, line=dict(color="gray", dash="dot", width=1),
                  annotation_text=f"chute: c = {c:.2f}", annotation_position="top left")
    fig.add_vline(x=b, line=dict(color="gray", dash="dot", width=1),
                  annotation_text=f"b = {b:.2f}", annotation_position="bottom right")
    fig.add_scatter(x=GRADE_THETA, y=prob[questao], mode="lines", name=f"{questao} (modelo 3PL)",
                    line=dict(color=COR_SERIE, width=2.5),
                    hovertemplate="θ = %{x:.2f}<br>P(acerto) = %{y:.0%}<extra></extra>")
    if empirica is not None:
        fig.add_scatter(x=empirica["theta"], y=empirica["proporcao"], mode="markers", name=f"{questao} (observado)",
                        marker=dict(color=COR_SERIE_2, size=9, line=dict(width=1, color="white")),
                        customdata=empirica["n"],
                        hovertemplate="θ médio = %{x:.2f}<br>acertaram: %{y:.0%} de %{customdata} alunos<extra></extra>")
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return _layout(fig, "Habilidade (θ): 0 = média dos alunos", "Probabilidade de acerto")


def grafico_informacao(info: pd.Series) -> go.Figure:
    pico = info.idxmax()
    fig = go.Figure(go.Scatter(
        x=info.index, y=info.to_numpy(), mode="lines", fill="tozeroy", name="informação da prova",
        line=dict(color=COR_SERIE, width=2.5), fillcolor="rgba(42, 120, 214, 0.12)",
        customdata=1 / np.sqrt(info.to_numpy()),
        hovertemplate="θ = %{x:.2f}<br>informação = %{y:.1f}<br>erro padrão ≈ %{customdata:.2f}<extra></extra>",
    ))
    fig.add_vline(x=pico, line=dict(color="gray", dash="dot", width=1),
                  annotation_text=f"pico: θ = {pico:.2f}", annotation_position="top right")
    return _layout(fig, "Habilidade (θ): 0 = média dos alunos", "Informação")


# ---------- página ----------

def painel(resumo: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alunos × questões", f"{resumo['n_alunos']} × {resumo['n_questoes']}")
    c2.metric("Acerto médio", f"{resumo['acerto_medio']:.0%}", f"prova {resumo['nivel_prova']}", delta_color="off")
    kr20 = "—" if np.isnan(resumo["kr20"]) else f"{resumo['kr20']:.2f}"
    c3.metric("Confiabilidade (KR-20)", kr20, resumo["confiabilidade"], delta_color="off",
              help="Consistência da prova: ≥ 0,7 aceitável · ≥ 0,8 boa · ≥ 0,9 excelente")
    n_alerta = len(resumo["questoes_com_alerta"])
    c4.metric("Questões com alerta", n_alerta, ", ".join(resumo["questoes_com_alerta"][:6]) or "nenhuma",
              delta_color="off")


def aba_questoes(tabela: pd.DataFrame) -> None:
    so_alertas = st.toggle("Mostrar só as questões com alerta")
    exibida = tabela[tabela["alertas"] != ""] if so_alertas else tabela
    exibida = exibida.assign(alertas=exibida["alertas"].map(lambda t: f"⚠ {t}" if t else ""))

    def destacar(linha: pd.Series) -> list[str]:
        return [FUNDO_ALERTA if linha["Alertas"] else ""] * len(linha)

    estilo = (exibida.rename(columns=NOMES_COLUNAS).rename_axis("Questão").style
              .apply(destacar, axis=1)
              .format({"Acerto (p)": "{:.0%}", "Discriminação (D)": "{:.2f}", "Ponto-bisserial": "{:.2f}",
                       "KR-20 sem a questão": "{:.3f}", "a (TRI)": "{:.2f}", "b (TRI)": "{:.2f}", "c (TRI)": "{:.2f}"},
                      na_rep="—"))
    st.dataframe(estilo, width="stretch", height=min(38 + 35 * len(exibida), 640))
    st.caption("Nível pelo % de acerto (≥ 70% fácil · 30–70% média · < 30% difícil). "
               "Discriminação D = acerto dos 27% melhores − 27% piores (Ebel: < 0,2 fraca · ≥ 0,4 ótima).")


def main() -> None:
    st.set_page_config(page_title="Análise de Provas", page_icon="📝", layout="wide")
    st.title("Análise de Provas")
    st.caption("Nível de cada questão e da prova, pela Teoria Clássica dos Testes e pela TRI (3PL).")

    respostas, origem = barra_lateral()
    if respostas is None:
        st.info("Envie as respostas (e o gabarito) na barra lateral, ou clique em **Usar prova simulada**.")
        with st.expander("Formato dos arquivos"):
            st.markdown("**respostas.csv**")
            st.code("aluno,q1,q2,q3\na01,A,C,B\na02,A,D,B", language="text")
            st.markdown("**gabarito.csv** (opcional; sem ele, as respostas precisam ser 0/1)")
            st.code("questao,gabarito\nq1,A\nq2,C\nq3,B", language="text")
        st.stop()

    itens, resumo = analisar_classica(respostas)
    itens_tri, alunos, convergiu = analisar_tri(respostas)
    tabela = tabela_relatorio(itens, itens_tri)

    st.caption(f"Fonte: {origem}")
    painel(resumo)
    if len(respostas) < MIN_ALUNOS_TRI:
        st.warning(f"Com {len(respostas)} alunos os parâmetros da TRI (a, b, c e as curvas) ficam instáveis; "
                   f"o ideal é 1000+. As métricas clássicas continuam valendo.", icon="⚠️")
    if not convergiu:
        st.warning("A estimação da TRI não convergiu: trate a, b e c com cautela.", icon="⚠️")

    col_niveis, col_baixar = st.columns([3, 1], vertical_alignment="center")
    with col_niveis:
        st.markdown("**Distribuição dos níveis das questões**")
        st.plotly_chart(grafico_niveis(itens), width="stretch")
    with col_baixar:
        st.download_button("Baixar relatório (CSV)", tabela.round(4).to_csv(encoding="utf-8"),
                           file_name="relatorio_prova.csv", mime="text/csv", type="primary", width="stretch")
        st.caption("Uma linha por questão: métricas clássicas, a, b, c e alertas.")

    questoes, dispersao, curva, informacao = st.tabs(
        ["Questões", "Dificuldade × discriminação", "Curva característica", "Informação da prova"])

    with questoes:
        aba_questoes(tabela)

    with dispersao:
        st.plotly_chart(grafico_dificuldade_discriminacao(tabela), width="stretch")
        st.caption("A faixa azul é a dificuldade média (30–70%). Questões boas ficam nela e acima de D = 0,2; "
                   "abaixo de 0 quem sabe mais erra mais (gabarito errado ou enunciado ambíguo).")

    with curva:
        questao = st.selectbox("Questão", itens_tri.index,
                               format_func=lambda q: f"{q} ⚠" if tabela.at[q, "alertas"] else q)
        empirica = None
        if len(respostas) >= MIN_ALUNOS_CURVA_EMPIRICA:
            empirica = tri.curva_empirica(respostas[questao], alunos["theta"])
        st.plotly_chart(grafico_curva_item(itens_tri, questao, empirica), width="stretch")
        a, b, c = itens_tri.loc[questao, ["a", "b", "c"]]
        if a < 0:
            st.error(f"{questao}: a = {a:.2f} < 0, a curva desce: quem sabe mais erra mais. Revise gabarito e enunciado.",
                     icon="⚠️")
        else:
            st.caption(f"{questao}: a = {a:.2f} (inclinação: quanto separa alunos perto de b) · b = {b:.2f} "
                       f"(habilidade com chance de acerto no meio do caminho entre o chute e 100%) · "
                       f"c = {c:.2f} (chance de acerto no chute).")
        if empirica is None:
            st.caption(f"Os pontos observados aparecem com {MIN_ALUNOS_CURVA_EMPIRICA}+ alunos.")
        else:
            st.caption("Pontos laranja: % de acerto real dos alunos em 10 faixas de habilidade. "
                       "Longe da curva azul = o modelo não descreve bem a questão.")

    with informacao:
        info = tri.informacao(GRADE_THETA, itens_tri).sum(axis=1)
        st.plotly_chart(grafico_informacao(info), width="stretch")
        pico = info.idxmax()
        perto = float(((alunos["theta"] - pico).abs() <= 1).mean())
        st.caption(f"A prova mede com mais precisão alunos com θ perto de {pico:.2f} "
                   f"({perto:.0%} dos alunos estão a até 1 desvio desse ponto). Onde a informação é baixa, "
                   "o erro padrão da habilidade (≈ 1/√informação, no tooltip) é alto. "
                   "Para medir melhor uma faixa, inclua questões com b nela.")


if __name__ == "__main__":
    main()
