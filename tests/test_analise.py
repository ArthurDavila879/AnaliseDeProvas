import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import classica, cli, ingestao, relatorio, simulacao, tri

RAIZ = Path(__file__).resolve().parent.parent
EXEMPLOS = RAIZ / "exemplos"


# ---------- ingestão ----------

def test_corrigir_branco_e_dupla_marcacao_contam_como_erro():
    respostas = pd.Series(["ABCDE", "A.C*E", "EDCBA"])
    matriz = ingestao.corrigir(respostas, "ABCDE")
    assert matriz.values.tolist() == [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]]


def test_carregar_csv_com_gabarito():
    matriz = ingestao.carregar_csv(EXEMPLOS / "respostas_exemplo.csv", EXEMPLOS / "gabarito_exemplo.csv")
    assert matriz.shape == (12, 6)
    assert matriz.loc["a01"].tolist() == [1, 1, 1, 1, 1, 0]


def test_carregar_csv_aceita_arquivo_em_memoria():
    # É assim que o app passa o upload do Streamlit.
    respostas = io.BytesIO((EXEMPLOS / "respostas_exemplo.csv").read_bytes())
    gabarito = io.BytesIO((EXEMPLOS / "gabarito_exemplo.csv").read_bytes())
    assert ingestao.carregar_csv(respostas, gabarito).equals(
        ingestao.carregar_csv(EXEMPLOS / "respostas_exemplo.csv", EXEMPLOS / "gabarito_exemplo.csv"))


def test_carregar_csv_sem_gabarito_rejeita_letras():
    with pytest.raises(ValueError):
        ingestao.carregar_csv(EXEMPLOS / "respostas_exemplo.csv")


def test_carregar_enem_escolhe_caderno_mais_comum(tmp_path):
    # Imita o formato dos microdados: ';', latin-1, colunas TX_* e CO_PROVA_*.
    linhas = ["NU_INSCRICAO;CO_PROVA_MT;TP_PRESENCA_MT;TX_RESPOSTAS_MT;TX_GABARITO_MT;NU_NOTA_MT"]
    for i in range(6):
        linhas.append(f"{i};1001;1;ABCDE;ABCDE;700.5")   # caderno mais comum
    for i in range(6, 8):
        linhas.append(f"{i};1002;1;EDCBA;BCDEA;400.0")   # outro caderno
    linhas.append("8;1001;0;;;")                          # faltou: deve ser ignorado
    arquivo = tmp_path / "MICRODADOS_FAKE.csv"
    arquivo.write_text("\n".join(linhas), encoding="latin-1")

    matriz, nota, co_prova = ingestao.carregar_enem(arquivo, area="MT", n_alunos=None, chunksize=3)
    assert co_prova == 1001
    assert matriz.shape == (6, 5)
    assert matriz.values.all()
    assert (nota == 700.5).all()


def test_carregar_enem_area_invalida(tmp_path):
    with pytest.raises(ValueError):
        ingestao.carregar_enem(tmp_path / "x.csv", area="LC")


# ---------- métricas clássicas ----------

def test_dificuldade_e_proporcao_de_acerto():
    df = pd.DataFrame({"q1": [1, 1, 1, 0], "q2": [0, 0, 0, 1]})
    assert classica.dificuldade(df).tolist() == [0.75, 0.25]


def test_discriminacao_negativa_quando_fracos_acertam_mais():
    # 4 itens "normais" definem quem é forte; q5 só é acertada pelos fracos.
    fortes = [[1, 1, 1, 1, 0]] * 10
    fracos = [[0, 0, 0, 0, 1]] * 10
    df = pd.DataFrame(fortes + fracos, columns=[f"q{i}" for i in range(1, 6)])
    d = classica.discriminacao_grupos(df)
    assert d["q1"] == pytest.approx(1.0)
    assert d["q5"] == pytest.approx(-1.0)


def test_kr20_itens_identicos_vale_1():
    col = [1, 0, 1, 1, 0, 0, 1, 0]
    df = pd.DataFrame({f"q{i}": col for i in range(5)})
    assert classica.kr20(df) == pytest.approx(1.0)


def test_bisserial_nan_quando_todos_acertam():
    df = pd.DataFrame({"q1": [1, 1, 1, 1], "q2": [1, 0, 1, 0], "q3": [1, 0, 0, 1]})
    assert np.isnan(classica.ponto_bisserial(df)["q1"])


# ---------- classificação ----------

@pytest.mark.parametrize("p,esperado", [(0.85, "fácil"), (0.5, "média"), (0.1, "difícil")])
def test_nivel_questao(p, esperado):
    assert relatorio.nivel_questao(p) == esperado


@pytest.mark.parametrize("d,esperado", [(-0.1, "problemática"), (0.1, "fraca"), (0.25, "regular"), (0.35, "boa"), (0.5, "ótima")])
def test_qualidade_discriminacao(d, esperado):
    assert relatorio.qualidade_discriminacao(d) == esperado


# ---------- validação com simulação ----------

def test_metricas_recuperam_parametros_simulados():
    respostas, parametros, _ = simulacao.simular_prova(n_alunos=3000, n_itens=30, n_problematicas=2, seed=7)
    itens, confiab = relatorio.classificar_itens(respostas)

    # Mais difícil na TRI (b maior) => menor % de acerto.
    assert np.corrcoef(parametros["b"], itens["dificuldade"])[0, 1] < -0.85

    # As questões com discriminação negativa precisam ser sinalizadas.
    problematicas = parametros.index[parametros["problematica"]]
    assert (itens.loc[problematicas, "qualidade"] == "problemática").all()
    assert itens.loc[problematicas, "alertas"].str.contains("revisar").all()

    assert 0.6 < confiab < 1.0


# ---------- TRI (3PL) ----------

@pytest.fixture(scope="module")
def prova_tri():
    respostas, parametros, theta = simulacao.simular_prova(n_alunos=3000, n_itens=25, n_problematicas=2, seed=11)
    itens, alunos = tri.ajustar_3pl(respostas)
    return respostas, parametros, theta, itens, alunos


def test_tri_recupera_parametros_simulados(prova_tri):
    _, parametros, _, itens, _ = prova_tri
    assert itens.attrs["convergiu"]
    normais = ~parametros["problematica"]
    assert np.corrcoef(parametros.loc[normais, "b"], itens.loc[normais, "b"])[0, 1] > 0.95
    assert np.corrcoef(parametros.loc[normais, "a"], itens.loc[normais, "a"])[0, 1] > 0.8
    # c é o parâmetro mais difícil de identificar: basta ficar perto do verdadeiro.
    assert (parametros.loc[normais, "c"] - itens.loc[normais, "c"]).abs().mean() < 0.08


def test_tri_discriminacao_negativa_nas_problematicas(prova_tri):
    _, parametros, _, itens, _ = prova_tri
    assert (itens.loc[parametros["problematica"], "a"] < 0).all()


def test_tri_theta_correlaciona_com_verdadeiro(prova_tri):
    respostas, _, theta, _, alunos = prova_tri
    assert alunos.shape == (len(respostas), 2)
    assert np.corrcoef(theta, alunos["theta"])[0, 1] > 0.9
    assert (alunos["erro_padrao"] > 0).all()


def test_tri_theta_ordena_pelo_numero_de_acertos():
    # Itens iguais: θ só depende do total de acertos, e precisa crescer com ele.
    itens = pd.DataFrame({"a": [1.0] * 4, "b": [0.0] * 4, "c": [0.2] * 4}, index=[f"q{i}" for i in range(4)])
    respostas = pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], columns=itens.index)
    theta = tri.estimar_theta(respostas, itens)["theta"]
    assert theta.is_monotonic_increasing and theta.iloc[0] < 0 < theta.iloc[-1]


def test_tri_exige_duas_questoes():
    with pytest.raises(ValueError):
        tri.estimar_itens(pd.DataFrame({"q1": [0, 1, 1]}))


def test_informacao_maxima_em_b_sem_chute():
    # Sem chute (c = 0), a informação é máxima em θ = b e vale (D·a)²/4.
    itens = pd.DataFrame({"a": [1.0, 2.0], "b": [-1.0, 0.5], "c": [0.0, 0.0]}, index=["q01", "q02"])
    grade = np.linspace(-3, 3, 601)
    info = tri.informacao(grade, itens)
    assert info.idxmax().tolist() == pytest.approx([-1.0, 0.5])
    assert info.max().tolist() == pytest.approx([(simulacao.D * 1.0) ** 2 / 4, (simulacao.D * 2.0) ** 2 / 4])


def test_informacao_cai_com_chute():
    sem = pd.DataFrame({"a": [1.0], "b": [0.0], "c": [0.0]}, index=["q01"])
    com = sem.assign(c=0.25)
    assert tri.informacao([0.0], com).iloc[0, 0] < tri.informacao([0.0], sem).iloc[0, 0]


def test_curva_empirica_por_faixa_de_theta():
    theta = pd.Series([-2.0, -1.0, 1.0, 2.0])
    acertos = pd.Series([0, 0, 1, 1])
    curva = tri.curva_empirica(acertos, theta, n_grupos=2)
    assert curva["theta"].tolist() == [-1.5, 1.5]
    assert curva["proporcao"].tolist() == [0.0, 1.0]
    assert curva["n"].tolist() == [2, 2]


def test_cli_tri_salva_itens_tri(tmp_path):
    cli.main(["tri", "--alunos", "500", "--questoes", "8", "--saida", str(tmp_path)])
    itens = pd.read_csv(tmp_path / "itens_tri.csv", index_col="questao")
    assert itens.columns.tolist() == ["a", "b", "c"]
    assert len(itens) == 8


def test_cli_tri_com_csv(tmp_path):
    cli.main(["tri", "--csv", str(EXEMPLOS / "respostas_exemplo.csv"),
              "--gabarito", str(EXEMPLOS / "gabarito_exemplo.csv"), "--saida", str(tmp_path)])
    assert len(pd.read_csv(tmp_path / "itens_tri.csv")) == 6


# ---------- validação contra os parâmetros oficiais do INEP ----------

ESCALA_A, ESCALA_B = 1.3, 0.5   # escala "do INEP" no teste: b_inep = 1.3·b + 0.5


@pytest.fixture(scope="module")
def enem_sintetico(tmp_path_factory):
    """Microdados + ITENS_PROVA no formato do INEP, com parâmetros verdadeiros conhecidos."""
    pasta = tmp_path_factory.mktemp("enem")
    respostas, parametros, _ = simulacao.simular_prova(n_alunos=2000, n_itens=20, n_problematicas=0, seed=5)
    k = respostas.shape[1]

    gabarito = "A" * k
    linhas = ["NU_INSCRICAO;CO_PROVA_MT;TP_PRESENCA_MT;TX_RESPOSTAS_MT;TX_GABARITO_MT;NU_NOTA_MT"]
    for i, acertos in enumerate(respostas.to_numpy()):
        texto = "".join("A" if x else "B" for x in acertos)
        linhas.append(f"{i};1211;1;{texto};{gabarito};{500 + 10 * acertos.sum()}")
    microdados = pasta / "MICRODADOS_ENEM_2099.csv"
    microdados.write_text("\n".join(linhas), encoding="latin-1")

    rng = np.random.default_rng(0)
    cab = "CO_POSICAO;SG_AREA;CO_ITEM;TX_GABARITO;IN_ITEM_ABAN;NU_PARAM_A;NU_PARAM_B;NU_PARAM_C;CO_PROVA"
    linhas = [cab]
    for j in rng.permutation(k):  # fora de ordem de propósito
        a, b, c = parametros.iloc[j][["a", "b", "c"]]
        anulado = j == 3
        par = ";;" if anulado else f"{a / ESCALA_A:.5f};{ESCALA_A * b + ESCALA_B:.5f};{c:.5f}"
        linhas.append(f"{136 + j};MT;{9000 + j};A;{int(anulado)};{par};1211")
    for j in range(k):  # outro caderno, que não pode ser misturado
        linhas.append(f"{136 + j};MT;{7000 + j};B;0;9.0;9.0;0.5;1212")
    linhas.append("46;CH;1;C;0;1.0;0.0;0.2;1211")  # outra área, mesmo CO_PROVA
    itens = pasta / "ITENS_PROVA_2099.csv"
    itens.write_text("\n".join(linhas), encoding="latin-1")
    return microdados, itens, parametros


def test_carregar_itens_inep_ordena_por_posicao_e_filtra_caderno(enem_sintetico):
    _, itens, parametros = enem_sintetico
    oficiais = ingestao.carregar_itens_inep(itens, "mt", 1211)
    assert oficiais.index.tolist() == parametros.index.tolist()
    assert oficiais["co_posicao"].is_monotonic_increasing
    assert oficiais.loc["q05", "b_inep"] == pytest.approx(ESCALA_A * parametros.loc["q05", "b"] + ESCALA_B, abs=1e-4)
    assert oficiais["anulado"].tolist() == [q == "q04" for q in oficiais.index]
    assert np.isnan(oficiais.loc["q04", "a_inep"])


def test_carregar_itens_inep_caderno_inexistente(enem_sintetico):
    _, itens, _ = enem_sintetico
    with pytest.raises(ValueError):
        ingestao.carregar_itens_inep(itens, "MT", 9999)


def test_comparar_com_inep_recupera_escala(enem_sintetico):
    _, itens, parametros = enem_sintetico
    oficiais = ingestao.carregar_itens_inep(itens, "MT", 1211)
    # Estimativas = verdadeiros + ruído: a comparação tem que achar a escala 1.3 / 0.5.
    rng = np.random.default_rng(1)
    estimados = parametros[["a", "b", "c"]] + rng.normal(0, 0.03, (len(parametros), 3))
    tabela, resumo = tri.comparar_com_inep(estimados, oficiais)
    assert resumo["n_comparados"] == len(parametros) - 1 and resumo["anulados"] == ["q04"]
    assert resumo["escala_A"] == pytest.approx(ESCALA_A, rel=0.05)
    assert resumo["escala_B"] == pytest.approx(ESCALA_B, abs=0.1)
    assert resumo["pearson_b"] > 0.99 and resumo["spearman_b"] > 0.98
    assert resumo["rmse_b_convertido"] < 0.1
    assert (tabela["a_convertido"] - tabela["a_inep"]).abs().max() < 0.1


def test_grafico_b_inep_gera_png(enem_sintetico, tmp_path):
    _, itens, parametros = enem_sintetico
    oficiais = ingestao.carregar_itens_inep(itens, "MT", 1211)
    tabela, resumo = tri.comparar_com_inep(parametros[["a", "b", "c"]], oficiais)
    png = relatorio.grafico_b_inep(tabela, resumo, tmp_path / "sub" / "b.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_tri_enem_com_itens_inep(enem_sintetico, tmp_path):
    microdados, itens, _ = enem_sintetico
    cli.main(["tri", "--enem", str(microdados), "--itens", str(itens), "--alunos", "0", "--saida", str(tmp_path)])
    validacao = pd.read_csv(tmp_path / "validacao_inep.csv", index_col="questao")
    assert len(validacao) == 20
    assert {"b", "b_inep", "b_convertido", "dif_b"} <= set(validacao.columns)
    validos = ~validacao["anulado"]
    assert np.corrcoef(validacao.loc[validos, "b"], validacao.loc[validos, "b_inep"])[0, 1] > 0.95
    assert (tmp_path / "b_vs_inep.png").exists()


def test_cli_tri_itens_sem_enem_falha(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["tri", "--itens", "x.csv", "--saida", str(tmp_path)])


# ---------- app (Streamlit) ----------

def _app_simulado(alunos: int, questoes: int):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    app.number_input(key="sim_alunos").set_value(alunos)
    app.number_input(key="sim_questoes").set_value(questoes)
    return app.button(key="usar_simulada").click().run()


def test_app_sem_dados_pede_a_prova():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert "Usar prova simulada" in app.info[0].value


def test_app_com_prova_simulada():
    app = _app_simulado(alunos=600, questoes=10)
    assert not app.exception
    assert app.metric[0].value == "600 × 10"
    assert len(app.dataframe) == 1
    assert len(app.tabs) == 4
    # Duas questões problemáticas por padrão: aparecem no painel de alertas.
    assert int(app.metric[3].value) >= 2


def test_app_troca_a_questao_da_curva():
    app = _app_simulado(alunos=600, questoes=10)
    app.selectbox[0].select("q05").run()
    assert not app.exception
