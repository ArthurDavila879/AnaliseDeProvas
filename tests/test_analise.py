from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import classica, ingestao, relatorio, simulacao

EXEMPLOS = Path(__file__).resolve().parent.parent / "exemplos"


# ---------- ingestão ----------

def test_corrigir_branco_e_dupla_marcacao_contam_como_erro():
    respostas = pd.Series(["ABCDE", "A.C*E", "EDCBA"])
    matriz = ingestao.corrigir(respostas, "ABCDE")
    assert matriz.values.tolist() == [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]]


def test_carregar_csv_com_gabarito():
    matriz = ingestao.carregar_csv(EXEMPLOS / "respostas_exemplo.csv", EXEMPLOS / "gabarito_exemplo.csv")
    assert matriz.shape == (12, 6)
    assert matriz.loc["a01"].tolist() == [1, 1, 1, 1, 1, 0]


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
