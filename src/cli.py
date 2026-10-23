"""Ponto de entrada: python -m src.cli <simular|enem|csv|tri> [opções]"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import ingestao, relatorio, simulacao, tri


def _analisar(respostas: pd.DataFrame, saida: str) -> pd.DataFrame:
    itens, confiab = relatorio.classificar_itens(respostas)
    resumo = relatorio.resumir_prova(respostas, itens, confiab)
    relatorio.imprimir(itens, resumo)
    pasta = relatorio.salvar(itens, resumo, saida)
    print(f"\nResultados salvos em {pasta}/ (itens.csv, prova.json)")
    return itens


def cmd_simular(args: argparse.Namespace) -> None:
    respostas, parametros, _ = simulacao.simular_prova(
        n_alunos=args.alunos, n_itens=args.questoes, n_problematicas=args.problematicas, seed=args.seed
    )
    itens = _analisar(respostas, args.saida)

    # Validação: as métricas clássicas recuperam os parâmetros verdadeiros?
    validacao = parametros.join(itens[["dificuldade", "discriminacao", "qualidade"]])
    validacao.round(4).to_csv(f"{args.saida}/validacao_simulacao.csv")
    r_dif = np.corrcoef(validacao["b"], validacao["dificuldade"])[0, 1]
    normais = validacao[~validacao["problematica"]]
    r_disc = np.corrcoef(normais["a"], normais["discriminacao"])[0, 1]
    detectadas = validacao.loc[validacao["problematica"], "qualidade"].eq("problemática").sum()

    print("\n=== VALIDAÇÃO (parâmetros verdadeiros vs. medidos) ===")
    print(f"Correlação b (dificuldade TRI) x % de acerto: {r_dif:+.3f}  (esperado: forte e negativa)")
    print(f"Correlação a (discriminação TRI) x índice D:   {r_disc:+.3f}  (esperado: positiva)")
    print(f"Questões problemáticas detectadas: {detectadas}/{int(validacao['problematica'].sum())}")


def cmd_enem(args: argparse.Namespace) -> None:
    print(f"Lendo {args.arquivo} ({ingestao.AREAS_ENEM[args.area.upper()]})... pode demorar alguns minutos.")
    respostas, nota, co_prova = ingestao.carregar_enem(
        args.arquivo, area=args.area, n_alunos=args.alunos, co_prova=args.caderno, seed=args.seed
    )
    print(f"Caderno {co_prova}: {len(respostas)} alunos na amostra.")
    _analisar(respostas, args.saida)

    acertos = respostas.sum(axis=1)
    validos = nota.notna()
    r = np.corrcoef(acertos[validos], nota[validos])[0, 1]
    print(f"\nCorrelação nº de acertos x nota oficial (TRI): {r:+.3f}")


def cmd_csv(args: argparse.Namespace) -> None:
    respostas = ingestao.carregar_csv(args.respostas, args.gabarito)
    _analisar(respostas, args.saida)


def cmd_tri(args: argparse.Namespace) -> None:
    if args.itens and not args.enem:
        raise SystemExit("--itens só funciona junto com --enem (os parâmetros oficiais são de um caderno do ENEM).")
    verdadeiros = nota = co_prova = None
    if args.csv:
        respostas = ingestao.carregar_csv(args.csv, args.gabarito)
    elif args.enem:
        print(f"Lendo {args.enem} ({ingestao.AREAS_ENEM[args.area.upper()]})... pode demorar alguns minutos.")
        respostas, nota, co_prova = ingestao.carregar_enem(
            args.enem, area=args.area, n_alunos=args.alunos, co_prova=args.caderno, seed=args.seed
        )
        print(f"Caderno {co_prova}: {len(respostas)} alunos na amostra.")
    else:
        respostas, verdadeiros, theta_verdadeiro = simulacao.simular_prova(
            n_alunos=args.alunos, n_itens=args.questoes, n_problematicas=args.problematicas, seed=args.seed
        )

    if len(respostas) < 500:
        print(f"Aviso: {len(respostas)} alunos é pouco para o 3PL; as estimativas ficam instáveis (ideal: 1000+).")
    print(f"Estimando o 3PL ({len(respostas)} alunos × {respostas.shape[1]} questões)...")
    itens, alunos = tri.ajustar_3pl(respostas)
    if not itens.attrs["convergiu"]:
        print("Aviso: a otimização não convergiu; trate os parâmetros com cautela.")

    print("\n=== PARÂMETROS TRI (3PL) ===")
    with pd.option_context("display.max_rows", None):
        print(itens.round(3).to_string())
    negativos = itens.index[itens["a"] < 0].tolist()
    if negativos:
        print(f"\nDiscriminação negativa (revisar gabarito/enunciado): {', '.join(negativos)}")

    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    itens.round(4).to_csv(pasta / "itens_tri.csv", encoding="utf-8")
    print(f"\nResultados salvos em {pasta}/itens_tri.csv")

    if verdadeiros is not None:
        normais = ~verdadeiros["problematica"]
        print("\n=== VALIDAÇÃO (parâmetros verdadeiros vs. estimados, questões normais) ===")
        for col in ("a", "b", "c"):
            r = np.corrcoef(verdadeiros.loc[normais, col], itens.loc[normais, col])[0, 1]
            print(f"Correlação {col}: {r:+.3f}")
        print(f"Correlação θ: {np.corrcoef(theta_verdadeiro, alunos['theta'])[0, 1]:+.3f}")
    if nota is not None:
        validos = nota.notna()
        r = np.corrcoef(alunos.loc[validos, "theta"], nota[validos])[0, 1]
        print(f"\nCorrelação θ estimado x nota oficial (TRI do INEP): {r:+.3f}")
    if args.itens:
        _validar_inep(itens, args.itens, args.area, co_prova, pasta)


def _validar_inep(itens: pd.DataFrame, caminho_itens: str, area: str, co_prova: int, pasta: Path) -> None:
    oficiais = ingestao.carregar_itens_inep(caminho_itens, area, co_prova)
    tabela, resumo = tri.comparar_com_inep(itens, oficiais)
    tabela.round(4).to_csv(pasta / "validacao_inep.csv", encoding="utf-8")
    grafico = relatorio.grafico_b_inep(tabela, resumo, pasta / "b_vs_inep.png")

    print(f"\n=== VALIDAÇÃO CONTRA O INEP (caderno {co_prova}, {resumo['n_comparados']} itens) ===")
    if resumo["anulados"]:
        print(f"Itens anulados (fora da comparação): {', '.join(resumo['anulados'])}")
    print("            Pearson  Spearman")
    for p in ("a", "b", "c"):
        print(f"  {p}         {resumo[f'pearson_{p}']:+.3f}    {resumo[f'spearman_{p}']:+.3f}")
    print(f"Escala (média-desvio): b_inep ≈ {resumo['escala_A']:.3f}·b {resumo['escala_B']:+.3f}")
    print(f"RMSE do b convertido: {resumo['rmse_b_convertido']:.3f}")
    piores = tabela.loc[
        tabela["dif_b"].abs().nlargest(5).index, ["b_convertido", "b_inep", "a", "a_inep", "c", "c_inep"]
    ]
    print("\nItens com maior diferença em b (já na escala do INEP):")
    print(piores.round(3).to_string())
    print(f"\nResultados salvos em {pasta}/validacao_inep.csv e {grafico}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="analise-provas", description="Análise de dificuldade de questões e provas.")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_sim = sub.add_parser("simular", help="Gera uma prova simulada (3PL) e analisa")
    p_sim.add_argument("--alunos", type=int, default=2000)
    p_sim.add_argument("--questoes", type=int, default=30)
    p_sim.add_argument("--problematicas", type=int, default=2)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.set_defaults(func=cmd_simular)

    p_enem = sub.add_parser("enem", help="Analisa os microdados do ENEM")
    p_enem.add_argument("arquivo", help="Caminho do MICRODADOS_ENEM_AAAA.csv")
    p_enem.add_argument("--area", default="MT", choices=["MT", "CN", "CH", "mt", "cn", "ch"])
    p_enem.add_argument("--alunos", type=int, default=10000, help="Tamanho da amostra (0 = todos)")
    p_enem.add_argument("--caderno", type=int, default=None, help="CO_PROVA (padrão: o mais comum)")
    p_enem.add_argument("--seed", type=int, default=42)
    p_enem.set_defaults(func=cmd_enem)

    p_csv = sub.add_parser("csv", help="Analisa uma prova própria em CSV")
    p_csv.add_argument("respostas", help="CSV com coluna 'aluno' + uma coluna por questão")
    p_csv.add_argument("--gabarito", default=None, help="CSV com colunas questao,gabarito")
    p_csv.set_defaults(func=cmd_csv)

    p_tri = sub.add_parser("tri", help="Estima o 3PL (a, b, c e θ). Sem --csv/--enem, usa uma prova simulada")
    fonte = p_tri.add_mutually_exclusive_group()
    fonte.add_argument("--csv", default=None, help="CSV de respostas (mesmo formato do comando csv)")
    fonte.add_argument("--enem", default=None, help="Caminho do MICRODADOS_ENEM_AAAA.csv")
    p_tri.add_argument("--gabarito", default=None, help="Gabarito do --csv")
    p_tri.add_argument(
        "--itens", default=None, help="ITENS_PROVA_AAAA.csv do INEP (com --enem): compara com os parâmetros oficiais"
    )
    p_tri.add_argument("--area", default="MT", choices=["MT", "CN", "CH", "mt", "cn", "ch"])
    p_tri.add_argument("--caderno", type=int, default=None, help="CO_PROVA do --enem (padrão: o mais comum)")
    p_tri.add_argument("--alunos", type=int, default=2000, help="Alunos simulados ou amostra do ENEM")
    p_tri.add_argument("--questoes", type=int, default=30)
    p_tri.add_argument("--problematicas", type=int, default=2)
    p_tri.add_argument("--seed", type=int, default=42)
    p_tri.set_defaults(func=cmd_tri)

    for p in (p_sim, p_enem, p_csv, p_tri):
        p.add_argument("--saida", default="saida", help="Pasta dos resultados")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
