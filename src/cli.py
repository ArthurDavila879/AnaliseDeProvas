"""Ponto de entrada: python -m src.cli <simular|enem|csv> [opções]"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import ingestao, relatorio, simulacao


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

    for p in (p_sim, p_enem, p_csv):
        p.add_argument("--saida", default="saida", help="Pasta dos resultados")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
