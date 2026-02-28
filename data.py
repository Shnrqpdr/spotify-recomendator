"""Carregamento e preparação dos dados."""

import numpy as np
import pandas as pd

from config import (
    ANO_MINIMO,
    COLUNAS_DROP_ANO,
    COLUNAS_DROP_GENEROS,
    COLUNAS_DROP_MUSICAS,
    FEATURES_CORRELACAO,
    SEED,
)


def load_data(
    path_musicas: str = './Dados_totais.csv',
    path_generos: str = './data_by_genres.csv',
    path_ano: str = './data_by_year.csv',
):
    """Carrega os três dataframes a partir dos CSVs."""
    dados_musicas = pd.read_csv(path_musicas)
    dados_por_generos = pd.read_csv(path_generos)
    dados_por_ano = pd.read_csv(path_ano)
    return dados_musicas, dados_por_generos, dados_por_ano


def prepare_data(dados_musicas, dados_por_generos, dados_por_ano):
    """
    Aplica filtros e remoção de colunas.
    Retorna (dados_musicas, dados_por_generos, dados_por_ano) preparados.
    """
    np.random.seed(SEED)

    dados_por_ano = dados_por_ano[dados_por_ano['year'] >= ANO_MINIMO].reset_index(drop=True)

    dados_musicas = dados_musicas.drop(COLUNAS_DROP_MUSICAS, axis=1)
    dados_por_generos = dados_por_generos.drop(COLUNAS_DROP_GENEROS, axis=1)
    dados_por_ano = dados_por_ano.drop(COLUNAS_DROP_ANO, axis=1)

    return dados_musicas, dados_por_generos, dados_por_ano


def get_features_correlacao(dados_musicas):
    """Retorna o subset de colunas numéricas para matriz de correlação."""
    return dados_musicas[FEATURES_CORRELACAO].copy()


def get_dados_generos_numericos(dados_por_generos):
    """Remove a coluna 'genres' para obter apenas features numéricas."""
    return dados_por_generos.drop('genres', axis=1)
