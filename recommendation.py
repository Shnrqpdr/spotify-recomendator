"""Lógica de recomendação por distância euclidiana no espaço PCA."""

import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def recommend(projection_m, dados_musicas, nome_musica, top_n=10):
    """
    Recomenda as top_n músicas mais próximas (no espaço das duas primeiras
    componentes PCA) da música dada por nome (artists_song).

    Retorna dataframe com colunas relevantes (song, id, distancias, etc.).
    """
    cluster = projection_m[projection_m['song'] == nome_musica]['cluster_pca'].iloc[0]
    musicas_recomendadas = projection_m[projection_m['cluster_pca'] == cluster][[0, 1, 'song']].copy()

    x_musica = projection_m.loc[projection_m['song'] == nome_musica, 0].iloc[0]
    y_musica = projection_m.loc[projection_m['song'] == nome_musica, 1].iloc[0]

    distancias = euclidean_distances(
        musicas_recomendadas[[0, 1]],
        [[x_musica, y_musica]]
    )
    musicas_recomendadas['id'] = dados_musicas['id']
    musicas_recomendadas['distancias'] = distancias

    return musicas_recomendadas.sort_values('distancias').head(top_n)
