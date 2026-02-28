"""Pipelines e modelos: PCA, KMeans, pré-processamento."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    GENRE_N_CLUSTERS,
    GENRE_PCA_COMPONENTS,
    MUSIC_N_CLUSTERS,
    MUSIC_PCA_VARIANCE,
    SEED,
)


def build_genre_pca_pipeline(n_components=GENRE_PCA_COMPONENTS):
    """Pipeline de escala + PCA para embedding de gêneros."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('PCA', PCA(n_components=n_components, random_state=SEED))
    ])


def fit_genre_embedding(dados_generos_numericos, pipeline):
    """Ajusta o pipeline e retorna a projeção 2D (x, y)."""
    embedding = pipeline.fit_transform(dados_generos_numericos)
    return pd.DataFrame(columns=['x', 'y'], data=embedding)


def fit_genre_clustering(projection, dados_por_generos, n_clusters=GENRE_N_CLUSTERS):
    """Ajusta KMeans na projeção de gêneros e adiciona cluster_pca aos dataframes."""
    kmeans = KMeans(n_clusters=n_clusters, verbose=True, random_state=SEED)
    kmeans.fit(projection)

    labels = kmeans.predict(projection)
    projection = projection.copy()
    projection['cluster_pca'] = labels
    dados_por_generos = dados_por_generos.copy()
    dados_por_generos['cluster_pca'] = labels

    return kmeans, projection, dados_por_generos


def prepare_music_dummies(dados_musicas):
    """
    Aplica OneHotEncoder em 'artists' e retorna dataframe com colunas dummies.
    """
    ohe = OneHotEncoder(dtype=int)
    colunas_ohe = ohe.fit_transform(dados_musicas[['artists']]).toarray()
    dados_artists = dados_musicas.drop('artists', axis=1)
    return pd.concat([
        dados_artists,
        pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['artists']))
    ], axis=1)


def build_music_pca_pipeline(n_components=MUSIC_PCA_VARIANCE):
    """Pipeline de escala + PCA para músicas (variância explicada)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('PCA', PCA(n_components=n_components, random_state=SEED))
    ])


def fit_music_embedding(dados_musicas_dummies, pipeline):
    """
    Ajusta o pipeline nos dados numéricos (sem id, name, artists_song)
    e retorna o dataframe de projeção.
    """
    X = dados_musicas_dummies.drop(['id', 'name', 'artists_song'], axis=1)
    embedding = pipeline.fit_transform(X)
    return pd.DataFrame(data=embedding)


def fit_music_clustering(projection_m, dados_musicas, n_clusters=MUSIC_N_CLUSTERS):
    """
    Ajusta KMeans na projeção de músicas, adiciona cluster_pca, artist e song.
    Modifica dados_musicas in-place com cluster_pca. Retorna (kmeans, projection_m).
    """
    kmeans = KMeans(n_clusters=n_clusters, verbose=False, random_state=SEED)
    kmeans.fit(projection_m)

    labels = kmeans.predict(projection_m)
    projection_m = projection_m.copy()
    projection_m['cluster_pca'] = labels
    projection_m['artist'] = dados_musicas['artists'].values
    projection_m['song'] = dados_musicas['artists_song'].values

    dados_musicas['cluster_pca'] = labels

    return kmeans, projection_m
