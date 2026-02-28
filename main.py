"""Script principal: carrega dados, treina modelos e exibe recomendações."""

from config import (
    GENRE_N_CLUSTERS,
    MUSIC_N_CLUSTERS,
)
from data import (
    get_dados_generos_numericos,
    get_features_correlacao,
    load_data,
    prepare_data,
)
from model import (
    build_genre_pca_pipeline,
    build_music_pca_pipeline,
    fit_genre_clustering,
    fit_genre_embedding,
    fit_music_clustering,
    fit_music_embedding,
    prepare_music_dummies,
)
from plots import (
    plot_correlation_matrix,
    plot_genre_clusters,
    plot_genre_pca,
    plot_music_clusters_3d,
    plot_year_comparison,
)
from recommendation import recommend


def main():
    # Carregar e preparar dados
    dados_musicas, dados_por_generos, dados_por_ano = load_data()
    dados_musicas, dados_por_generos, dados_por_ano = prepare_data(
        dados_musicas, dados_por_generos, dados_por_ano
    )

    # Análise exploratória: gráficos por ano e matriz de correlação
    plot_year_comparison(dados_por_ano)
    dados = get_features_correlacao(dados_musicas)
    plot_correlation_matrix(dados)

    # Pipeline e clustering de gêneros
    dados_generos1 = get_dados_generos_numericos(dados_por_generos)
    genre_pipeline = build_genre_pca_pipeline()
    projection = fit_genre_embedding(dados_generos1, genre_pipeline)
    kmeans_genre, projection, dados_por_generos = fit_genre_clustering(
        projection, dados_por_generos, n_clusters=GENRE_N_CLUSTERS
    )
    projection['genres'] = dados_por_generos['genres'].values

    plot_genre_pca(projection)
    print('Ratio: ', genre_pipeline[1].explained_variance_ratio_.sum())
    print('Features explained: ', genre_pipeline[1].explained_variance_.sum())

    plot_genre_clusters(projection, n_clusters=GENRE_N_CLUSTERS)

    # Pipeline e clustering de músicas
    dados_musicas_dummies = prepare_music_dummies(dados_musicas)
    music_pipeline = build_music_pca_pipeline()
    projection_m = fit_music_embedding(dados_musicas_dummies, music_pipeline)
    kmeans_music, projection_m = fit_music_clustering(
        projection_m, dados_musicas, n_clusters=MUSIC_N_CLUSTERS
    )

    plot_music_clusters_3d(projection_m, n_clusters=MUSIC_N_CLUSTERS)

    # Exemplo de recomendação
    nome_musica = 'OutKast - Ms. Jackson'
    recomendada = recommend(projection_m, dados_musicas, nome_musica, top_n=10)
    print(recomendada)

    return dados_musicas, dados_por_generos, dados_por_ano, projection, projection_m, recomendada


if __name__ == '__main__':
    main()
