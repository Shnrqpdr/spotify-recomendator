"""Funções de visualização."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_year_comparison(dados_por_ano):
    """Gráficos de Loudness e Popularity por ano."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))

    axes[0].plot(
        dados_por_ano['year'], dados_por_ano['loudness'],
        marker='o', linewidth=2.0, label="Loudness by year"
    )
    axes[0].set_ylabel("Loudness")
    axes[0].set_title('Loudness by year comparison')
    axes[0].grid(axis='y', linestyle='-', linewidth=0.5)
    axes[0].grid(axis='x', linestyle='-', linewidth=0.5)
    axes[0].legend(loc='upper right')

    axes[1].plot(
        dados_por_ano['year'], dados_por_ano['popularity'],
        marker='o', linewidth=2.0, label="Popularity by year"
    )
    axes[1].set_ylabel("Popularity")
    axes[1].set_title('Popularity by year comparison')
    axes[1].grid(axis='y', linestyle='-', linewidth=0.5)
    axes[1].grid(axis='x', linestyle='-', linewidth=0.5)
    axes[1].legend(loc='upper left')

    return fig, axes


def plot_correlation_matrix(dados):
    """Matriz de correlação com valores anotados."""
    f = plt.figure(figsize=(10, 10))
    plt.matshow(dados.corr(), fignum=f.number)
    plt.xticks(range(dados.shape[1]), dados.columns, fontsize=8, rotation=45)
    plt.yticks(range(dados.shape[1]), dados.columns, fontsize=8)

    for (i, j), z in np.ndenumerate(dados.corr()):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color='black')

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    return f


def plot_genre_pca(projection):
    """Scatter da projeção PCA de gêneros (sem clusters)."""
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes.scatter(projection['x'], projection['y'], marker='o', label="Our PCA projection")
    axes.set_ylabel("y")
    axes.set_xlabel("x")
    axes.set_title('PCA Projection')
    axes.grid(axis='y', linestyle='-', linewidth=0.5)
    axes.grid(axis='x', linestyle='-', linewidth=0.5)
    axes.legend(loc='upper right')
    return fig, axes


def plot_genre_clusters(projection, n_clusters=5):
    """Scatter da projeção PCA de gêneros colorido por cluster."""
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(n_clusters):
        x = projection[projection['cluster_pca'] == i]['x']
        y = projection[projection['cluster_pca'] == i]['y']
        axes.scatter(x, y, marker='o', label=f"Cluster {i+1}")

    axes.set_ylabel("y")
    axes.set_xlabel("x")
    axes.set_title('Clustering PCA Projection')
    axes.grid(axis='y', linestyle='-', linewidth=0.5)
    axes.grid(axis='x', linestyle='-', linewidth=0.5)
    axes.legend(loc='upper right')
    return fig, axes


def plot_music_clusters_3d(projection_m, n_clusters=50):
    """Gráfico 3D dos clusters de músicas (três primeiras componentes)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    for i in range(n_clusters):
        mask = projection_m['cluster_pca'] == i
        x = projection_m.loc[mask, 0]
        y = projection_m.loc[mask, 1]
        z = projection_m.loc[mask, 2]
        ax.scatter(x, y, z, label=f'Cluster {i}')

    ax.set_title('Clusters - Projection of 3 first features of PCA pipeline')
    ax.set_xlabel('Column $0$')
    ax.set_ylabel('Column $1$')
    ax.set_zlabel('Column $2$')
    return fig, ax
