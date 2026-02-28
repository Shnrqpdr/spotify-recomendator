"""Configurações e constantes do projeto."""

SEED = 1224

# Colunas removidas no pré-processamento
COLUNAS_DROP_MUSICAS = ['explicit', 'key', 'mode']
COLUNAS_DROP_GENEROS = ['mode', 'key']
COLUNAS_DROP_ANO = ['mode', 'key']

# Features numéricas para análise de correlação
FEATURES_CORRELACAO = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'popularity'
]

# Parâmetros dos modelos
ANO_MINIMO = 2000
GENRE_PCA_COMPONENTS = 2
GENRE_N_CLUSTERS = 5
MUSIC_PCA_VARIANCE = 0.7
MUSIC_N_CLUSTERS = 50
