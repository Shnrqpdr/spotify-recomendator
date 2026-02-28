# Spotify Recomendator

Recomendações de músicas usando **aprendizado não supervisionado**: PCA e K-Means sobre dados de áudio da [Web API do Spotify](https://developer.spotify.com/documentation/web-api). O projeto replica em linhas gerais a ideia de sistemas de recomendação baseados em clusterização.

As [audio features](https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features) utilizadas incluem: `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `popularity`, entre outras.

---

## O que o repositório contém

- **Código em Python** (refatorado a partir do notebook): carregamento de dados, pré-processamento, pipelines de PCA/K-Means, visualizações e lógica de recomendação.
- **Notebook** `spotify-nb.ipynb`: apenas células de texto (markdown) com a narrativa do projeto; o código foi migrado para os módulos `.py`.
- **Dados**: três CSVs na raiz do repositório (ver [Dados](#dados)).

---

## Estrutura do projeto

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Script principal: carrega dados, treina modelos, gera gráficos e exibe um exemplo de recomendação. |
| `config.py` | Constantes: seed, colunas removidas, features de correlação, parâmetros (PCA, número de clusters, ano mínimo). |
| `data.py` | Carregamento dos CSVs e preparação: filtro de ano (≥ 2000), remoção de colunas, helpers para análise. |
| `model.py` | Pipelines e ajuste: PCA + K-Means para **gêneros**; OneHotEncoder + PCA + K-Means para **músicas**. |
| `plots.py` | Funções de plot: loudness/popularity por ano, matriz de correlação, projeção PCA de gêneros, clusters 2D e 3D. |
| `recommendation.py` | Função `recommend()`: dado o nome de uma música, retorna as N mais próximas no espaço PCA (distância euclidiana). |
| `spotify-nb.ipynb` | Notebook com a narrativa do projeto (sem código; execução via `main.py` e módulos). |

---

## O que foi feito (pipeline)

1. **Carregamento e pré-processamento**
   - Leitura de `Dados_totais.csv`, `data_by_genres.csv` e `data_by_year.csv`.
   - Filtro de `dados_por_ano` para anos ≥ 2000 (alinhado ao período das músicas).
   - Remoção de colunas (`explicit`, `key`, `mode` em músicas; `mode`, `key` em gêneros e por ano).

2. **Análise exploratória**
   - Gráficos de **loudness** e **popularity** por ano.
   - **Matriz de correlação** das features numéricas das músicas.

3. **Gêneros**
   - PCA (2 componentes) sobre as features numéricas por gênero.
   - K-Means (5 clusters) na projeção; visualização em 2D.

4. **Músicas**
   - OneHotEncoder em `artists`; remoção de `id`, `name`, `artists_song` para o PCA.
   - PCA com 70% da variância explicada; K-Means com 50 clusters na projeção.
   - Visualização 3D das três primeiras componentes.

5. **Recomendação**
   - Para uma música de referência (ex.: `"OutKast - Ms. Jackson"`): localizar o cluster, restringir às músicas do mesmo cluster e ranquear pelas **distâncias euclidianas** nas duas primeiras componentes PCA; retornar as top N.

---

## Dados

Os arquivos de entrada devem estar na raiz do projeto (ou os caminhos alterados em `data.load_data()`):

| Arquivo | Conteúdo |
|---------|----------|
| `Dados_totais.csv` | Músicas: features de áudio, artista, nome, ano, popularidade, etc. |
| `data_by_genres.csv` | Médias das features por **gênero**. |
| `data_by_year.csv` | Médias das features por **ano**. |

Para obter dados da Web API do Spotify, siga a [documentação oficial](https://developer.spotify.com/documentation/web-api).

---

## Requisitos

- Python 3
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

Instalação sugerida (ambiente virtual):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install pandas numpy scikit-learn matplotlib
```

---

## Como rodar

Na raiz do repositório, com os CSVs no lugar:

```bash
python main.py
```

O script irá:

- Carregar e preparar os dados.
- Gerar os gráficos (loudness/popularity por ano, correlação, PCA de gêneros, clusters 2D, clusters 3D de músicas).
- Treinar os pipelines de gêneros e músicas.
- Imprimir as 10 músicas recomendadas para o exemplo `"OutKast - Ms. Jackson"`.

Para usar o sistema de recomendação em código:

```python
from data import load_data, prepare_data
from model import (
    build_music_pca_pipeline, fit_music_embedding, fit_music_clustering,
    prepare_music_dummies,
)
from recommendation import recommend

dados_musicas, dados_por_generos, dados_por_ano = load_data()
dados_musicas, dados_por_generos, dados_por_ano = prepare_data(
    dados_musicas, dados_por_generos, dados_por_ano
)
dados_musicas_dummies = prepare_music_dummies(dados_musicas)
pipeline = build_music_pca_pipeline()
projection_m = fit_music_embedding(dados_musicas_dummies, pipeline)
_, projection_m = fit_music_clustering(projection_m, dados_musicas)

recomendadas = recommend(projection_m, dados_musicas, "Nome Artista - Nome da Música", top_n=10)
```

---

## Parâmetros (config.py)

| Constante | Valor padrão | Uso |
|-----------|--------------|-----|
| `SEED` | 1224 | Reproducibilidade (numpy, PCA, K-Means). |
| `ANO_MINIMO` | 2000 | Filtro de anos em `dados_por_ano`. |
| `GENRE_PCA_COMPONENTS` | 2 | Componentes PCA para gêneros. |
| `GENRE_N_CLUSTERS` | 5 | Número de clusters de gêneros. |
| `MUSIC_PCA_VARIANCE` | 0.7 | Variância explicada no PCA de músicas. |
| `MUSIC_N_CLUSTERS` | 50 | Número de clusters de músicas. |

Alterar esses valores em `config.py` e rodar `main.py` de novo aplica as mudanças em todo o pipeline.

---

## Projeto de estudo

Este repositório é um projeto de estudo. Sinta-se à vontade para usar e adaptar o código. As features de áudio e a documentação da API estão em:  
https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features
