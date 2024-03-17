import json
import string
from collections import defaultdict

# noinspection PyUnresolvedReferences
import benepar, spacy
import en_core_web_md
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.figure import Figure
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from compare_clustering_solutions import evaluate_clustering

# load the spaCy model
benepar.download('benepar_en3')
nlp = en_core_web_md.load()
# load the benepar parser
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


# clean the query text for processing
def clean_query(query: str) -> str:
    return query.lower().strip('\r\n')


# create a line generator from filename
def read_lines(filename: str) -> DataFrame:
    df = pd.read_csv(filename)
    df['text'] = df['text'].map(clean_query)
    return df


# create a sentence transformer model
def create_sentence_transformer_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')


# encode sentences using the sentence transformer model
def encode_sentences(df: DataFrame, model: SentenceTransformer):
    df['encoded'] = df['text'].map(lambda x: model.encode(x))


# give a title to a cluster
def title_clusters_dataframe(df: DataFrame):
    df['title'] = 'cluster_' + df['cluster'].astype(str)
    for cluster in df['cluster'].unique():
        if cluster == -1:
            df.loc[df['cluster'] == cluster, 'title'] = 'unclustered'
            continue

        # get all sentences in the cluster to a list
        sentence_list = df.loc[df['cluster'] == cluster, 'text'].to_list()
        # strip the sentences of any extra spaces (anywhere) and periods (at the end)
        sentence_list = [' '.join(sentence.split()).strip('.') for sentence in sentence_list]
        # concatenate the sentences and make sure that they end with punctuation
        paragraph = ' '.join([sentence if sentence.endswith(string.punctuation) else sentence + '.'
                              for sentence in sentence_list])
        # analyze the concatenated sentences (the entire cluster)
        doc = nlp(paragraph)

        # initialize the max similarity and the max string
        max_sim = float('-inf')
        max_str = ""

        # find the single-standalone sentence that is the most similar to all the sentences concatenated
        for sent in doc.sents:
            for constituent in sent._.constituents:
                if 'S' in constituent._.labels:
                    if (similarity := doc.similarity(constituent)) > max_sim:
                        max_sim = similarity
                        max_str = constituent.text

        # set the most similar as the title
        title = max_str.strip('.')
        if title != '':
            df.loc[df['cluster'] == cluster, 'title'] = title


# reduces the dimensionality of the data to k dimensions
def reduce_dimensions(df: DataFrame, dims: int):
    pca = PCA(n_components=dims)
    reduced = pca.fit_transform(df['encoded'].to_list())
    for i in range(dims):
        df[f'reduced{str(i)}'] = reduced[:, i]


# plot results - plotly
def plot_results(df: DataFrame,
                 axis: tuple[str, str, str] | None = ('reduced0', 'reduced1', 'reduced2'),
                 color: str = 'cluster') -> Figure:
    x, y, z = axis
    fig = px.scatter_3d(df, x=x, y=y, z=z,
                        size_max=1.5,
                        color=color if color in df else None)
    fig.show()
    return fig


# read the data and initialize the dataframe
def init_dataframe(data_file: str, model: SentenceTransformer) -> DataFrame:
    df = read_lines(data_file)
    df['visited'] = False
    df['cluster'] = -1
    df['radius'] = 0.0
    encode_sentences(df, model)
    find_closest_neighbors(df)
    return df


# delete the clusters that are too small
def filter_clusters(df: DataFrame, min_size: int) -> bool:
    # find the clusters that are too small
    clusters_index = df['cluster'].value_counts().lt(min_size)

    # set the cluster to -1 if it is too small (unclustered)
    df.loc[df['cluster'].map(lambda x: clusters_index[x]), ['cluster', 'visited']] = [-1, False]

    # renumber the clusters to be consecutive (not necessary per se, but good for my sanity)
    unique = set(df['cluster'].unique())
    unique.remove(-1)
    new_cluster_id = {cluster: index for index, cluster in enumerate(unique)}
    df['cluster'] = df['cluster'].map(lambda x: -1 if x == -1 else new_cluster_id[x])

    # return if any clusters were too small
    return any(clusters_index)


# find the closest neighbors of each sentence
def find_closest_neighbors(df: DataFrame):
    df['closest_to_number'] = 0
    for index, row in df.iterrows():
        min_distance = float('inf')
        min_index = -1
        for index2, row2 in df.iterrows():
            if index == index2:
                continue
            if (distance := np.linalg.norm(row['encoded'] - row2['encoded'])) < min_distance:
                min_distance = distance
                min_index = index2

        df.at[index, 'closest_neighbor_index'] = min_index
        df.at[index, 'closest_neighbor_distance'] = min_distance
        df.at[min_index, 'closest_to_number'] += 1


# count the number of sentences within a certain range of each sentence
def count_within_range(df: DataFrame, eps: float):
    df['within_range'] = 0
    for index, row in df.iterrows():
        for index2, row2 in df[df.index > index].iterrows():
            if np.linalg.norm(row['encoded'] - row2['encoded']) < eps:
                df.at[index, 'within_range'] += 1
                df.at[index2, 'within_range'] += 1


# find the neighbors of a sentence within a certain range
def find_neighbors(df: DataFrame, index: int, eps: float) -> set[int]:
    neighbors = set()
    for i in range(len(df)):
        if i == index or df.at[i, 'cluster'] != -1:
            continue
        if np.linalg.norm(df.at[index, 'encoded'] - df.at[i, 'encoded']) < eps:
            neighbors.add(i)
    return neighbors


# try to add sentences to a cluster recursively
def try_add_to_cluster(df: DataFrame, index: int, cluster_id: int, eps: float) -> int:
    if df.at[index, 'visited']:
        return 0

    df.at[index, 'visited'] = True
    df.at[index, 'cluster'] = cluster_id
    df.at[index, 'radius'] = eps

    neighbors = find_neighbors(df, index, eps)
    for neighbor in neighbors:
        df.at[neighbor, 'cluster'] = cluster_id

    return sum(try_add_to_cluster(df,
                                  neighbor,
                                  cluster_id,
                                  eps - np.linalg.norm(df.at[index, 'encoded'] - df.at[neighbor, 'encoded']) / 1.5
                                  ) for neighbor in neighbors) + 1


# my implementation of the DBSCAN clustering algorithm
def dbscan_clustering(df: DataFrame,
                      quantile: float = 0.95) -> DataFrame:
    # shuffle the dataframe
    df = df.sample(frac=1)
    count_within_range(df, df['closest_neighbor_distance'].quantile(quantile))
    df.sort_values(by='within_range', inplace=True, ascending=False, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    eps = df['closest_neighbor_distance'].quantile(quantile)
    cluster_id = df['cluster'].max() + 1
    for index in range(len(df)):
        if df.at[index, 'visited']:
            continue

        print(f"Starting cluster {cluster_id} with \'{df.at[index, 'text']}\'")
        print(f"Added {try_add_to_cluster(df, index, cluster_id, eps)} sentences to cluster {cluster_id}")
        cluster_id += 1

    return df


# reverse cluster sentences to existing clusters
def reverse_cluster(df: DataFrame) -> bool:
    converged: bool = True

    for index, row in df.iterrows():
        if row['cluster'] != -1:
            continue
        df2 = df[df['cluster'] != -1]
        df2['distance'] = df2['encoded'].map(lambda x: np.linalg.norm(row['encoded'] - x))
        df2.sort_values(by='distance', inplace=True, ascending=True)
        df2.reindex()
        df2 = df2[df2['distance'] < df2['radius']]
        if len(df2) > 0:
            converged = False
            index2 = df2.index[0]
            df.at[index, 'cluster'] = df2.at[index2, 'cluster']
            df.at[index, 'visited'] = True
            df.at[index, 'radius'] = df2.at[index2, 'radius'] - df2.at[index2, 'distance']

    return converged


# save the clustering results to a json file
def dataframe_to_json(df: DataFrame, output_file: str) -> dict:
    # create the cluster list and the unclustered list
    cluster_list = []
    unclustered = []
    # give each cluster a title based on the sentences in the cluster
    title_clusters_dataframe(df)

    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            unclustered.extend(df[df['cluster'] == cluster_id]['text'].to_list())
        else:
            cluster_list.append({
                "cluster_name": f"{df.loc[df['cluster'] == cluster_id, 'title'].iat[0]}",
                "requests": df[df['cluster'] == cluster_id]['text'].to_list()
            })

    results = {"cluster_list": cluster_list, "unclustered": unclustered}

    with open(output_file, 'w+', encoding='utf8') as fout:
        json.dump(results, fout, indent=4)

    return results


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # load the model
    model = create_sentence_transformer_model()

    # read the data and initialize the dataframe
    df = init_dataframe(data_file, model)

    # go through the dataframe and match the sentences to the clusters
    df = dbscan_clustering(df, 0.9)

    # try to match remaining sentences to existing clusters
    max_iter = 12
    curr_iter = 0
    while curr_iter < max_iter and filter_clusters(df, int(min_size)) and not reverse_cluster(df):
        curr_iter += 1
        print(f"iteration {curr_iter} complete.")

    if curr_iter == max_iter:
        print("The algorithm did not converge in the maximum number of iterations.")
    else:
        print(f"The algorithm converged in {curr_iter} iterations.")

    # create visualization base
    dims = 3
    reduce_dimensions(df, dims)

    # give each cluster a title based on the sentences in the cluster
    title_clusters_dataframe(df)

    # save the clustering results to a json file
    dataframe_to_json(df, output_file)

    # plot the results
    fig = plot_results(df, color='title')
    fig.write_html('clustering_results.html')


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
