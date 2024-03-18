import json
import string
import time
from multiprocessing.pool import ThreadPool

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
from spacy import Language

from compare_clustering_solutions import evaluate_clustering


# prepare the spacy pipeline for the naming part of the task
def prepare_spacy_pipeline() -> Language:
    # load the spaCy model
    benepar.download('benepar_en3')
    nlp = en_core_web_md.load()
    # load the benepar parser
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    return nlp


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
def title_clusters_dataframe(df: DataFrame, nlp: Language):
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
                 title: str = 'Clustering Results',
                 axis: tuple[str, str, str] | None = ('reduced0', 'reduced1', 'reduced2'),
                 color: str = 'cluster') -> Figure:
    x, y, z = axis
    fig = px.scatter_3d(df, x=x, y=y, z=z,
                        title=title,
                        hover_data={
                            'text': True,
                            'title': True,
                            x: True,
                            y: True,
                            z: True
                        },
                        size_max=1.5,
                        color=color if color in df else None)
    fig.show()
    return fig


# read the data and initialize the dataframe
def init_dataframe(data_file: str, model: SentenceTransformer) -> DataFrame:
    df = read_lines(data_file)
    df['cluster'] = -1
    encode_sentences(df, model)
    find_closest_neighbors(df)
    return df


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
def count_within_range(df: DataFrame, eps: float, unclustered_only: bool = False):
    df['within_range'] = 0
    for index, row in df.iterrows():
        for index2, row2 in df[df.index > index].iterrows():
            if unclustered_only and row2['cluster'] != -1:
                continue
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
    df.at[index, 'tried_to_cluster'] = True

    neighbors = find_neighbors(df, index, eps)

    return sum(try_add_to_cluster(df,
                                  neighbor,
                                  cluster_id,
                                  eps
                                  ) for neighbor in neighbors) + 1


# my implementation of the DBSCAN clustering algorithm
def dbscan_clustering(df: DataFrame,
                      quantile: float,
                      min_size: int,
                      sort: bool = True,
                      prioritize_unclustered: bool = True,
                      prioritize_existing_clusters: bool = False,
                      count_unclustered_only: bool = True,
                      create_clusters: bool = True) -> DataFrame:
    # shuffle the dataframe
    df = df.sample(frac=1)
    df['tried_to_cluster'] = False
    df['visited'] = False

    # count the number of sentences within a certain range of each sentence
    count_within_range(df, df['closest_neighbor_distance'].quantile(quantile), count_unclustered_only)
    print(f"Max number of sentences within range: {df['within_range'].max()}")
    print(f"Max number within range for clustered: {df[df['cluster'] != -1]['within_range'].max()}")

    if sort:
        df.sort_values(by='within_range', inplace=True, ascending=False, ignore_index=True)
        df.reset_index(drop=True, inplace=True)

    # go through the dataframe and cluster the sentences
    eps = df['closest_neighbor_distance'].quantile(quantile)
    for i in range(3):
        # if prioritize_unclustered is False, skip the first iteration (meant to prioritize the unclustered sentences)
        if i == 0 and not prioritize_unclustered:
            continue
        if i == 1 and not prioritize_existing_clusters:
            continue

        for index in range(len(df)):
            new_cluster = False

            if (df.at[index, 'visited'] or
                    df.at[index, 'tried_to_cluster'] or
                    (i == 0 and prioritize_unclustered and df.at[index, 'cluster'] != -1) or
                    (i == 1 and prioritize_existing_clusters and df.at[index, 'cluster'] == -1)):
                continue
            elif (t_cluster_id := df.at[index, 'cluster']) != -1:
                cluster_id = t_cluster_id
            else:
                if not create_clusters:
                    continue
                cluster_id = df['cluster'].max() + 1
                new_cluster = True

            cluster_size = try_add_to_cluster(df, index, cluster_id, eps)
            if new_cluster and cluster_size < min_size:
                df.loc[df['cluster'] == cluster_id, 'cluster'] = -1
                df.loc[df['cluster'] == cluster_id, 'visited'] = False

    df.drop(columns=['tried_to_cluster'], inplace=True)
    return df


# save the clustering results to a json file
def dataframe_to_json(df: DataFrame, output_file: str) -> dict:
    # create the cluster list and the unclustered list
    cluster_list = []
    unclustered = []

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


# linear interpolation between 2 values
def linear_interpolation(min_val: float, max_val: float, steps: int, step: int) -> float:
    return min_val + (((max_val - min_val) / steps) * step)


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # prepare the spacy pipeline for the naming part of the task
    pool = ThreadPool(processes=1)
    nlp_async = pool.apply_async(prepare_spacy_pipeline)

    # load the model
    model = create_sentence_transformer_model()

    # read the data and initialize the dataframe
    df = init_dataframe(data_file, model)

    # initialize the clustering parameters
    max_iter_new_clusters = 2
    max_iter_existing_clusters = 2
    curr_iter = 0
    min_quantile = 0.6  # 0.55
    max_quantile_new_clusters = 0.7
    max_quantile = 0.9

    # todo: remove timing code
    start_time = time.time()

    # try to cluster the sentences
    while curr_iter < max_iter_new_clusters:
        curr_quantile = linear_interpolation(min_val=min_quantile,
                                             max_val=max_quantile_new_clusters,
                                             steps=max_iter_new_clusters - 1,
                                             step=curr_iter)
        print(f"Starting to cluster the unclustered sentences using the {round(curr_quantile, 2)} quantile as radius.")

        df = dbscan_clustering(df, quantile=curr_quantile, min_size=int(min_size),
                               prioritize_unclustered=True,
                               prioritize_existing_clusters=False,
                               create_clusters=True)
        print(f"Found {df['cluster'].nunique() - 1} clusters.\n"
              f"{df[df['cluster'] != -1]['text'].count()} clustered and "
              f"{df[df['cluster'] == -1]['text'].count()} unclustered sentences.")

        curr_iter += 1
        print(f"iteration {curr_iter} complete.\n")

    print("Now prioritizing matching to existing clusters rather than creating new ones.\n")

    # try to match the remaining sentences to existing clusters
    while curr_iter < max_iter_new_clusters + max_iter_existing_clusters:
        curr_quantile = linear_interpolation(min_val=max_quantile_new_clusters,
                                             max_val=max_quantile,
                                             steps=max_iter_existing_clusters,
                                             step=curr_iter - max_iter_new_clusters + 1)
        print(f"Starting to cluster the unclustered sentences using the {round(curr_quantile, 2)} quantile as radius.")

        df = dbscan_clustering(df, quantile=curr_quantile, min_size=int(min_size),
                               prioritize_unclustered=False,
                               prioritize_existing_clusters=True,
                               create_clusters=True)
        print(f"Found {df['cluster'].nunique() - 1} clusters.\n"
              f"{df[df['cluster'] != -1]['text'].count()} clustered and "
              f"{df[df['cluster'] == -1]['text'].count()} unclustered sentences.")

        curr_iter += 1
        print(f"iteration {curr_iter} complete.\n")

    print(f"Clustering time: {round(time.time() - start_time, 0)} seconds.")

    # todo: remove timing code
    start_time = time.time()

    # create visualization base
    dims = 3
    reduce_dimensions(df, dims)

    # wait for the spacy pipeline to be ready
    nlp = nlp_async.get()

    # give each cluster a title based on the sentences in the cluster
    title_clusters_dataframe(df, nlp)

    # save the clustering results to a json file
    dataframe_to_json(df, output_file)

    # plot the results
    fig = plot_results(df, color='title')
    fig.write_html('clustering_results.html')

    print(f"Naming and visualization time: {round(time.time() - start_time, 0)} seconds.")


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
