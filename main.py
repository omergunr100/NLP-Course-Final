import json
from collections import defaultdict
from typing import Any, Dict

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
from torch import Tensor

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


# find the closest centroid to an encoded sentence
def find_closest_cluster(clusters: dict, sentence: Tensor, threshold: float, black_list: set | None = None):
    min_distance = float('inf')
    closest_cluster = None
    for key, value in clusters.items():
        if black_list and key in black_list:
            continue
        distance = np.sqrt(np.sum((value['centroid'] - sentence) ** 2))
        if distance < min_distance and distance < max(threshold, 1.5 * value['average_distance']):
            min_distance = distance
            closest_cluster = key
    return closest_cluster


# try to find clusters for sentences in a cluster that aren't the same cluster
def re_cluster_sentences(df: DataFrame, clusters: dict, threshold: float, cluster,
                         black_list: set = None) -> defaultdict:
    remapping = defaultdict(lambda: None)
    for index, row in df[df['id'].isin(clusters[cluster]['sentences'])].iterrows():
        if (closest := find_closest_cluster(clusters,
                                            row['encoded'],
                                            threshold,
                                            {cluster} if black_list is None else black_list.union({cluster})
                                            )) is not None:
            remapping[index] = closest

    return remapping


# move a sentence from one centroid to another
def move_sentence(df: DataFrame, clusters: dict, sent_id, new_cluster):
    # keep the old_cluster
    old_cluster = df[df['id'] == sent_id]['cluster'].iat[0]
    exists = old_cluster is not None

    # move to the new cluster
    df.loc[df['id'] == sent_id, 'cluster'] = new_cluster
    if new_cluster is not None:
        cluster_data = clusters[new_cluster]
        cluster_data['sentences'].add(sent_id)
        cluster_data['size'] += 1

    # remove from old cluster
    if old_cluster is not None:
        cluster_data = clusters[old_cluster]
        cluster_data['sentences'].remove(sent_id)
        cluster_data['size'] -= 1
        # if cluster is empty, delete it
        if cluster_data['size'] == 0:
            exists = False
            del cluster_data

    return old_cluster, exists


# match the sentences to the clusters
def iterate_over_data(df: DataFrame, clusters: dict, threshold: float, mid_calc: bool,
                      create_centroids: bool = True) -> bool:
    # initialize the convergence flag
    converged = True
    # check if the dataframe has a cluster column
    no_cluster = 'cluster' not in df
    if no_cluster:
        df['cluster'] = None

    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # find the closest centroid to the encoded sentence
        encoded = row['encoded']
        cluster = find_closest_cluster(clusters, encoded, threshold)

        if cluster is None and create_centroids:
            converged = False

            new_cluster = df['cluster'].max() + 1
            if np.isnan(new_cluster):
                new_cluster = 0
            clusters[new_cluster] = {'centroid': encoded,
                                     'size': 0,
                                     'sentences': set(),
                                     'average_distance': 0.0}
            cluster = new_cluster

        if cluster != row['cluster']:
            converged = False

            old_cluster, exists = move_sentence(df, clusters, row['id'], cluster)
            if mid_calc:
                calculate_centroids(df, clusters, {old_cluster, cluster} if exists else {cluster})

    return converged


# calculate the centroids of the clusters for specific keys
def calculate_centroids(df: DataFrame, clusters: dict, keys: set | None = None):
    for key in (keys if keys else clusters.keys()):
        clusters[key]['centroid'] = np.mean(df[df['cluster'] == key]['encoded'])
        clusters[key]['average_distance'] = np.mean(df.loc[df['cluster'] == key, 'encoded'].map(
            lambda x: np.sqrt(np.sum((x - clusters[key]['centroid']) ** 2))))


# save the clustering results to a json file
def save_clusters(df: DataFrame, clusters: dict, output_file: str):
    cluster_list = []
    unclustered = []
    # for each cluster that passed the threshold add it to the final cluster list
    for key, value in clusters.items():
        cluster_list.append({
            "cluster_name": value['title'],
            "requests": df[df['id'].isin(value['sentences'])]['text'].to_list()
        })
    # all the clusters that didn't pass should be None, add them to the unclustered category
    unclustered.extend(df[df['cluster'].isna()]['text'].to_list())

    clusters = {"cluster_list": cluster_list, "unclustered": unclustered}

    with open(output_file, 'w+', encoding='utf8') as fout:
        json.dump(clusters, fout, indent=4)


# give a title to each cluster
def title_clusters(df: DataFrame, clusters: dict, min_size: int):
    for key, value in clusters.items():
        if value['size'] >= min_size:
            # for each cluster that passed the threshold get all sentences
            sentence_list = df[df['id'].isin(value['sentences'])]['text'].to_list()
            # analyze the concatenated sentences
            doc = nlp(' '.join('. '.join(sentence_list).split()))

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
            value['title'] = max_str.strip('.')
            df.loc[df['cluster'] == key, 'title'] = value['title']

    df.loc[df['cluster'].isna(), 'title'] = 'unclustered'


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


# do a single iteration of the clustering algorithm
def iterate_once(df: DataFrame, clusters: dict, threshold: float, mid_calc: bool = False,
                 create_centroids: bool = True) -> bool:
    # iterate over the data points and check if we reached convergence
    if iterate_over_data(df, clusters, threshold, mid_calc, create_centroids):
        return True

    # calculate the centroids
    calculate_centroids(df, clusters)

    return False


# goes over the entire data set once
def do_epoch(df: DataFrame,
             threshold: float,
             min_size: int,
             cycles: int = 2,
             iterations_base: int = 0,
             iterations_mult: int = 10) -> dict:
    # initialize the clusters
    clusters = {}
    for i in range(cycles):
        # the smaller the threshold the more erratic the changes will be (at least that's my hypothesis)
        # therefore it'll take longer to converge, so we'll iterate more times
        for j in range(iterations_base + round(iterations_mult * (1.0 / threshold))):
            # randomize the row order
            df = df.sample(frac=1, random_state=42)
            df.reset_index(drop=True, inplace=True)

            # do a single iteration of the clustering algorithm
            if iterate_once(df, clusters, threshold):
                print("Converged after", j + 1, "iterations!")
                break

            print("Iteration", j + 1, "done")

        # filter the clusters
        filter_clusters(df, clusters, min_size)

        # iterate last time to match abandoned sentences to existing clusters
        iterate_once(df, clusters, threshold, create_centroids=False)

        print("Hyper iteration", i + 1, "done")

    # cannibalize the smaller clusters into the larger ones if possible
    cannibalize_clusters(df, clusters, threshold, int(min_size))

    return clusters


# delete the centroids that are too small
def filter_clusters(df: DataFrame, clusters: dict, min_size: int):
    # remove the clusters that are too small
    keys_to_remove = set()
    for key, value in clusters.items():
        if value['size'] < min_size:
            df.loc[df['cluster'] == key, 'cluster'] = None
            keys_to_remove.add(key)

    for key in keys_to_remove:
        del clusters[key]


# cannibalize the smaller clusters into the larger ones
def cannibalize_clusters(df: DataFrame, clusters: dict, threshold: float, min_size: int):
    # initialize the set of keys to delete
    keys_to_delete = set()
    for key, value in sorted(clusters.items(), key=lambda item: item[1]['size']):
        # if the cluster isn't too small, skip it
        if clusters[key]['size'] > min_size * 2:
            continue

        # remap the sentences to the new clusters
        remapping = re_cluster_sentences(df, clusters, threshold, key, keys_to_delete)
        if (len(remapping) / clusters[key]['size']) > threshold:
            # if the remapping is successful enough, mark the old cluster for deletion
            keys_to_delete.add(key)
            # apply the remapping to the sentences in the cluster
            for index, row in df[df['cluster'] == key].iterrows():
                move_sentence(df, clusters, row['id'], remapping[index])

    # delete the clusters that were emptied
    for key in keys_to_delete:
        del clusters[key]


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # load the model
    model = create_sentence_transformer_model()
    # create dataframe of the data
    df = read_lines(data_file)

    # encode the sentences to the dataframe
    encode_sentences(df, model)

    # create visualization base
    dims = 3
    reduce_dimensions(df, dims)

    # go through the dataframe and match the sentences to the clusters
    threshold = 0.7
    clusters = do_epoch(df, threshold, int(min_size))

    # give each cluster a title based on the sentences in the cluster
    title_clusters(df, clusters, int(min_size))

    # save the clustering results to a json file
    save_clusters(df=df, clusters=clusters, output_file=output_file)

    # plot the results
    plot_results(df, color='title')

    # todo: remove return statement, meant for testing in jupyter notebook
    return df, clusters


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
