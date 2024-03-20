import json
import string

import benepar
import en_core_web_md
import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly.graph_objs import Figure
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from spacy import Language
import plotly.express as px

from compare_clustering_solutions import evaluate_clustering


# clean the query text for processing
def clean_query(query: str) -> str:
    return query.lower().strip('\r\n')


# create a line generator from filename
def read_lines(filename: str) -> DataFrame:
    df = pd.read_csv(filename, index_col='id')
    df['text'] = df['text'].map(clean_query)
    df['cluster'] = -1
    return df


# create a sentence transformer model
def create_sentence_transformer_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')


# encode sentences using the sentence transformer model
def encode_sentences(df: DataFrame, model: SentenceTransformer):
    df['encoded'] = df['text'].map(lambda x: model.encode(x))


# calculate the distance between every 2 encodings
def calculate_distances(df: DataFrame):
    for i in range(len(df)):
        index = df.index[i]
        col = f"{index}_distance"
        encoded = df.at[index, 'encoded']
        df[col] = df['encoded'].map(lambda x: np.linalg.norm(x - encoded))
        df.at[index, col] = pd.NA


# find the closest neighbors for each request
def find_closest_neighbor_distance(df: DataFrame):
    distance_columns = [col for col in df.columns if '_distance' in col]
    df['closest_neighbor_distance'] = df.loc[:, df.columns.isin(distance_columns)].min(axis=1)


# count closest neighbors for each request
def count_closest_neighbors(df: DataFrame, threshold: float):
    df['closest_neighbors'] = 0
    # iterate over the data
    for i in range(len(df)):
        i_idx = df.index[i]
        # find the closest neighbor distance
        min_distance = df.at[i_idx, 'closest_neighbor_distance']
        # if the distance is greater than the threshold, skip the request
        if min_distance > threshold:
            continue
        # iterate over the data
        for j in range(len(df)):
            if i == j:
                continue
            # check if the distance is the closest
            j_idx = df.index[j]
            if min_distance == df.at[j_idx, f"{i_idx}_distance"]:
                # if it is, increment the closest neighbors count
                df.at[j_idx, 'closest_neighbors'] += 1


# calculate centroids for the modified clusters
def calculate_centroids(df: DataFrame, clusters: dict):
    keys = (key for key, value in clusters.items() if value['modified'])
    for key in keys:
        clusters[key]['centroid'] = np.mean(df[df.index.isin(clusters[key]['requests'])]['encoded'], axis=0)
        clusters[key]['modified'] = False


# iterate over the data and cluster the requests
def cluster_requests(df: DataFrame, clusters: dict, threshold: float, sort_by_count: bool = False) -> int:
    num_changes = 0

    # shuffle the data
    if sort_by_count:
        order = df.sort_values(by='closest_neighbors', ascending=False).index
    else:
        order = df.sample(frac=1).index

    # create a dictionary to store the clusters
    max_key = max(clusters.keys(), default=-1) + 1

    # iterate over the data
    for i in range(len(df)):
        # calculate the centroids every 200 requests
        if i % 200 == 0:
            calculate_centroids(df, clusters)

        # get the row
        encoded = df.at[order[i], 'encoded']
        cluster = df.at[order[i], 'cluster']

        # find the closest cluster to the sample
        min_distance = float('inf')
        min_cluster_id = -1
        for key, value in clusters.items():
            distance = np.linalg.norm(encoded - value['centroid'])
            if distance < min_distance:
                min_distance = distance
                min_cluster_id = key

        # if the distance is within the threshold, add the request to the cluster
        if min_distance < threshold:
            df.at[order[i], 'cluster'] = min_cluster_id
            clusters[min_cluster_id]['requests'].add(order[i])
            clusters[min_cluster_id]['modified'] = True

            if cluster != min_cluster_id:
                num_changes += 1
                # remove the request from the previous cluster
                if cluster != -1:
                    clusters[cluster]['requests'].remove(order[i])
                    clusters[cluster]['modified'] = True

        # else, create a new cluster
        else:
            num_changes += 1
            df.at[order[i], 'cluster'] = max_key
            clusters[max_key] = {
                'centroid': encoded,
                'requests': {order[i]},
                'modified': False
            }
            max_key += 1
            # remove the request from the previous cluster
            if cluster != -1:
                clusters[cluster]['requests'].remove(order[i])
                clusters[cluster]['modified'] = True

    return num_changes


# give a title to a cluster
def title_clusters(df: DataFrame, clusters: dict, nlp: Language, min_size: int):
    df['title'] = 'cluster_' + df['cluster'].astype(str)
    for cluster in clusters.keys():
        # if the cluster size is below the threshold, set the title to unclustered
        if len(clusters[cluster]['requests']) < min_size:
            df.loc[df['cluster'] == cluster, 'title'] = 'unclustered'
            df.loc[df['cluster'] == cluster, 'cluster'] = -1
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
                if 2 < len(constituent) < 8:
                    if (similarity := doc.similarity(constituent)) > max_sim:
                        max_sim = similarity
                        max_str = constituent.text

        # set the most similar as the title
        title = max_str.strip('.')
        if title != '':
            df.loc[df['cluster'] == cluster, 'title'] = title
            clusters[cluster]['title'] = title


# transform the clusters into the required format
def transform_clusters_dict(df: DataFrame, clusters: dict, min_size: int) -> dict:
    cluster_list = []
    unclustered = []
    for key, value in clusters.items():
        # if the cluster size is greater than the threshold, add it to the result
        if len(value['requests']) > min_size:
            cluster_list.append({
                'cluster_name': value['title'] if 'title' in value else f'Cluster {key}',
                'requests': df[df.index.isin(value['requests'])]['text'].tolist()
            })
        # else, add the requests to the unclustered list
        else:
            unclustered.extend(df[df.index.isin(value['requests'])]['text'].tolist())

    return {'cluster_list': cluster_list, 'unclustered': unclustered}


# output the result to a json file
def output_result(result: dict, filename: str):
    with open(filename, 'w+', encoding='utf8') as fout:
        json.dump(result, fout, indent=4)


# remove clusters below the minimum size from the dictionary
def destroy_below_min_size(df: DataFrame, clusters: dict, min_size: int) -> tuple[int, int]:
    to_destroy = []
    num_requests = 0
    # search for clusters below the minimum size
    for key, value in clusters.items():
        if len(value['requests']) < min_size:
            # remove the requests from the cluster
            for request in value['requests']:
                df.at[request, 'cluster'] = -1
            to_destroy.append(key)
            num_requests += len(value['requests'])

    # remove the clusters from the dictionary
    for key in to_destroy:
        del clusters[key]

    return len(to_destroy), num_requests


# prepare the spacy pipeline for the naming part of the task
def prepare_spacy_pipeline() -> Language:
    # load the spaCy model
    benepar.download('benepar_en3')
    nlp = en_core_web_md.load()
    # load the benepar parser
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    return nlp


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
    return fig


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # read the data
    df = read_lines(data_file)

    # create a sentence transformer model
    model = create_sentence_transformer_model()

    # encode the sentences
    encode_sentences(df, model)

    # calculate the distances
    calculate_distances(df)

    # find the closest neighbors for each request
    find_closest_neighbor_distance(df)

    # count closest neighbors for each request
    count_closest_neighbors(df, 0.756)

    # initialize the clusters dictionary
    clusters = dict()
    max_iterations = 100
    # initial values to prevent errors
    num_changes = 1
    iteration = None
    destroy_below_threshold = True
    # cluster the requests
    for iteration in range(max_iterations):
        # cluster the requests
        num_changes = cluster_requests(df=df,
                                       clusters=clusters,
                                       threshold=0.756,
                                       sort_by_count=iteration % 5 == 0)
        print(f"Number of changes: {num_changes}")

        # remove clusters below the minimum size the first time the number of changes is 0
        if num_changes == 0 and destroy_below_threshold and iteration < max_iterations - 1:
            destroy_below_threshold = False
            num_clusters, num_requests = destroy_below_min_size(df, clusters, int(min_size))
            print(f"Removed {num_clusters} clusters below the minimum size, with a total of {num_requests} requests.")
            num_changes = num_requests

        print(f"Iteration {iteration + 1} completed\n")

        # if the number of changes is 0, the algorithm has converged
        if num_changes == 0:
            break

    if num_changes == 0:
        print(f"Converged after {iteration + 1} iterations.\n")
    else:
        print(f"Did not converge after {max_iterations} iterations, maximum number of iterations reached.\n")

    # prepare the spacy pipeline
    nlp = prepare_spacy_pipeline()

    # title the clusters
    title_clusters(df, clusters, nlp, int(min_size))

    # transform the clusters into the required format
    result = transform_clusters_dict(df, clusters, int(min_size))

    # reduce the dimensions of the data for visualization
    reduce_dimensions(df, 3)

    # plot the results
    fig = plot_results(df, title='Clustering Results', axis=('reduced0', 'reduced1', 'reduced2'), color='title')
    fig.write_html(f'clustering_results.html')

    # output the result to a json file
    output_result(result, output_file)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
