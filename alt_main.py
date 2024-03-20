import json

import numpy as np
import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from compare_clustering_solutions import evaluate_clustering


# clean the query text for processing
def clean_query(query: str) -> str:
    return query.lower().strip('\r\n')


# create a line generator from filename
def read_lines(filename: str) -> DataFrame:
    df = pd.read_csv(filename, index_col='id')
    df['text'] = df['text'].map(clean_query)
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
def find_closest_neighbor(df: DataFrame):
    distance_columns = [col for col in df.columns if '_distance' in col]
    df['closest_neighbor_distance'] = df.loc[:, df.columns.isin(distance_columns)].min(axis=1)


# find the radius which encompasses the n closest neighbors
def find_radius(df: DataFrame, n: int):
    df['radius'] = pd.NA
    df_dist = df[[col for col in df.columns if '_distance' in col]]
    for i in range(len(df)):
        index = df.index[i]
        df.at[index, 'radius'] = np.partition(df_dist[df_dist.index == index].to_numpy().flatten(), n)[n - 1]


# calculate centroids for the modified clusters
def calculate_centroids(df: DataFrame, clusters: dict):
    keys = (key for key, value in clusters.items() if value['modified'])
    for key in keys:
        clusters[key]['centroid'] = np.mean(df[df.index.isin(clusters[key]['requests'])]['encoded'], axis=0)
        clusters[key]['modified'] = False


# iterate over the data and cluster the requests
def cluster_requests(df: DataFrame, threshold: float) -> tuple[DataFrame, dict]:
    # shuffle the data
    df = df.sample(frac=1)

    # create a dictionary to store the clusters
    clusters = dict()
    max_key = 0

    # iterate over the data
    for i in range(len(df)):
        # calculate the centroids every 200 requests
        if i % 200 == 0:
            calculate_centroids(df, clusters)

        # get the row
        encoded = df.at[df.index[i], 'encoded']

        # find the most similar cluster to the sample
        min_distance = float('inf')
        min_cluster_id = -1
        for key, value in clusters.items():
            similarity = np.linalg.norm(encoded - value['centroid'])
            if similarity < min_distance:
                min_distance = similarity
                min_cluster_id = key

        # if the similarity is less than the threshold, add the request to the cluster
        if min_distance < threshold:
            clusters[min_cluster_id]['requests'].add(df.index[i])
            clusters[min_cluster_id]['modified'] = True
        # else, create a new cluster
        else:
            clusters[max_key] = {
                'centroid': encoded,
                'requests': {df.index[i]},
                'modified': False
            }
            max_key += 1

    return df, clusters


# transform the clusters into the required format
def transform_clusters_dict(df: DataFrame, clusters: dict, min_size: int) -> dict:
    cluster_list = []
    unclustered = []
    for key, value in clusters.items():
        # if the cluster size is greater than the threshold, add it to the result
        if len(value['requests']) > min_size:
            cluster_list.append({
                'cluster_name': f'cluster_{key}',
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


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # read the data
    df = read_lines(data_file)

    # create a sentence transformer model
    model = create_sentence_transformer_model()

    # encode the sentences
    encode_sentences(df, model)

    # cluster the requests
    df, clusters = cluster_requests(df, 0.756)

    # transform the clusters into the required format
    result = transform_clusters_dict(df, clusters, int(min_size))

    # output the result to a json file
    output_result(result, output_file)

    return result


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
