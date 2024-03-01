import json

# noinspection PyUnresolvedReferences
import benepar, spacy
import en_core_web_md
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
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


# create a line generator from filename
def read_lines(filename: str) -> DataFrame:
    return pd.read_csv(filename)


# create a sentence transformer model
def create_sentence_transformer_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')


# encode sentences using the sentence transformer model
def encode_sentences(df: DataFrame, model: SentenceTransformer):
    df['encoded'] = df['text'].map(lambda x: model.encode(x))


# find the closest centroid to an encoded sentence
def find_closest_cluster(clusters: dict, sentence: Tensor, threshold: float):
    min_distance = float('inf')
    closest_cluster = None
    for key, value in clusters.items():
        distance = np.sqrt(np.sum((value['centroid'] - sentence) ** 2))
        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_cluster = key
    return closest_cluster


# move a sentence from one centroid to another
def move_sentence(df: DataFrame, clusters: dict, sentence_index: int, new_cluster):
    # keep the old_cluster
    old_cluster = df.at[sentence_index, 'centroid']

    # move to the new cluster
    df.loc[sentence_index, 'cluster'] = new_cluster
    if new_cluster is not None:
        cluster_data = clusters[new_cluster]
        cluster_data['sentences'].add(df.at[sentence_index, 'id'])
        cluster_data['size'] += 1

    # remove from old cluster
    if old_cluster is not None:
        cluster_data = clusters[old_cluster]
        cluster_data['sentences'].remove(df.at[sentence_index, 'id'])
        cluster_data['size'] -= 1
        # if cluster is empty, delete it
        if cluster_data['size'] == 0:
            del cluster_data


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

        if no_cluster:
            converged = False
        elif cluster != (row_cent := row['cluster']):
            converged = False
            if row_cent:
                try:
                    clusters[row_cent]['sentences'].remove(row['id'])
                    clusters[row_cent]['size'] -= 1
                    if row_cent == row['id']:
                        if cluster[row_cent]['size'] == 0:
                            del clusters[row_cent]
                        else:
                            clusters[temp := cluster[row_cent]['sentences'][0]] = clusters[row_cent]
                            del clusters[row_cent]
                            row_cent = temp

                    if mid_calc:
                        calculate_centroids(df, clusters, {row_cent})
                except KeyError:
                    print(f"KeyError: {row['id']} not in {row_cent}")
                    print(f"{clusters[row_cent]['sentences']}")

        if cluster:
            clusters[cluster]['sentences'].add(row['id'])
            clusters[cluster]['size'] += 1
            df.loc[index, 'cluster'] = cluster
            if mid_calc:
                calculate_centroids(df, clusters, {cluster})
        elif create_centroids:
            clusters[row['id']] = {'centroid': encoded,
                                   'size': 1,
                                   'sentences': {row['id']}}
            df.loc[df['id'] == row['id'], 'cluster'] = row['id']

    return converged


# calculate the centroids of the clusters for specific keys
def calculate_centroids(df: DataFrame, clusters: dict, keys: set | None = None):
    for key in (keys if keys else clusters.keys()):
        clusters[key]['centroid'] = np.mean(df[df['cluster'] == key]['encoded'].to_numpy())


# save the clustering results to a json file
def save_clusters(df: DataFrame, clusters: dict, min_size: int, output_file: str):
    cluster_list = []
    unclustered = []
    for key, value in clusters.items():
        if value['size'] >= min_size:
            cluster_list.append({
                "cluster_name": value['title'],
                "requests": df[df['id'].isin(value['sentences'])]['text'].to_list()
            })
        else:
            unclustered.extend(df[df['id'].isin(value['sentences'])]['text'].to_list())

    clusters = {"cluster_list": cluster_list, "unclustered": unclustered}

    with open(output_file, 'w+', encoding='utf8') as fout:
        json.dump(clusters, fout, indent=4)


# give a title to each cluster
def title_clusters(df: DataFrame, clusters: dict, min_size: int):
    for key, value in clusters.items():
        if value['size'] >= min_size:
            sentence_list = df[df['id'].isin(value['sentences'])]['text'].to_list()
            doc = nlp(' '.join('. '.join(sentence_list).split()))

            max_sim = float('-inf')
            max_str = ""
            for sent in doc.sents:
                for constituent in sent._.constituents:
                    if 'S' in constituent._.labels:
                        if (similarity := doc.similarity(constituent)) > max_sim:
                            max_sim = similarity
                            max_str = constituent.text

            value['title'] = max_str


# reduces the dimensionality of the data to k dimensions
def reduce_dimensions(df: DataFrame, dims: int):
    pca = PCA(n_components=dims)
    reduced = pca.fit_transform(df['encoded'].to_list())
    for i in range(dims):
        df[f'reduced{str(i)}'] = reduced[:, i]


# plots a cluster on a figure
def plot_cluster(ax: Axes, df: DataFrame, clusters: dict, key: str | None) -> Axes:
    if key:
        filtered_df = df[df['centroid'] == key]
    elif len(clusters) > 0:
        filtered_df = df[df['centroid'].isna()]
    else:
        filtered_df = df

    ax.scatter(filtered_df['reduced0'].to_numpy(),
               filtered_df['reduced1'].to_numpy(),
               filtered_df['reduced2'].to_numpy(),
               label=key if key else 'unclustered')
    return ax


# plot results
def plot_results(df: DataFrame, clusters: dict, subplot: bool = False) -> Figure:
    plt.clf()

    if subplot:
        fig = plt.figure()
        for key in list(clusters.keys()) + [None]:
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('dim0')
            ax.set_ylabel('dim1')
            ax.set_zlabel('dim2')
            plot_cluster(ax, df, clusters, key)
            ax.set_title(key if key else 'unclustered')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('dim0')
        ax.set_ylabel('dim1')
        ax.set_zlabel('dim2')
        for key in list(clusters.keys()) + [None]:
            plot_cluster(ax, df, clusters, key)
    plt.show()
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


# delete the centroids that are too small
def filter_clusters(df: DataFrame, clusters: dict, min_size: int):
    # remove the clusters that are too small
    keys_to_remove = set()
    for key, value in clusters.items():
        if value['size'] < min_size:
            df.loc[df['centroid'] == key, 'centroid'] = None
            keys_to_remove.add(key)

    for key in keys_to_remove:
        del clusters[key]


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # load the model
    model = create_sentence_transformer_model()
    # create dataframe of the data
    df = read_lines(data_file)
    # encode the sentences to the dataframe
    encode_sentences(df, model)
    # create a dictionary of clusters
    clusters = {}

    # create visualization base
    dims = 3
    reduce_dimensions(df, dims)

    # go through the dataframe and match the sentences to the clusters
    threshold = 0.91
    for i in range(max_iterations := 15):
        # randomize the row order
        # df = df.sample(frac=1, random_state=42)

        # plot the results
        plot_results(df, clusters)

        # do a single iteration of the clustering algorithm
        if iterate_once(df, clusters, threshold):
            print("Converged after", i + 1, "iterations!")
            break

        # filter the clusters
        filter_clusters(df, clusters, int(min_size))

        # iterate last time to match abandoned sentences to existing clusters
        iterate_once(df, clusters, threshold, create_centroids=False)

        print("Iteration", i + 1, "done")

    # give each cluster a title based on the sentences in the cluster
    title_clusters(df, clusters, int(min_size))

    # save the clustering results to a json file
    save_clusters(df=df, clusters=clusters, min_size=int(min_size), output_file=output_file)

    # todo: implement this function
    #  you are encouraged to break the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file

    pass


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
