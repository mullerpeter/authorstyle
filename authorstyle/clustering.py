import numpy as np
import random

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.neighbors import DistanceMetric

random.seed(54321)


def convert_truth(problem):
    """
    Converts the truth from breaches to cluster labeling
    :type problem: Problem
    :param problem: Problem to be clustered
    :rtype  int []
    :returns True Clustering
    """
    cluster_number = 0
    borders = problem.truth['switches']
    structure = problem.truth['structure']

    mapping = {'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4}
    truth = []
    full_text = problem.text.text
    for fragment in problem.text.fragments:
        if len(structure) > 0:
            cluster_number = mapping[structure[0]]

        position = full_text.find(fragment.sentences[0])
        if position == -1 and len(fragment.sentences) > 1:
            position = full_text.find(fragment.sentences[1])
        if len(borders) > 0 and (position > borders[0]):
            del borders[0]
            del structure[0]

        truth.append(cluster_number)
    return truth


# noinspection PyPep8Naming
def ordinal_classification_index(confussion_matrix):
    """
    Returns the Ordinal Classification Index of a given confusion Matrix
    proposed by Cardoso and Recardo (2011) (Implementation by PAN19 comitee)
    :type confussion_matrix: int [][]
    :param confussion_matrix: Confusion Matrix
    :rtype  float
    :returns Ordinal Classification Index
    """
    K = np.shape(confussion_matrix)[0]
    N = np.sum(confussion_matrix)
    ggamma = 1
    bbeta = 0.75 / np.power(N * (K - 1), ggamma)

    helperM2 = np.zeros_like(confussion_matrix)

    for r in range(0, K):
        for c in range(0, K):
            helperM2[r][c] = confussion_matrix[r][c] * np.power((abs(r - c)), ggamma)

    TotalDispersion = (np.power(np.sum(helperM2), (1 / ggamma)))
    helperM1 = confussion_matrix / (TotalDispersion + N)

    errMatrix = np.zeros_like(confussion_matrix, dtype=np.float)
    errMatrix[0][0] = 1 - helperM1[0][0] + bbeta * helperM2[0][0]

    for r in range(1, K):
        c = 0
        errMatrix[r][c] = errMatrix[r - 1][c] - helperM1[r][c] + bbeta * helperM2[r][c]

    for c in range(1, K):
        r = 0
        errMatrix[r][c] = errMatrix[r][c - 1] - helperM1[r][c] + bbeta * helperM2[r][c]

    for c in range(1, K):
        for r in range(1, K):
            costup = errMatrix[r - 1, c]
            costleft = errMatrix[r, c - 1]
            lefttopcost = errMatrix[r - 1, c - 1]
            aux = np.min([costup, costleft, lefttopcost])
            errMatrix[r][c] = aux - helperM1[r][c] + bbeta * helperM2[r][c]

    return errMatrix[-1][-1]


def connectivity_matrix(size):
    """
    Returns a simple connectivity matrix, connecting each point n to n-1 & n+1
    :type size: int
    :param size: Size of the matrix
    :rtype  float [] []
    :returns Connectivity Matrix
    """
    cm = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            if np.absolute(x - y) < 2:
                cm[x][y] = 1.0

    return cm


def score_clustering(labels_true, labels_pred):
    """
    Returns some default Clustering Scores
    :rtype 6 * float
    :returns Homogeneity Score, Completeness Score, V-measure, Adjusted Random Index, Adjusted Mutual Information,
    Fowlkes-Mallows Score
    """

    if len(labels_true) != len(labels_pred):
        return None

    homogeneity_score = metrics.homogeneity_score(labels_true, labels_pred)
    completeness_score = metrics.completeness_score(labels_true, labels_pred)
    v_measure_score = metrics.v_measure_score(labels_true, labels_pred)

    adjusted_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels_true, labels_pred)

    return homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, \
        fowlkes_mallows_score


def agglomerative_clustering(problem, linkage='average', max_clusters=8, connectivity=None, feature_mask=None):
    """
    Generates a clustering using Agglomerative/Hierarchical Clustering
    :type problem: Problem
    :param problem: Problem to be clustered
    :type linkage: str
    :param linkage: Which linkage criterion to use (ward, complete, average, single)
    :type max_clusters: int
    :param max_clusters: Max Number of clusters
    :type connectivity: []
    :param connectivity: Connectivity Matrix (optional)
    :type feature_mask: []
    :param feature_mask: List of feature indexes to use for clustering
    :rtype int, int []
    :returns Number of Clusters, Cluster Assignment
    """

    if len(problem.feature_vector) < 2:
        return 1, [0]

    feature_vector = problem.feature_vector

    if feature_mask is not None:
        feature_vector = feature_vector[:, feature_mask]

    max_clusters = min(len(feature_vector) - 1, max_clusters)

    labels = []
    sil_coeff = []

    for n in range(2, max_clusters + 1):
        ac = AgglomerativeClustering(linkage=linkage, connectivity=connectivity, n_clusters=n)
        ac.fit(feature_vector)
        labels.append(ac.labels_)
        sil_coeff.append(silhouette_score(feature_vector, ac.labels_, metric='euclidean'))

    if len(sil_coeff) > 0:
        best_n = sil_coeff.index(max(sil_coeff))
        return best_n + 2, labels[best_n]
    else:
        return 1, [0]


def dbscan_clustering(problem, eps=0.5, min_samples=5):
    """
    Generates a clustering using DBSCAN
    :type problem: Problem
    :param problem: Problem to be clustered
    :type eps: float
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood
    :type min_samples: int
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a
    core point. This includes the point itself.
    :rtype int, int []
    :returns Number of Clusters, Cluster Assignment
    """

    feature_vector = problem.feature_vector

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_vector)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return n_clusters_, labels


def kmeans_clustering(problem, max_clusters=8, feature_mask=None):
    """
    Generates a clustering using KMeans
    :type problem: Problem
    :param problem: Problem to be clustered
    :type max_clusters: int
    :param max_clusters: Max Number of clusters
    :type feature_mask: []
    :param feature_mask: List of feature indexes to use for clustering
    :rtype int, int []
    :returns Number of Clusters, Cluster Assignment
    """

    feature_vector = problem.feature_vector

    if feature_mask is not None:
        feature_vector = feature_vector[:, feature_mask]

    if len(problem.feature_vector) < 2:
        return 1, [0]

    labels = []
    sil_coeff = []

    max_clusters = min(len(feature_vector) - 1, max_clusters)

    kmeans = KMeans(n_clusters=1, random_state=54321)
    kmeans.fit(feature_vector)

    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=54321)
        kmeans.fit(feature_vector)
        label = kmeans.labels_
        labels.append(label)
        sil_coeff.append(0.0 if len(list(set(label))) == 1 else silhouette_score(feature_vector, label))

    if len(sil_coeff) > 0:
        best_n = sil_coeff.index(max(sil_coeff))
        return best_n + 2, labels[best_n]
    else:
        return 1, [0]


def correlation_clustering(problem):
    """
    Generates a clustering using Correlation Clustering
    :type problem: Problem
    :param problem: Problem to be clustered
    :rtype int, int []
    :returns Number of Clusters, Cluster Assignment
    """

    feature_vector = problem.feature_vector

    dist_matrix = DistanceMetric.get_metric('euclidean').pairwise(feature_vector)
    mean_distance = np.mean(dist_matrix)
    edges = ((dist_matrix - mean_distance) * -1) * connectivity_matrix(len(feature_vector))

    clusters = []
    v = np.arange(len(feature_vector))

    while len(v) > 0:
        c = []
        i_index = random.choice(range(len(v)))
        i = v[i_index]
        c.append(i)
        v = np.delete(v, [i_index])

        for j_index in range(len(v)):
            j = v[j_index - len(c) + 1]
            if edges[i][j] > 0:
                c.append(j)
                v = np.delete(v, [j_index])

        clusters.append(c)

    labels = [0] * len(feature_vector)
    for cluster_index in range(len(clusters)):
        for point in clusters[cluster_index]:
            labels[point] = cluster_index

    return len(clusters), labels
