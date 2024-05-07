import numpy as np
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_bow(descriptors_list, num_clusters=200):
    bow_kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    for desc in descriptors_list:
        if desc is not None:
            bow_kmeans.partial_fit(desc)
    return bow_kmeans


def features_bow(descriptors_list, bow_kmeans):
    bow_features = np.zeros((len(descriptors_list), bow_kmeans.n_clusters), dtype=np.float32)
    for i, desc in enumerate(descriptors_list):
        if desc is not None:
            words = bow_kmeans.predict(desc)
            bow_features[i, words] += 1
    return bow_features


def train_svm_classifier(features, labels):
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
    clf.fit(features, labels)
    return clf