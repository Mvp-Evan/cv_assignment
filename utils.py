import numpy as np
import torch
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import TrainConf


def build_bow(descriptors_list, num_clusters=TrainConf.NumClusters):
    bow_kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    for desc in descriptors_list:
        if desc is not None:
            desc = np.array(desc, dtype=np.float32)
            bow_kmeans.partial_fit(desc)
    return bow_kmeans


def features_bow(descriptors_list, bow_kmeans):
    bow_features = np.zeros((len(descriptors_list), bow_kmeans.n_clusters), dtype=np.float32)
    for i, desc in enumerate(descriptors_list):
        if desc is not None:
            desc = np.array(desc, dtype=np.float32)
            words = bow_kmeans.predict(desc)
            bow_features[i, words] += 1
    return bow_features


def train_svm_classifier(features, labels):
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
    clf.fit(features, labels)
    return clf


def to_numpy(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(data_loader))
    images, labels = data
    return images.numpy().transpose((0, 2, 3, 1)), labels.numpy()
