import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import DatasetConf, TrainConf
from utils import to_numpy, build_bow, features_bow, train_svm_classifier


def extract_sift_features(X):
    sift = cv2.SIFT_create()
    descriptors_list = []

    for img in X:
        img = (img * 255).astype(np.uint8)  # 转换为8位图像
        kp, desc = sift.detectAndCompute(img, None)
        # 如果检测到的特征数小于TrainConf.NumClusters，则添加0向量并跳过
        if desc is None or len(kp) < TrainConf.NumClusters:
            desc = np.zeros((TrainConf.NumClusters, 128))

        descriptors_list.append(desc)

    return descriptors_list

# 下载训练集和测试集
trainset = DatasetConf.TrainDataset
testset = DatasetConf.TestDataset
X_train, y_train = to_numpy(trainset)
X_test, y_test = to_numpy(testset)

# 提取特征
X_train_sift = extract_sift_features(X_train)
X_test_sift = extract_sift_features(X_test)

# 构建并训练 BoW 模型
bow_kmeans = build_bow(X_train_sift)
X_train_bow = features_bow(X_train_sift, bow_kmeans)
X_test_bow = features_bow(X_test_sift, bow_kmeans)

# 数据标准化
scaler = StandardScaler().fit(X_train_bow)
X_train_bow = scaler.transform(X_train_bow)
X_test_bow = scaler.transform(X_test_bow)

# 训练 SVM
svm = train_svm_classifier(X_train_bow, y_train)

# 预测
predictions = svm.predict(X_test_bow)

# 评估模型
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
