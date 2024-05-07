import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DatasetConf, TrainConf
from utils import build_bow, features_bow, train_svm_classifier, to_numpy


# 提取 SIFT 特征
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        # 将图像转换为灰度
        gray_image = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        _, descriptors = sift.detectAndCompute(gray_image, None)
        # 若未检测到特征则添加零向量
        if descriptors is None:
            descriptors = np.zeros((1, 128))
        descriptors_list.append(descriptors.mean(axis=0))
    return np.array(descriptors_list)

# 绘制图像与特征
def plot_images(images, titles, row_titles, cmap=None):
    n = len(images)
    _, axs = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axs[0, i].imshow(images[i], cmap=cmap)
        axs[0, i].set_title(titles[i])
        axs[0, i].axis('off')
        axs[1, i].imshow(row_titles[i], cmap='hot')
        axs[1, i].axis('off')
    plt.show()

# 主流程
# 下载训练集和测试集
trainset = DatasetConf.TrainDataset
testset = DatasetConf.TestDataset
X_train, y_train = to_numpy(trainset)
X_test, y_test = to_numpy(testset)

# 提取特征
X_train_sift = extract_sift_features(X_train)
X_test_sift = extract_sift_features(X_test)

# 训练分类器
clf = train_svm_classifier(X_train_sift, y_train)

# 预测
predictions = clf.predict(X_test_sift)

# 评估模型
print(classification_report(y_test, predictions))

# 准备绘图
original_images = [X_test[i] for i in range(5)]
sift_images = [cv2.drawKeypoints(cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY),
                                 cv2.SIFT_create().detect(cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY), None),
                                 None) for img in original_images]
plot_images(original_images, predictions, sift_images)
