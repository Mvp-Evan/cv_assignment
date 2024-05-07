from skimage import feature, color
from sklearn import svm, metrics
import numpy as np


import torch
import torchvision
import torchvision.transforms as transforms

from config import DatasetConf
from utils import train_svm_classifier

# 下载训练集和测试集
trainset = DatasetConf.TrainDataset
testset = DatasetConf.TestDataset


# 转换为NumPy数组的辅助函数
def to_numpy(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(data_loader))
    images, labels = data
    # 转换为NumPy数组并调整图像格式为HWC（高度，宽度，通道）
    return images.numpy().transpose((0, 2, 3, 1)), labels.numpy()


# 获取NumPy格式的数据
X_train, y_train = to_numpy(trainset)
X_test, y_test = to_numpy(testset)


# 特征提取
def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray_image = color.rgb2gray(image)  # 转换为灰度图
        hog_feature = feature.hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
        hog_features.append(hog_feature)
    return np.array(hog_features)


# 训练SVM
print('Extracting HOG features...')
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)
print('Training SVM...')
clf = train_svm_classifier(X_train_hog, y_train)

# 进行预测
predictions = clf.predict(X_test_hog)

# 评估模型
print(metrics.classification_report(y_test, predictions))

# 展示hog特征图

import matplotlib.pyplot as plt

# 选择多个图像
fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 创建一个2行5列的subplot网格

for i in range(5):
    image = X_train[i]
    gray_image = color.rgb2gray(image)
    # 提取HOG特征
    hog_feature, hog_image = feature.hog(gray_image, visualize=True)

    # 显示原始图像
    axs[0, i].imshow(image)
    axs[0, i].set_title(f'Original Image {i + 1}')
    axs[0, i].axis('off')

    # 显示HOG特征图
    axs[1, i].imshow(hog_image, cmap='gray')
    axs[1, i].set_title(f'HOG Feature {i + 1}')
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()
