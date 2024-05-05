from skimage import feature, color
from sklearn import svm, metrics
import numpy as np


import torch
import torchvision
import torchvision.transforms as transforms

from config import DatasetConf

# 设置transform来将数据转换为tensor
transform = DatasetConf.Transforms

# 下载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


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
clf = svm.SVC(kernel='linear')
clf.fit(X_train_hog, y_train)

# 评估
print('Evaluating...')
y_pred = clf.predict(X_test_hog)
print(metrics.classification_report(y_test, y_pred))
