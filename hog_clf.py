import torch
from skimage import feature
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# 设置数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# 提取整个数据集
train_features, train_labels = next(iter(train_loader))
test_features, test_labels = next(iter(test_loader))

# 将数据转换为numpy格式用于图像处理
train_images = train_features.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
test_images = test_features.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

# 函数：提取Harris角点特征
def extract_features(images):
    features = []
    for image in images:
        gray_image = rgb2gray(image)
        hog_feature = feature.hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
        features.append(hog_feature)
    return np.array(features)

# 提取特征
train_features = extract_features(train_images)
test_features = extract_features(test_images)

# 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(train_features, train_labels.numpy().ravel())

# 进行预测
predictions = clf.predict(test_features)

# 评估模型
print(classification_report(test_labels.numpy(), predictions))

