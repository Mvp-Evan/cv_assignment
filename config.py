import torch
import torchvision
from torchvision import transforms


class DatasetConf:
    Type = ['CIFAR-10', 'STL-10'][1]
    Root = './data'
    AllInGrey = False
    Transforms = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class TrainConf:
    Epochs = 50
    BatchSize = 512
    LearningRate = 0.001
    Device = torch.device("mps")


class CNNConf:
    UseAutoEncoder = True
