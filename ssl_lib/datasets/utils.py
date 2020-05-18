import os
import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision.datasets import SVHN, CIFAR10, CIFAR100, STL10


class InfiniteSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        epochs = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(epochs)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_svhn(root):
    train_data = SVHN(root, "train", download=True)
    test_data = SVHN(root, "test", download=True)
    train_data = {"images": np.transpose(train_data.data.astype(np.float32), (0, 2, 3, 1)),
                  "labels": train_data.labels.astype(np.int32)}
    test_data = {"images": np.transpose(test_data.data.astype(np.float32), (0, 2, 3, 1)),
                 "labels": test_data.labels.astype(np.int32)}
    return train_data, test_data


def get_cifar10(root):
    train_data = CIFAR10(root, download=True)
    test_data = CIFAR10(root, False)
    train_data = {"images": train_data.data.astype(np.float32),
                  "labels": np.asarray(train_data.targets).astype(np.int32)}
    test_data = {"images": test_data.data.astype(np.float32), 
                 "labels": np.asarray(test_data.targets).astype(np.int32)}
    return train_data, test_data


def get_cifar100(root):
    train_data = CIFAR100(root, download=True)
    test_data = CIFAR100(root, False)
    train_data = {"images": train_data.data.astype(np.float32),
                  "labels": np.asarray(train_data.targets).astype(np.int32)}
    test_data = {"images": test_data.data.astype(np.float32),
                 "labels": np.asarray(test_data.targets).astype(np.int32)}
    return train_data, test_data


def get_stl10(root):
    train_data = STL10(root, split="train", download=True)
    ul_train_data = STL10(root, split="unlabeled")
    test_data = STL10(root, split="test")
    train_data = {"images": np.transpose(train_data.data.astype(np.float32), (0, 2, 3, 1)),
                  "labels": train_data.labels}
    ul_train_data = {"images": np.transpose(ul_train_data.data.astype(np.float32), (0, 2, 3, 1)),
                    "labels": ul_train_data.labels}
    test_data = {"images": np.transpose(test_data.data.astype(np.float32), (0, 2, 3, 1)),
                 "labels": test_data.labels}
    return train_data, ul_train_data, test_data


def dataset_split(data, num_data, num_classes, random=False):
    """split dataset into two datasets
    
    Parameters
    -----
    data: dict with keys ["images", "labels"]
        each value is numpy.array
    num_data: int
        number of dataset1
    num_classes: int
        number of classes
    random: bool
        if True, dataset1 is randomly sampled from data.
        if False, dataset1 is uniformly sampled from data,
        which means that the dataset1 contains the same number of samples per class.

    Returns
    -----
    dataset1, dataset2: the same dict as data.
        number of data in dataset1 is num_data.
        number of data in dataset1 is len(data) - num_data.
    """
    dataset1 = {"images": [], "labels": []}
    dataset2 = {"images": [], "labels": []}
    images = data["images"]
    labels = data["labels"]

    # random sampling
    if random:
        dataset1["images"] = images[:num_data]
        dataset1["labels"] = labels[:num_data]
        dataset2["images"] = images[num_data:]
        dataset2["labels"] = labels[num_data:]

    else:
        data_per_class = num_data // num_classes
        for c in range(num_classes):
            c_idx = (labels == c)
            c_imgs = images[c_idx]
            c_lbls = labels[c_idx]
            dataset1["images"].append(c_imgs[:data_per_class])
            dataset1["labels"].append(c_lbls[:data_per_class])
            dataset2["images"].append(c_imgs[data_per_class:])
            dataset2["labels"].append(c_lbls[data_per_class:])
        for k in ("images", "labels"):
            dataset1[k] = np.concatenate(dataset1[k])
            dataset2[k] = np.concatenate(dataset2[k])

    return dataset1, dataset2


def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp
