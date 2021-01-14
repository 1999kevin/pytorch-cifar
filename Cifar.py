### cifar 10 ori data


import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names

"""show it works"""

cifar_10_dir = './data/cifar-10-batches-py'

train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)


### cifar 100 ori data
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

meta = unpickle('./data/cifar-100-python/meta')

fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

## cifar 100 selected train data
train = unpickle('./data/cifar-100-python/train')
filenames = [t.decode('utf8') for t in train[b'filenames']]
fine_labels = train[b'fine_labels']
data = train[b'data']
images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    images.append(image)
## flowers
flowers_label = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
flowers_idx = []
for i in range(len(fine_label_names)):
    if fine_label_names[i] in flowers_label:
        flowers_idx.append(i)
train_cifar100_data = []
for i in range(len(fine_labels)):
    if fine_labels[i] in flowers_idx:
        train_cifar100_data.append(images[i])

## tree
trees_label = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']
trees_idx = []
for i in range(len(fine_label_names)):
    if fine_label_names[i] in trees_label:
        trees_idx.append(i)
for i in range(len(fine_labels)):
    if fine_labels[i] in trees_idx:
        train_cifar100_data.append(images[i])
train_cifar100_label = []
for i in range(np.shape(train_cifar100_data)[0]):
    train_cifar100_label.append(10)

## train_cifar100_data + train_cifar100_label

## cifar 100 selected test data
test = unpickle('./data/cifar-100-python/test')
filenames = [t.decode('utf8') for t in test[b'filenames']]
fine_labels = test[b'fine_labels']
data = test[b'data']
images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    images.append(image)
    
## flowers
flowers_label = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
flowers_idx = []
for i in range(len(fine_label_names)):
    if fine_label_names[i] in flowers_label:
        flowers_idx.append(i)
test_cifar100_data = []
for i in range(len(fine_labels)):
    if fine_labels[i] in flowers_idx:
        test_cifar100_data.append(images[i])

## tree
trees_label = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']
trees_idx = []
for i in range(len(fine_label_names)):
    if fine_label_names[i] in trees_label:
        trees_idx.append(i)
for i in range(len(fine_labels)):
    if fine_labels[i] in trees_idx:
        test_cifar100_data.append(images[i])
test_cifar100_label = []
for i in range(np.shape(test_cifar100_data)[0]):
    test_cifar100_label.append(10)

## test_cifar100_data + test_cifar100_label

train_new_data = np.vstack((train_data,np.array(train_cifar100_data)))
train_new_label = np.hstack((train_labels,np.array(train_cifar100_label)))

test_new_data = np.vstack((test_data,np.array(test_cifar100_data)))
test_new_label = np.hstack((test_labels,np.array(test_cifar100_label)))

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
class Cifar(Dataset):
    def __init__(self, data, label, transform=None):

        # 根据是否为训练集，得到文件名前缀
        self.transform = transform


        # 读取文件数据，返回图片和标签
        self.images, self.labels = data, label
        
    def __getitem__(self, index):
        image, label = self.images[index], int(self.labels[index])

        # 如果需要转成 tensor 则使用 tansform
        if self.transform is not None:
            image = self.transform(np.array(image))  # 此处需要用 np.array(image)
        return image, label

    def __len__(self):
        return len(self.labels)

cifar_train = Cifar(
    data=train_new_data,
    label=train_new_label,
    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1037,), (0.3081,))
    ])
)
cifar_train_batch = DataLoader(
    dataset=cifar_train,
    batch_size=32,
    shuffle=True
)
cifar_test = Cifar(
    data=test_new_data,
    label=test_new_label,
    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1037,), (0.3081,))
    ])
)
cifar_test_batch = DataLoader(
    dataset=cifar_test,
    batch_size=32,
    shuffle=True
)

