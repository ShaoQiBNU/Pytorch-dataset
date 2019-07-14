Pytorch自定义dataset
===

# 一. 简介

> Pytorch读取自己的数据集需要自定义Dataset，然后调用DataLoader产生batch数据。Pytorch的数据读取主要包含三个类: 
>
> - Dataset 
> - DataLoader 
> - DataLoaderIter
>
> 这三者是依次封装的关系:  **Dataset**  被封装进 **DataLoader**，**DataLoader** 被装进 **DataLoaderIter**。

# 二. 代码解析

## (一) torch.utils.data.Dataset

> 这个类的源代码解析见：https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset ，具体解析如下：

```python
class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
```

> 这是一个抽象类, 自定义的Dataset需要继承它并且实现下面两个成员方法： __getitem__() 和 __len__()

## (二) 实例

### 1. MNIST

> 查看官方的MNIST的代码，如下：

```python
class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        # PASS

    def __repr__(self):
        # PASS
        return fmt_str
```

#### (1) \\__getitem__()

> 该方法实现了每次如何读取数据，以及对数据做的各种处理 transform，常用的有resize、Resize、RandomCrop、Normalize和ToTensor(可以把一个 `PIL或numpy` 图片转为 `torch.Tensor`)

#### (2) \\__len__()

> 返回整个数据集的长度

### 2. 自定义

> 依照1的例子，实现自定义dataset，采用的数据集为iris数据集，csv文件，代码如下：

```Python
###################### load packages ########################
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


###################### IrisDataset class ########################
class IrisDataset(torch.utils.data.Dataset):

    ############ init ###########
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.train = train
        self.target_transform = target_transform
        self.data = pd.read_csv(data_path, sep=',')
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.3)

    ############ get data ###########
    def __getitem__(self, index):
        labels = {'setosa': 0, 'virginica': 1, 'versicolor': 2}

        if self.train:
            feature, label = self.train_data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]
        else:
            feature, label = self.test_data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]

        return feature, int(labels[label])


    ############ get data length ###########
    def __len__(self):
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)
```

## (二) torch.utils.data.DataLoader

> 这个类的代码具体见：https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html ，解析如下：

```python
class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use :func:`torch.initial_seed()` to access the PyTorch seed for
              each worker in :attr:`worker_init_fn`, and use it to set other
              seeds before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        # PASS
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)
```

> init里的主要参数如下：
>
> - `dataset` : 即上面自定义的 dataset
> - `collate_fn`: 这个函数用来打包 batch
> - `num_worker`: 非常简单的多线程方法，只要设置为>=1，就可以多线程预读数据

> 这个类是下面`DataLoaderIter` 的一个框架，主要功能如下：
>
> - 定义了一堆成员变量，赋给 `DataLoaderIter`
> - 有一个 `__iter__()` 函数, 把自己 "装进" DataLoaderIter 里面

```python
def __iter__(self):
        return DataLoaderIter(self)
```

## (三) torch.utils.data.dataloader.DataLoaderIter

> `DataLoader` 是`DataLoaderIter`的一个框架，用来传给`DataLoaderIter` 一堆参数，并把自己装进`DataLoaderIter` 里。

## (四) 实例分析

> 自定义一个框架如下：

```python
class CustomDataset(Dataset):
   # 自定义自己的dataset

dataset = CustomDataset()
dataloader = Dataloader(dataset, ...)

for data in dataloader:
   # training...
```

> 在for 循环里有三点操作：
>
> - 调用了 `dataloader` 的 `__iter__()` 方法，产生了一个`DataLoaderIter`
> - 反复调用 `DataLoaderIter` 的 `__next__()` 来得到 batch，具体操作就是：多次调用 dataset 的 `__getitem__()` 方法 (如果`num_worker>0`就多线程调用)，然后用`collate_fn`来把它们打包成batch，中间还会涉及到`shuffle` 以及 `sample`的方法等。
> - 数据读完后，`__next__()` 抛出一个 `StopIteration` 异常， for循环结束，`dataloader` 失效。

# 三. 代码

## (一) csv数据集

> 读取iris数据集，搭建单层神经网络训练模型，代码如下：

### main

```python
###################### load packages ########################
import torch
import torch.utils.data
import Dataset
import model
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

###################### 参数设置 ########################
path = "iris.csv"
batch_size = 10
epoch = 2

input_size = 4
hidden_size = 4
num_classes = 3


###################### load data ########################
dataset_train = Dataset.IrisDataset(path, train=True)
dataset_test = Dataset.IrisDataset(path, train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)


################# train #################
def train(train_loader, model, optimizer, cost, epoch):

    print("Start training:")

    ########### epoch ##########
    for i in range(epoch):
        train_correct = 0
        total_cnt = 0

        ############## batch #############
        for batch_idx, (data, target) in enumerate(train_loader):

            ############ get data and target #########
            data, target = Variable(data), Variable(target)

            ############ optimizer ############
            optimizer.zero_grad()

            ############ get model output ############
            output = model(data)

            ############ get predict label ############
            _, pred = torch.max(output.data, 1)

            ############ loss ############
            loss = cost(output, target)
            loss.backward()

            ############ optimizer ############
            optimizer.step()

            ############ result ############
            total_cnt += data.data.size()[0]
            train_correct += torch.sum(pred == target.data)

            ############ show train result ############
            if (batch_idx+1) % 10 == 0:
                print("epoch: {}, batch_index: {}, train loss: {:.6f}, train correct: {:.2f}%".format(
                    i, batch_idx+1, loss, 100*train_correct/total_cnt))

    print("Training is over!")


################# test #################
def test(test_loader, model, cost):

    print("Start testing:")

    ############ batch ############
    for batch_idx, (data, target) in enumerate(test_loader):

        ############ get data and target ############
        data, target = Variable(data), Variable(target)

        ############ get model output ############
        output = model(data)


        ############ get predict label ############
        _,pred = torch.max(output.data, 1)

        ############ loss ############
        loss = cost(output, target)

        ############ accuracy ############
        test_correct = torch.sum(pred == target.data)

        print("batch_index: {}, test loss: {:.6f}, test correct: {:.2f}%".format(
                batch_idx + 1, loss.item(), 100*test_correct/data.data.size()[0]))

    print("Testing is over!")


###################### model ########################
fcn = model.NeuralNet(input_size, hidden_size, num_classes)

################# optimizer and loss #################
optimizer = optim.Adam(fcn.parameters(), lr=0.001)
cost = nn.CrossEntropyLoss()


################# train #################
train(data_loader_train, fcn, optimizer, cost, epoch)

################# test #################
test(data_loader_test, fcn, cost)
```

### Dataset

```python
###################### load packages ########################
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


###################### IrisDataset class ########################
class IrisDataset(torch.utils.data.Dataset):

    ############ init ###########
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.train = train
        self.target_transform = target_transform
        self.data = pd.read_csv(data_path, sep=',')
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.3)

    ############ get data ###########
    def __getitem__(self, index):
        labels = {'setosa': 0, 'virginica': 1, 'versicolor': 2}

        if self.train:
            feature, label = self.train_data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]
        else:
            feature, label = self.test_data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]

        return feature, int(labels[label])


    ############ get data length ###########
    def __len__(self):
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)

```

### model

```python
###################### load packages ########################
import torch.nn as nn


###################### model  ########################
'''
Fully connected neural network
'''
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x.float())
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

## (二) 影像数据集
> 对于存储在文件夹里的图片数据集，首先需要生成一个存储图片路径和标签的文件，用于下面读取数据，具体代码如下：

### 文件生成

```python
# -*- coding: utf-8 -*-
"""
读取枸杞数据，将数据的存储路径和label写入csv文件
"""

##################### load packages #####################
import numpy as np
import os
from PIL import Image
from skimage import io, data
import scipy.io
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd


labels = {'MQ2':0, 'NNQ9':1, 'NQ1':2, 'NQ5':3, 'NQ7':4, 'NQ8':5}


##################### load gouqi data ##########################
def gouqi_preprocess(gouqi_folder):
    '''
    gouqi_floder: gouqi original path 原始枸杞的路径
    '''

    ######## gouqi data path 数据保存路径 ########
    gouqi_dir = list()

    ######## gouqi data dirs 生成保存数据的名称 ########
    for img_dir in os.listdir(gouqi_folder):

        for img in os.listdir(os.path.join(gouqi_folder, img_dir)):
            gouqi_dir.append(img)

    ######## random shuffle gouqi data dirs 打乱数据的绝对路径和名称排序 ########
    random.shuffle(gouqi_dir)

    length = int(0.8 * len(gouqi_dir))


    ###################### gouqi train data #####################
    gouqi_train=[]
    
    for img_dir in gouqi_dir[0:length]:

        ######## get label ########
        label = labels[img_dir.split('_')[0]]
        
        gouqi_train.append(['./data/'+img_dir,label])

    gouqi_train_df = pd.DataFrame(gouqi_train)
    gouqi_train_df.to_csv("C:/Users/shaoqi/Desktop/gouqi_train.csv", index=False, header=None)
    
    print("train is over!")
    
    
    
    ###################### gouqi test data #####################
    gouqi_test=[]
    
    for img_dir in gouqi_dir[length:len(gouqi_dir)]:  

        ######## get label ########
        label = labels[img_dir.split('_')[0]]
        
        gouqi_test.append(['./data/'+img_dir,label])
    
    gouqi_test_df = pd.DataFrame(gouqi_test)
    gouqi_test_df.to_csv("C:/Users/shaoqi/Desktop/gouqi_test.csv", index=False, header=None)
    
    print("test is over!")


################ main函数入口 ##################
if __name__ == '__main__':
    ######### gouqi path 枸杞数据存放路径 ########
    gouqi_folder = 'D:\\SQ\\枸杞\\分类\\单个'

    ######## 数据预处理 ########
    gouqi_preprocess(gouqi_folder)
```


### Dataset
```python
###################### load packages ########################
import torch.utils.data
import pandas as pd
from skimage import io, data

###################### Dataset class ########################
class ImageDataset(torch.utils.data.Dataset):

    ############ init ###########
    def __init__(self, filenames, train=True, transform=None):
        self.transform = transform
        self.train = train

        self.train_data = pd.read_csv(filenames, sep=',', header=None)
        self.test_data = pd.read_csv(filenames, sep=',', header=None)


    ############ get data ###########
    def __getitem__(self, index):

        if self.train:
            fn, label = self.train_data.iloc[index:index+1, 0].values[0], self.train_data.iloc[index:index+1, 1].values[0]
        else:
            fn, label = self.test_data.iloc[index:index + 1, 0].values[0], self.test_data.iloc[index:index + 1, 1].values[0]

        img = io.imread(fn)
        label = int(label)

        return img, label


    ############ get data length ###########
    def __len__(self):
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)
```
