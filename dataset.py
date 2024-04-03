import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
from pytorch_adapt import datasets as adapt_ds

def get_dataset(name, path):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'OFFICE31_AMAZON':
        return get_Office31(path, "amazon")
    elif name == 'OFFICE31_WEBCAM':
        return get_Office31(path, "webcam")
    elif name == 'OFFICE31_DSLR':
        return get_Office31(path, "dslr")
    elif name == 'OFFICEHOME_ART':
        return get_OfficeHome(path, "art")
    elif name == 'OFFICEHOME_CLIPART':
        return get_OfficeHome(path, "clipart")
    elif name == 'OFFICEHOME_PRODUCT':
        return get_OfficeHome(path, "product")
    elif name == 'OFFICEHOME_RW':
        return get_OfficeHome(path, "real")

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    print('svhn data shape: ',X_tr.shape, Y_tr.shape)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_Office31(path, domain):
    data_tr = adapt_ds.Office31(root=path, domain=domain, train=True, download=True) #da.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = adapt_ds.Office31(root=path, domain=domain, train=False, download=True)
    
    datapoint_list_tr = list()
    for image_path_tr in data_tr.img_paths:
        datapoint = pil_loader(image_path_tr)
        datapoint_list_tr.append(datapoint)
    X_tr = np.asarray(datapoint_list_tr)
    Y_tr = torch.from_numpy(np.array(data_tr.labels))
    
    datapoint_list_te = list()
    for image_path_te in data_te.img_paths:
        datapoint = pil_loader(image_path_te)
        datapoint_list_te.append(datapoint)
    X_te = np.asarray(datapoint_list_te)
    Y_te = torch.from_numpy(np.array(data_te.labels))
    
    return X_tr, Y_tr, X_te, Y_te


def get_OfficeHome(path, domain):
    data_tr = adapt_ds.OfficeHome(root=path, domain=domain, train=True, download=True) #da.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = adapt_ds.OfficeHome(root=path, domain=domain, train=False, download=True)
    
    datapoint_list_tr = list()
    for image_path_tr in data_tr.img_paths:
        datapoint = pil_loader(image_path_tr)
        datapoint_list_tr.append(datapoint)
    X_tr = np.asarray(datapoint_list_tr)
    Y_tr = torch.from_numpy(np.array(data_tr.labels))
    
    datapoint_list_te = list()
    for image_path_te in data_te.img_paths:
        datapoint = pil_loader(image_path_te)
        datapoint_list_te.append(datapoint)
    X_te = np.asarray(datapoint_list_te)
    Y_te = torch.from_numpy(np.array(data_te.labels))
    
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'OFFICE31_AMAZON' or name == 'OFFICE31_WEBCAM' or name == 'OFFICE31_DSLR':
        return DataHandler3
    else:
        return DataHandler3

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler5(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        y = self.Y[index]
        x = pil_loader(self.X[index])
        if self.transform is not None:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


def pil_loader(path: str, resize=224) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
        new_img = img.resize((resize, resize))
        return np.array(new_img)