"""
MIT License

Copyright (c) 2024 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Any, Iterator
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchvision import datasets

from pytorch_adapt import datasets as adapt_ds
from PIL import Image
import numpy as np
import random

import torch.nn.functional as F
import pickle

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import torchvision.transforms as transforms 

# deepview code
from deepview import DeepView
from deepview.evaluate import evaluate_umap
import matplotlib.pyplot as plt
from collections import Counter
import statistics


class fedacross_client(nn.Module):
        def __init__(self, num_classes=40, adapt_module_dim=256, pretrain=False):
            super(fedacross_client, self).__init__()
            resnet50_pre = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone_out_features = resnet50_pre.fc.in_features
            self.backbone = torch.nn.Sequential(*(list(resnet50_pre.children())[:-1]))
            self.adapt_module_dim = adapt_module_dim
            self.embed_dim = self.backbone_out_features
            self.adapt_module = nn.Sequential(
                nn.Linear(self.backbone_out_features, self.adapt_module_dim),
                nn.BatchNorm1d(self.adapt_module_dim),
                nn.ReLU())
            self.head = nn.Linear(self.adapt_module_dim, num_classes)
            self.pretrain = pretrain
            
        def forward(self, x):
            #print(x.shape)
            #B, W, H, C = x.shape
            #x = x.view(B, C, W, H)
            x = self.backbone(x)            
            x = x.view(-1, self.backbone_out_features)
            emb = self.adapt_module(x)
            out = self.head(emb)
            return out, emb
        
        def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
            if self.pretrain:
                params =[{"params": self.backbone.parameters(), "lr": 0.1},
                {"params": list(self.adapt_module.parameters()) + list(self.head.parameters()), "lr": 0.001}  # 公用层learning_rate应取平均
                ]
            else:
                params = list(self.adapt_module.parameters())
            return params

        def get_embedding_dim(self):
            return self.adapt_module_dim
        

def pil_loader(path: str, resize=224) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
        new_img = img.resize((resize, resize))
        return np.array(new_img)
        
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

def get_CIFAR10(path, transform=None):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True, transform=transform)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True, transform=transform)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te
        
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")

print('Load model')
model = fedacross_client(num_classes=10)
print('Load weights')
model.load_state_dict(torch.load("model_weights_CIFAR10.pt"))
model.eval()
model.to(device)


print('Load data')
preprocess = transforms.Compose([
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = get_CIFAR10("data", preprocess)
softmax = torch.nn.Softmax(dim = 1)


def filter_samples_by_class(samples, labels_in, k):
    """
    Reduce the lists of samples and labels to only contain k samples for each unique class.

    Parameters:
    - samples (list): The list of samples.
    - labels (list): The list of corresponding labels.
    - k (int): The maximum number of samples to keep for each class.

    Returns:
    - filtered_samples (list): The reduced list of samples for each class.
    - filtered_labels (list): The reduced list of labels for each class.
    """
    labels = [t.item() for t in labels_in]
    class_samples = {}
    for sample, label in zip(samples, labels):
        if label not in class_samples:
            class_samples[label] = [sample]
        elif len(class_samples[label]) < k:
            class_samples[label].append(sample)

    # Flatten the dictionary back into lists
    filtered_samples = []
    filtered_labels = []
    for label, samples in class_samples.items():
        filtered_samples.extend(samples)
        filtered_labels.extend([label] * len(samples))

    return filtered_samples, [torch.tensor(i) for i in filtered_labels]


def get_samples_and_labels(ids, dataset, transform, source="training"):
    samples = []
    labels = []
    if source == "training":
        X = dataset[0]
        y = dataset[1]
    else:
        X = dataset[2]
        y = dataset[3]

    for id_ in ids:
        # Retrieve the sample and label using the ID
        sample = X[id_]
        label = y[id_]
        
        # Apply the transformation to the sample
        transformed_sample = transform(sample)
        
        # Append the transformed sample and label to their respective lists
        samples.append(transformed_sample)
        labels.append(label)

    return samples, labels


def torch_wrapper(x):
    with torch.inference_mode():
        x = np.array(x, dtype=np.float32)
        tensor = torch.from_numpy(x).to(device)
        logits, _ = model(tensor)
        #probs = softmax(logits).cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs


# the classes in the dataset to be used as labels in the plots
classes_str = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualization(image, point2d, pred, label=None, title=None):
    print("Vis called")
    f, a = plt.subplots()
    predict_str = classes_str[pred]
    
    label_str = "?" if label==None else classes_str[label]
    title_n = "Model Prediction: " + predict_str + ", Label: " + label_str
    a.set_title(title_n)
    a.imshow(image.transpose([1, 2, 0]))
    plt.show()



classes = np.arange(10)
# --- Deep View Parameters ----
batch_size = 128
max_samples = 600
data_shape = (3, 32, 32)
resolution = 100
N = 10
lam = 0.65
cmap = 'Paired'
# to make shure deepview.show is blocking,
# disable interactive mode
interactive = False
title = ''

print("Load DeepView")
deepview = DeepView(torch_wrapper, classes_str, max_samples, batch_size, data_shape, 
	lam=lam, cmap=cmap, interactive=interactive, title=title, data_viz=visualization)


### Load selected sample IDs here
selected_samples_id = []
with open("chosenSamples_test_60", "rb") as fp:   # Unpickling
    selected_samples_id = pickle.load(fp)
    
prefetch_X, prefetch_y = get_samples_and_labels(selected_samples_id, dataset, preprocess, source="test")
# print label statistics
frequency_vec = torch.tensor([t.item() for t in prefetch_y])
unique_elements, counts = torch.unique(frequency_vec, return_counts=True)
for element, count in zip(unique_elements, counts):
    print(f"Class {element.item()} was sampled {count.item()} times")

print("Mean of sampled set is % s" %(statistics.mean(counts.tolist())))
print("Variance of sampled set is % s" %(statistics.variance(counts.tolist())))
print("Stdev of sampled set is % s" %(statistics.stdev(counts.tolist())))

# random samples without selected
random_ids = random.sample(range(len(dataset[0])), 440)
a_filtered = [elem for elem in random_ids if elem not in selected_samples_id]

# combine lists and merge
input_sample_ids = selected_samples_id + a_filtered
input_sample_ids.sort()

# fetch samples
selected_samples_X, selected_samples_y = get_samples_and_labels(input_sample_ids, dataset, preprocess)


#selected_samples_X, selected_samples_y = filter_samples_by_class(selected_samples_X, selected_samples_y, 2)

print("Add samples")
deepview.add_samples(selected_samples_X, selected_samples_y, input_sample_ids)

# highlight specific samples
print("Highlight samples")
deepview.highlight_samples(selected_samples_id, "white")

print("Show samples")
deepview.show()

# add more samples
#random_ids = random.sample(range(len(dataset[0])), 200)
#selected_samples_X, selected_samples_y = get_samples_and_labels(random_ids, dataset, preprocess)
#deepview.add_samples(selected_samples_X, selected_samples_y, random_ids)
#deepview.update_mappings()


#deepview.show()
'''
for l in np.linspace(0.0, 1.0, 6):
    deepview.verbose = False
    deepview.set_lambda(l)
    q_knn = evaluate_umap(deepview, True)
    print('Lambda: %.2f - Q_kNN: %.3f' % (l, q_knn))
'''
deepview.close()