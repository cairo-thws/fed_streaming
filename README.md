# Sparse Uncertainty-Informed Sampling from Federated Streaming Data

This respository is an implementation of the algorithm proposed in the paper "Sparse Uncertainty-Informed Sampling for Federated Streaming Data", currently under review.

![bw](/assets/bw.png)

This code was built on [A. Saran's VolumE Sampling for Streaming Active Learning (VeSSAL) repository](https://github.com/asaran/VeSSAL) and [M. Roeder's FedAcross repository](https://github.com/cairo-thws/FedAcross).


# Dependencies

To run the experiments and visualizations, you'll need [PyTorch and Torchvision](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/), [cholupdates](https://github.com/marvinpfoertner/cholupdates) and [DeepView](https://github.com/LucaHermes/DeepView).
Additional datasets from Domain Adaptation(Office-31, Office Home, ...) are optionally obtained with [pytorch-adapt](https://github.com/KevinMusgrave/pytorch-adapt).

We've tested the code with PyTorch 2.1.2, Torchvision 0.16, Python 3.9 and 

# Running the experiments

First, pre-train FedAcross model on CIFAR10 dataset using the "pretrain_model"-flag: \
`python run.py --model=fedacross --nQuery=100 --data=CIFAR10 --alg=fed_stream --lr=0.01 --pretrain_model` \
Using the resulting model weights for FedAcross, execute \
`python run.py --model=fedacross --nQuery=60", --data=CIFAR10 --alg=fed_stream --sort_by_pc --embs=penultimate` \
to run federated stream sampling using the FedAcross model, collecting 60 samples from the CIFAR10 test data under artifical drift.

To run precision comparison and wall-clock runtime experiments, adapt the code in minimal.py accordingly and execute \
`python minimal.py`

# Running the visualizations

Change the code to load the desired model weights and run \
`python visualize.py`
If you want to enable sample highlighting, the DeepView.py code needs to include these changes: [sample highlighting patch](patch/highlight_support.patch).

# Bibliography
If you find our work to be useful in your research, please cite:
```
TBA
```
