
import numpy as np
import torch
from sklearn.datasets import make_blobs
from torch_kmeans import KMeans

#%%

# function to generate some clustering data
def get_data(bs: int = 1,
             n: int = 20,
             d: int = 2,
             k: int = 4,
             different_k: bool = False,
             k_lims = (2, 5),
             add_noise: bool = True,
             fp_dtype = torch.float32,
             seed: int = 42):
    torch.manual_seed(seed)
    if different_k:
        a, b = k_lims
        k = torch.randint(low=a, high=b, size=(bs,)).long()
    else:
        k = torch.empty(bs).fill_(k).long()

    # generate pseudo clustering data
    x, y = [], []
    for i, k_ in enumerate(k.numpy()):
        x_, y_ = make_blobs(n_samples=n, centers=k_, n_features=d, random_state=seed+i)
        x.append(x_)
        y.append(y_)
    x = torch.from_numpy(np.stack(x, axis=0))
    y = torch.from_numpy(np.stack(y, axis=0))
    if add_noise:
        x += torch.randn(x.size())

    return x.to(fp_dtype), y, k



#%%
# create some data (BS, N, D)
# i.e. 1 instance with N=20 points and D=2 features
BS = 1
K = 4
x, _, _ = get_data(bs=BS, n=20, d=2, k=K, different_k=False)
print(x.shape)

#%%

# initialize model
model = KMeans(n_clusters=K)
print(model)

#%%
# the returned result tuple has many different attributes
# associated with the input and the clustering result:

# ClusterResult:

# "Named and typed result tuple for kmeans algorithms"

# - labels: label for each sample in x
# - centers: corresponding coordinates of cluster centers
# - inertia: sum of squared distances of samples to their closest cluster center
# - x_org: original x
# - x_norm: normalized x which was used for cluster centers and labels
# - k: number of clusters
# - soft_assignment: assignment probabilities of soft kmeans

#%%
# pytorch style call
result = model(x)
print(result)

#%%
# Instead we can also use scikit-learn style calls:
model = KMeans(n_clusters=K)
fitted_model = model.fit(x)
print(fitted_model.is_fitted)
labels = fitted_model.predict(x)
print(labels)

#%%

# or directly
labels = KMeans(n_clusters=K).fit_predict(x)
print(labels)

#%%

# instead of at model initialization, we can also specify the number of clusters
# directly in the forward pass

model = KMeans()
result = model(x, k=K)
print(result.k)

#%%

# the algorithm works for mini-batches of instances
# here we create a batch of 3 instances with N=20 points and D=2 features
BS = 3
K = 4
x, _, _ = get_data(bs=BS, n=20, d=2, k=K, different_k=False)
print(x.shape)

model = KMeans()
result = model(x, k=K)
print(result.k)
print(result.labels)

#%%

# we can also provide different numbers of clusters per instance
BS = 3
K = 4
x, _, k_per_instance = get_data(bs=BS, n=20, d=2, k=K, different_k=True)

# in case of different clusters we need to provide a Tensor
# holding the number of clusters per instance
print(k_per_instance)

model = KMeans()
result = model(x, k=k_per_instance)
print(result.k)
print(result.labels)

#%%

# if we have a good prior idea of where the cluster centers should be located,
# e.g. based on domain knowledge, we can provide these centers to the algorithm
# to speed up convergence and possibly improve the final quality of the clustering assignment

BS = 1
K = 4
x, y, k_per_instance = get_data(bs=BS, n=20, d=2, k=K, different_k=False)

# here we just use the ground truth labels from the generator
# to set the centers to the group mean + a bit of noise

centers = []
for i in range(K):
    msk = y == i
    centers.append(x[msk].mean(dim=0) + torch.randn(1))
centers = torch.stack(centers).unsqueeze(0) # add batch dimension
print(centers)

# initialize only one run, since we only provide one set of centers
# (otherwise a warning will be raised)

model = KMeans(num_init=1)
result = model(x, k=K, centers=centers)
print(result.centers)

#%%

import torch
torch.cuda.is_available()



#%%
# One nice feature of torch_kmeans is the possibility to leverage huge parallelism via execution on GPU.
# (On Colab you need to change the runtime type to 'GPU')

# To use the GPU if it is available, simply transfer your inputs to the GPU device and the model will automatically
# execute the algorithm on GPU. In order to keep thins simple and easy to maintain, torch_kmeans does not use a dedicated
# custom GPU kernel for kmeans but simply leverages the torch tensor operators on GPU for the most expensive computation steps.
# (Most of these steps are JIT compiled via torch.script as well)

#assert torch.cuda.is_available()
device = torch.device("cuda")

x_cuda = x.to(device=device)
print(x_cuda.is_cuda, x_cuda.device)

model = KMeans()
result = model(x_cuda, k=K)
# be careful if you use the tensors of the result object
# since now they are located on the GPU:
lbl = result.labels
print(lbl.is_cuda, lbl.device)
# to transfer them back, just use '.cpu()'
lbl = lbl.cpu()
print(lbl.is_cuda, lbl.device)

#%%

# The massive parallelization can for example also be used to very efficiently
# find the best number of clusters for a given dataset by computing KMeans for
# different numbers of clusters k all in parallel and using the 'Elbow method'
# ([https://en.wikipedia.org/wiki/Elbow_method_(clustering)](https://en.wikipedia.org/wiki/Elbow_method_(clustering))).

BS = 1
K = 6
x, _, _ = get_data(bs=BS, n=50, d=2, k=6, different_k=False)

#assert torch.cuda.is_available()
device = torch.device("cuda")

# replicate instances of x
n_tries = 8
x = x.expand(n_tries, 50, 2)
x_cuda = x.to(device=device)
# use different k between 2 and 10
k_per_isntance = torch.arange(start=2, end=10).to(device=device)

model = KMeans()
result = model(x_cuda, k=k_per_isntance)
# find k according to 'elbow method'
for k, inrt in zip(k_per_isntance, result.inertia.cpu()):
    print(f"k={k}: {inrt}")
# the decrease in inertia after k=6 is much smaller than for the prior steps,
# forming the characteristic 'elbow'