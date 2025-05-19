# create the 2D dataset, with n_clusters = 3, n_samples = 1000, random_state = 42

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import torch
import torch.optim as optim

n_clusters = 3
n_epochs = 100

# Generate data
n_features = 2
X, y = make_blobs(n_samples=200, centers=3, n_features=n_features, 
                  random_state=42, cluster_std=3.2)
X_tensor = torch.FloatTensor(X)

# plot the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Initialize centroids as a tensor with requires_grad=True
centroids = torch.randn(n_clusters, n_features, requires_grad=True)
optimizer = optim.SGD([centroids], lr=1)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    distances = torch.cdist(X_tensor, centroids)
    min_dist, assignments = torch.min(distances, dim=1)
    loss = min_dist.mean()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    # Get final assignments
    distances = torch.cdist(X_tensor, centroids)
    _, final_assignments = torch.min(distances, dim=1)
    print(final_assignments)

    # Plot results
    plt.figure(figsize=(5, 5))

    # Clustered data
    plt.scatter(X[:, 0], X[:, 1], c=final_assignments.numpy())
    plt.scatter(centroids.numpy()[:, 0], 
            centroids.numpy()[:, 1], 
            c='red', marker='x', s=200, linewidths=3)

    plt.tight_layout()
    plt.show()