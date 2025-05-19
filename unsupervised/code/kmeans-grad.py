# create the 2D dataset, with n_clusters = 3, n_samples = 1000, random_state = 42

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import torch

n_clusters = 3
n_epochs = 100
learning_rate = 1.0

# Generate data
n_features = 2
X, y = make_blobs(n_samples=200, centers=3, n_features=n_features, 
                  random_state=42, cluster_std=3.2)
X_tensor = torch.FloatTensor(X)

# plot the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Initialize centroids as a tensor with requires_grad=True
C = torch.randn(n_clusters, n_features, requires_grad=True)

for epoch in range(n_epochs):
    # Compute distances between points and centroids
    distances = torch.cdist(X_tensor, C)
    min_dist, assignments = torch.min(distances, dim=1)
    loss = min_dist.mean()
    
    # Compute gradient
    loss.backward()
    
    # Update centroids manually
    with torch.no_grad():
        C -= learning_rate * C.grad
        C.grad.zero_()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Get final assignments
with torch.no_grad():
    distances = torch.cdist(X_tensor, C)
    _, final_assignments = torch.min(distances, dim=1)
    print(final_assignments)

    # Plot results
    plt.figure(figsize=(5, 5))

    # Clustered data
    plt.scatter(X[:, 0], X[:, 1], c=final_assignments.numpy())
    plt.scatter(C.detach().numpy()[:, 0], 
            C.detach().numpy()[:, 1], 
            c='red', marker='x', s=200, linewidths=3)

    plt.tight_layout()
    plt.show()