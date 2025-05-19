# load the zoo data sets from zoo.xlsx
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Read the Excel file
df = pd.read_excel('zoo.xlsx')

# Select only numerical columns for X
X = df.select_dtypes(include=['int64', 'float64']).values
column_names = df.columns.tolist()[2:]
type_list = df.iloc[:, 0].tolist()  # First column
name_list = df.iloc[:, 1].tolist()  # Second column

# Convert data to PyTorch tensor and standardize
X_tensor = torch.FloatTensor(X)
X_centered = X_tensor - X_tensor.mean(dim=0)

# Initialize projection matrix (2D PCA)
n_features = X.shape[1]
n_components = 2
W = torch.randn(n_features, n_components, requires_grad=True)
W.data = torch.nn.functional.normalize(W.data, dim=0)  # Initialize with normalized vectors

# Setup optimizer
optimizer = optim.SGD([W], lr=0.01)
n_epochs = 1000
norm_rate = 0.5  # rate for L1 normalization penalty

# Training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    Z = X_centered @ W  # project data
    variance = -torch.var(Z, dim=0).sum()  # compute variance, negative because we want to maximize
    orthogonality_penalty = torch.sum((W.T @ W - torch.eye(n_components))**2)
    l1_norm_penalty =  torch.sum(torch.abs(W))  # L1 normalization penalty
    loss = variance + 10.0 * orthogonality_penalty + norm_rate *l1_norm_penalty
    
    loss.backward()
    optimizer.step()
    
    # Re-normalize W after each step
    with torch.no_grad():
        W.data = torch.nn.functional.normalize(W.data, dim=0)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Variance: {-variance.item():.4f}, '
              f'Orthogonality: {orthogonality_penalty.item():.4f}, '
              f'L1 norm: {l1_norm_penalty.item():.4f}')

# Get final projections
with torch.no_grad():
    Z_final = X_centered @ W

# Convert type_list to numerical values for coloring
unique_types = list(set(type_list))
type_to_num = {t: i for i, t in enumerate(unique_types)}
type_nums = [type_to_num[t] for t in type_list]

# Plot results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(Z_final[:, 0].numpy(), Z_final[:, 1].numpy(), 
                     c=type_nums, cmap='tab10')

# Add labels for each point
for i, name in enumerate(name_list):
    plt.annotate(name, (Z_final[i, 0].numpy(), Z_final[i, 1].numpy()),
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('PCA1')
plt.ylabel('PCA2')

# Create custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.tab10(i/len(unique_types)), 
                            label=type_name, markersize=10)
                  for i, type_name in enumerate(unique_types)]
plt.legend(handles=legend_elements, title='Animal Types', 
          bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

with torch.no_grad():
    for pc in range(2):  # For first and second principal component
        # Get absolute weights for current component
        weights = W[:, pc].abs().numpy()
        feature_weights = list(zip(column_names, weights))
        
        # Sort by weight magnitude and get top features
        top_features = sorted(feature_weights, key=lambda x: x[1], reverse=True)[:n_features]
        
        # Print results
        print(f"\nLoadings for PC{pc+1}:")
        print("Feature".ljust(20), "Weight")
        print("-" * 30)
        for feature, weight in top_features:
            print(f"{feature.ljust(20)} {weight:.2f}")
