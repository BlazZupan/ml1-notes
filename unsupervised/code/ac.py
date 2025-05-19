# load the zoo data sets from zoo.xlsx
import pandas as pd
import torch
import torch.nn as nn
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

# Define autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 2)  # bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Initialize model and optimizer
n_features = X.shape[1]
model = Autoencoder(n_features)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    encoded, decoded = model(X_centered)
    loss = criterion(decoded, X_centered)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Get final projections
with torch.no_grad():
    encoded, _ = model(X_centered)
    Z_final = encoded.numpy()

# Convert type_list to numerical values for coloring
unique_types = list(set(type_list))
type_to_num = {t: i for i, t in enumerate(unique_types)}
type_nums = [type_to_num[t] for t in type_list]

# Plot results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(Z_final[:, 0], Z_final[:, 1], 
                     c=type_nums, cmap='tab10')

# Add labels for each point
for i, name in enumerate(name_list):
    plt.annotate(name, (Z_final[i, 0], Z_final[i, 1]),
                xytext=(5, 5), textcoords='offset points')

plt.title('Autoencoder Projection of Zoo Dataset')
plt.xlabel('First Latent Dimension')
plt.ylabel('Second Latent Dimension')

# Create custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.tab10(i/len(unique_types)), 
                            label=type_name, markersize=10)
                  for i, type_name in enumerate(unique_types)]
plt.legend(handles=legend_elements, title='Animal Types', 
          bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Print network architecture
print("\nAutoencoder Architecture:")
print(model)
