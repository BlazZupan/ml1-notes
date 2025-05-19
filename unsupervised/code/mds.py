import torch
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define distances between cities
distances = {
    ("Novo Mesto", "Maribor"): 172,
    ("Novo Mesto", "Celje"): 83,
    ("Novo Mesto", "Koper"): 172,
    ("Novo Mesto", "Kranj"): 102,
    ("Novo Mesto", "Ljubljana"): 72,
    ("Novo Mesto", "Postojna"): 118,
    ("Maribor", "Celje"): 55,
    ("Maribor", "Koper"): 234,
    ("Maribor", "Kranj"): 156,
    ("Maribor", "Ljubljana"): 128,
    ("Maribor", "Postojna"): 180,
    ("Celje", "Koper"): 184,
    ("Celje", "Kranj"): 107,
    ("Celje", "Ljubljana"): 79,
    ("Celje", "Postojna"): 131,
    ("Koper", "Kranj"): 132,
    ("Koper", "Ljubljana"): 107,
    ("Koper", "Postojna"): 60,
    ("Kranj", "Ljubljana"): 33,
    ("Kranj", "Postojna"): 77,
    ("Ljubljana", "Postojna"): 53,
}

# Get unique cities
cities = list(set(i for pair in distances.keys() for i in pair))
n_cities = len(cities)
c2i = {city: idx for idx, city in enumerate(cities)}

# Initialize positions randomly
W = torch.randn(n_cities, 2, requires_grad=True)

# Setup optimizer
optimizer = torch.optim.SGD([W], lr=0.01)
n_epochs = 3000

# Training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Compute loss
    loss = 0
    for (city1, city2), target_dist in distances.items():
        idx1, idx2 = c2i[city1], c2i[city2]
        pos1, pos2 = W[idx1], W[idx2]
        pred_dist = torch.sqrt(torch.sum((pos1 - pos2) ** 2))
        loss += (pred_dist - target_dist) ** 2
    
    loss = loss / len(distances)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Plot results
plt.figure(figsize=(10, 10))
with torch.no_grad():
    for i, city in enumerate(cities):
        x, y = W[i].numpy()
        plt.scatter(x, y)
        plt.text(x, y, city, fontsize=12)

plt.title('MDS of Slovenian Cities')
plt.axis('equal')  # Make the plot square
plt.tight_layout()
# plt.savefig("mds-mesta.svg")
plt.show()