import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k  # Number of clusters
        self.max_iters = max_iters  # Maximum iterations
        self.centroids = None

    def fit(self, X):
        # Step 1: Randomly choose 'k' data points as initial centroids
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iters):  # Move loop inside 'fit'
            # Step 2: Assign each point to the nearest centroid
            labels = np.array([np.argmin(np.linalg.norm(x - self.centroids, axis=1)) for x in X])

            # Step 3: Compute new centroids as the mean of assigned points
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Step 4: Check if centroids don't change
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids  # Update centroids

        self.labels = labels  # Store labels for later use

    def predict(self, X):
        # Assign each point to the closest centroid
        return np.array([np.argmin(np.linalg.norm(x - self.centroids, axis=1)) for x in X])

# Generate synthetic dataset (300 points, 3 clusters)
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Create and fit the K-Means model
kmeans = KMeans(k=3)
kmeans.fit(X)

# Predict cluster labels for visualization
labels = kmeans.predict(X)

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X')  # Centroids
plt.title("K-Means Clustering Results")
plt.show()