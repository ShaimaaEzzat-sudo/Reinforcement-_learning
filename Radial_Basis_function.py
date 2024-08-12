import numpy as np
'''''
create an RBF kernel for a given data matrix X with
n rows and d columns. The resulting square kernel matrix
[ K[i,j] =\cdot \exp(-\gamma \cdot ||X[i] - X[j]||^2) ]
'''''



# Generate random data (you can replace this with your actual data)
n_samples, n_features = 25000, 512
X = np.random.randn(n_samples, n_features)

# Parameters
var = 5.0
gamma = 0.01

# Compute squared Euclidean distances
X_norm = np.sum(X ** 2, axis=-1)
distances = X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)

# Compute the RBF kernel
K = var * np.exp(-gamma * distances)

print(K)
