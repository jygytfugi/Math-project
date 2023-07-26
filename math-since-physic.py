import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Create data points
x = np.array([1, 2, 3, 4, 5,5,3])
y = np.array([3, 5, 4, 6, 8,6,1])

# Perform PCA
data = np.column_stack((x, y))
pca = PCA(n_components=1)
transformed_data = pca.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Get the principal component (eigenvector)
principal_component = pca.components_[0]
print("Principal Component:", principal_component)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Plotting example
plt.scatter(x, y, label='Original Data')
plt.plot([0, principal_component[0]], [0, principal_component[1]], color='red', label='Principal Component')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Principal Component Analysis (PCA)')
plt.legend()
plt.show()
