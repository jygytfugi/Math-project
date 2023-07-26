import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Create data points
x = np.array([1, 2,2, 3, 4, 5,5,3])
y = np.array([3, 5,3, 4, 6, 8,6,1])

# Perform PCA
data = np.column_stack((x, y))
pca = PCA(n_components=1)
transformed_data = pca.fit_transform(data)

# Get the principal component (eigenvector)
principal_component = pca.components_[0]
print("Principal Component:", principal_component)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Plotting example
plt.scatter(x, y, label='Original Data')
plt.plot([0, principal_component[0]], [0, principal_component[1]], color='red', label='Principal Component')
plt.plot(x_test, y_pred, color='green', label='Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Principal Component Analysis and Linear Regression')
plt.legend()
plt.show()
