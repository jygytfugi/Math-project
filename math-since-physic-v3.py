import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Generate some sample data for demonstration
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Support Vector Machines (SVM)
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train.ravel())

# Generate some test data for predictions
X_test_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)

# Make predictions
lr_predictions = lr_model.predict(X_test_range)
svm_predictions = svm_model.predict(X_test_range)

# Visualize the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test_range, lr_predictions, color='red', label='Linear Regression')
plt.plot(X_test_range, svm_predictions, color='green', label='SVM')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression vs SVM')
plt.legend()
plt.show()
