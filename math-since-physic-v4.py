import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate some sample data for demonstration
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)

# Support Vector Machines (SVM)
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train.ravel())
svm_predictions = svm_model.predict(X_test)
svm_mse = mean_squared_error(y_test, svm_predictions)

# Neural Networks
nn_model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=42)
nn_model.fit(X_train, y_train.ravel())
nn_predictions = nn_model.predict(X_test.reshape(-1, 1))
nn_mse = mean_squared_error(y_test, nn_predictions)

# Visualize the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, lr_predictions, color='red', label='Linear Regression (MSE: %.2f)' % lr_mse)
plt.plot(X_test, svm_predictions, color='green', label='SVM (MSE: %.2f)' % svm_mse)
plt.plot(X_test, nn_predictions, color='orange', label='Neural Network (MSE: %.2f)' % nn_mse)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression vs SVM vs Neural Network')
plt.legend()
plt.show()
