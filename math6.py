import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define input data X and output data Y
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Input data
Y = np.array([3, 5, 7, 9, 11])  # Output data

# Apply data preprocessing using StandardScaler from sklearn
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define variables for the slope (b) and intercept (a) of the linear regression model
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# Define linear regression model
Y_pred = a + b * X

# Define loss function
loss = tf.reduce_mean(tf.square(Y_pred - Y))

# Define optimizer to minimize the loss function
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Training loop to update the variables based on the optimizer
epochs = 1000

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        Y_pred = a + b * X
        loss = tf.reduce_mean(tf.square(Y_pred - Y))

    gradients = tape.gradient(loss, [a, b])
    optimizer.apply_gradients(zip(gradients, [a, b]))

# Print the final values of a and b
print("Final values:")
print("a =", a.numpy())
print("b =", b.numpy())

# Plot the original data and the fitted line using Matplotlib
plt.scatter(X, Y, label='Original data')
plt.plot(X, Y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Predict the output for a new input
new_X = np.array([6]).reshape(-1, 1)
new_X = scaler.transform(new_X)
new_Y = a + b * new_X
print("Predicted output for X =", new_X, ":", new_Y.numpy())
