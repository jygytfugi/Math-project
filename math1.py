import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])  # Data points
w = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  # Corresponding weights

weighted_mean = np.sum(w * x) / np.sum(w)
print("Weighted Mean:", weighted_mean)

plt.bar(x, w)
plt.xlabel('Data Points')
plt.ylabel('Weights')
plt.title('Weighted Data')
plt.show()
