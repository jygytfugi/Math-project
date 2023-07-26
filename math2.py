import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

x = np.array([1, 2, 3, 4, 5,6,7])  # Data points
w = np.array([0.1, 0.3, 0.4, 0.2, 0.2,0.1,0.4])  # Corresponding weights

weighted_mean = np.sum(w * x) / np.sum(w) + np.sum(x)
print("Weighted Mean:", weighted_mean)

normalized_weights = preprocessing.normalize(w[:, np.newaxis], axis=0).ravel()
X_train, X_test, y_train, y_test = train_test_split(x, w, test_size=0.33, random_state=42)

plt.bar(x,normalized_weights)
plt.bar(X_train,y_train)
plt.bar(X_test,y_test)
plt.xlabel('Data Points')
plt.ylabel('Normalized Weights')
plt.title('Weighted Data')
plt.show()
