import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def calculate_mean_and_plot():
    np.random.seed(0)
    X = np.random.rand(3,10)
    y = np.random.rand(3,10)
    mean1 = np.mean(X)
    mean2 = np.mean(y)
    print(mean1)
    print(mean2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    reg = LinearRegression().fit(X, y)
    print("-----------reg.coef_-----------")
    print(reg.coef_)
    print("-----------reg.score-----------")
    print(reg.score)
    print("-----------X_train-----------")
    print(X_train)
    print("-----------X_test-----------")
    print(X_test)
    print("-----------y_train-----------")
    print(y_train)
    print("-----------y_test-----------")
    print(y_test)
    plt.scatter(X_train,y_train)
    plt.scatter(X_test,y_test)
    plt.show()

calculate_mean_and_plot()