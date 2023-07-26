import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt

def calculate_mean(numbers):
    mean = np.mean(numbers)
    return mean

def plot_numbers(numbers):
    n = len(numbers)
    x_values = list(range(1, n+1))
    plt.plot(x_values, numbers, 'bo-')
    plt.xlabel('Index')
    plt.ylabel('Number')
    plt.title('Number Plot')
    plt.show()

numbers = [4.5, 6.7, 2.1, 8.9, 5.3]
mean = calculate_mean(numbers)
print("Mean:", mean)

plot_numbers(numbers)
