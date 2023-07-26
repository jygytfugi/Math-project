from __future__ import print_function, division, absolute_import
import numpy as np
from matplotlib.pylab import imshow,show
from timeit import default_timer as timer
from numba import cuda
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

@cuda.jit(device=True)
def mandelbort(x,y,max_iters):
    c = complex(x,y)
    z = 0.0j
    i = 0

    for i in range(max_iters):
        z = z * z + c
        if(z.real * z.real + z.imag * z.imag) >= 4:
            return i
        
        return 255
    
@cuda.jit
def create_fractal(min_x,max_x,min_y,max_y,image,iters):
    width = image.shape[1]
    height=image.shape[0]

    pixel_size_x = (max_x - min_x)/width
    pixel_size_y = (max_y - min_y)/height

    x,y=cuda.grid(2)

    if x < width and y < height:
        real = min_x + x*pixel_size_x
        imag = min_y + y*pixel_size_y
        color = mandelbort(real,imag,iters)
        image[y,x]=color

def ml():
    # Generate some sample data for demonstration
    np.random.seed(0)
    X = np.random.rand(100,1) 
    y = np.random.rand(100,1) 

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

    # Show information in X_train, X_test, y_train, y_test
    print("------------------X_train------------------")
    print(X_train)
    print("------------------X_test------------------")
    print(X_test)
    print("------------------y_train------------------")
    print(y_train)
    print("------------------y_test------------------")
    print(y_test)

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

image = np.zeros((500*10*2,750*10*2),dtype=np.uint8)

pixels = 500*10*2*750*10*2
nthreads = 32
nblocksy = ((500*10*2)//nthreads) + 1 + 2
nblocksx = ((750*10*2)//nthreads) + 1 + 2

s=timer()
create_fractal[(nblocksx,nblocksy),(nthreads,nthreads)](
    -2.0,1.0,-1.0,1.0,image,20
)
e = timer()
print("Execution time on GPU: %f seconds"%(e-s))

ml()
imshow(image)
show()