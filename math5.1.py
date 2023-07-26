from __future__ import print_function, division, absolute_import
import numpy as np
from matplotlib.pylab import imshow,show
from timeit import default_timer as timer
from numba import cuda
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def D3():
    # Generate data
    x = np.random.rand(400)
    y = np.random.rand(400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create MLPRegressor model
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
    mlp.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_train_pred_mlp = mlp.predict(X_train.reshape(-1, 1))
    y_test_pred_mlp = mlp.predict(X_test.reshape(-1, 1))
    mse_train_mlp = mean_squared_error(y_train, y_train_pred_mlp)
    mse_test_mlp = mean_squared_error(y_test, y_test_pred_mlp)
    print("MLPRegressor - Train MSE: {:.4f}, Test MSE: {:.4f}".format(mse_train_mlp, mse_test_mlp))

    # Create LinearRegression model
    lr = LinearRegression()
    lr.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_train_pred_lr = lr.predict(X_train.reshape(-1, 1))
    y_test_pred_lr = lr.predict(X_test.reshape(-1, 1))
    mse_train_lr = mean_squared_error(y_train, y_train_pred_lr)
    mse_test_lr = mean_squared_error(y_test, y_test_pred_lr)
    print("LinearRegression - Train MSE: {:.4f}, Test MSE: {:.4f}".format(mse_train_lr, mse_test_lr))

    # Create SVR model
    svr = SVR()
    svr.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_train_pred_svr = svr.predict(X_train.reshape(-1, 1))
    y_test_pred_svr = svr.predict(X_test.reshape(-1, 1))
    mse_train_svr = mean_squared_error(y_train, y_train_pred_svr)
    mse_test_svr = mean_squared_error(y_test, y_test_pred_svr)
    print("SVR - Train MSE: {:.4f}, Test MSE: {:.4f}".format(mse_train_svr, mse_test_svr))

    # Visualize the results
    plt.scatter(x, y, color='blue', label='Data')
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Result')
    plt.legend()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Result')
    ax1.title.set_text('Result 1')
    ax1.plot(X_train, y_train)
    ax1.plot(X_test, y_test)
    ax1.scatter(X_train, y_train)
    ax1.scatter(X_test, y_test)
    ax2.title.set_text('Result 2')
    ax2.scatter(X_train, y_train)
    ax2.scatter(X_test, y_test)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z)

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot')

    # Show the plot
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

D3()
imshow(image)
show()