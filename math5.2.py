from __future__ import print_function, division, absolute_import
import numpy as np
from matplotlib.pylab import imshow,show
from timeit import default_timer as timer
from numba import cuda
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

@cuda.jit(device=True)
def mandelbort(x, y, max_iters):
    c = complex(x, y)
    z = 0.0j
    i = 0

    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
        
    return 255
    
@cuda.jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    width = image.shape[1]
    height = image.shape[0]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandelbort(real, imag, iters)
            image[y, x] = color 

def D3():
    # Generate data
    x = np.random.rand(90)
    y = np.random.rand(90)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.column_stack((X.flatten(), Y.flatten())))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, (Z > 0).flatten(), test_size=0.2, random_state=42)
    print("----------------------X_train----------------------")
    print(X_train)
    print("----------------------X_test----------------------")
    print(X_test)
    print("----------------------y_train----------------------")
    print(y_train)
    print("----------------------y_test----------------------")
    print(y_test)

    # Create RandomForestClassifier model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)
    accuracy_train_rf = accuracy_score(y_train, y_train_pred_rf)
    accuracy_test_rf = accuracy_score(y_test, y_test_pred_rf)
    print("RandomForestClassifier - Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(accuracy_train_rf, accuracy_test_rf))

    # Create MLPRegressor model
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
    mlp.fit(X_train, y_train)
    y_train_pred_mlp = mlp.predict(X_train)
    y_test_pred_mlp = mlp.predict(X_test)
    mse_train_mlp = mean_squared_error(y_train, y_train_pred_mlp)
    mse_test_mlp = mean_squared_error(y_test, y_test_pred_mlp)
    print("MLPRegressor - Train MSE: {:.4f}, Test MSE: {:.4f}".format(mse_train_mlp, mse_test_mlp))

    # Create LinearRegression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred_lr = lr.predict(X_train)
    y_test_pred_lr = lr.predict(X_test)
    mse_train_lr = mean_squared_error(y_train, y_train_pred_lr)
    mse_test_lr = mean_squared_error(y_test, y_test_pred_lr)
    print("LinearRegression - Train MSE: {:.4f}, Test MSE: {:.4f}".format(mse_train_lr, mse_test_lr))

    # Create SVR model
    svr = SVR()
    svr.fit(X_train, y_train)
    y_train_pred_svr = svr.predict(X_train)
    y_test_pred_svr = svr.predict(X_test)
    mse_train_svr = mean_squared_error(y_train, y_train_pred_svr)
    mse_test_svr = mean_squared_error(y_test, y_test_pred_svr)
    print("SVR - Train MSE: {:.4f}, Test MSE: {:.4f}".format(mse_train_svr, mse_test_svr))

    # Visualize the results
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(Z > 0).flatten(), cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Data with PCA')
    plt.colorbar()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Regression Results')
    ax1.title.set_text('RandomForestClassifier')
    ax1.scatter(X_train[:, 0], y_train, color='blue', label='Train')
    ax1.scatter(X_test[:, 0], y_test, color='red', label='Test')
    ax1.plot(X_train[:, 0], y_train_pred_rf, color='green', label='Predicted')
    ax1.legend()
    ax2.title.set_text('MLPRegressor')
    ax2.scatter(X_train[:, 0], y_train, color='blue', label='Train')
    ax2.scatter(X_test[:, 0], y_test, color='red', label='Test')
    ax2.plot(X_train[:, 0], y_train_pred_mlp, color='green', label='Predicted')
    ax2.legend()

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

    # Show the plots
    plt.show()

image = np.zeros((500*10*2,750*10*2),dtype=np.uint8)

pixels = 500*10*2*750*10*2
nthreads = 32
nblocksy = ((500*10*2)//nthreads) + 1 
nblocksx = ((750*10*2)//nthreads) + 1 

s=timer()
create_fractal[(nblocksx,nblocksy),(nthreads,nthreads)](
    -2.0,1.0,-1.0,1.0,image,20
)
e = timer()
print("Execution time on GPU: %f seconds"%(e-s))

D3()
imshow(image)
show()