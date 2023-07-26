from __future__ import print_function, division, absolute_import
import numpy as np
from matplotlib.pylab import imshow,show
from timeit import default_timer as timer
from numba import cuda
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

image = np.zeros((500*10*2,750*10*2),dtype=np.uint8)

pixels = 500*10*2*750*10*2
nthreads = 32
nblocksy = ((500*10*2*2)//nthreads) + 1 * 100
nblocksx = ((750*10*2*2)//nthreads) + 1 * 100

s=timer()
create_fractal[(nblocksx,nblocksy),(nthreads,nthreads)](
    -2.0,1.0,-1.0,1.0,image,20
)
e = timer()
print("Execution time on GPU: %f seconds"%(e-s))

calculate_mean_and_plot()
imshow(image)
show()