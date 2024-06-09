import numpy as np
import matplotlib.pyplot as plt
import random



def f(x):
    return 1*x**4 - 5*x**2 - 3*x

def data_generator(N=1000, seed=1234):
    np.random.seed(seed)
    x       = np.random.uniform(low=-3., high=3., size=[N,1])
    epsilon = np.random.normal(loc=0.0, scale=5.0, size=[N,1])    
    y = f(x) + epsilon
    return (x, y)

def plot_groundtruth(ax):
    x = np.linspace(-3.1, 3.1, 1000)
    ax.plot(x, f(x), linestyle='-', color='black')
    
def plot_data(ax, x, y):
    ax.scatter(x, y, marker='o', color='gray', alpha=0.7)   
      
def transformation(x, K):
    phi = np.zeros([len(x), K])
    for k in range(K):
        phi[:, [k]] = x**k
    return phi

def f_predict(w, phi):
    return np.matmul(phi, w)

def evaluation(y, y_hat): #
    return np.mean(np.sqrt((y-y_hat)**2))



### FILL IN  ############################################################
def optimize_CF(phi, y):
    result = np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)), np.matmul(np.transpose(phi), y))
    return result


def compute_gradient(w, phi, y): 
    '''
        Note. ignore the effect of sigma (of the noise) as this will be capture in the step-size
    '''
    wT = np.transpose(w)
    phiT = np.transpose(phi)
    yT = np.transpose(y)
    grad = np.matmul(wT, np.matmul(phiT, phi)) - np.matmul(yT, phi)
    return 1/len(phi) * grad
    

# do not use tolerance and make it run until the final iteration
def optimize_GD(phi, y, step_size = 1e-4, iteration = 20000):    

    error     = np.zeros([iteration])
    w         = np.zeros([phi.shape[1], 1]) #initialization
    
    itr       = 0    
    while(itr < iteration):
        grad = compute_gradient(w, phi, y)
        w    = w - step_size * grad

        error[itr] = evaluation(y, np.matmul(phi, np.transpose(w)))
        
        itr += 1        
        
    print('w_opt_GD: {}'.format(w))
    return w, error


#########################################################################





# do not touch this... 
def run_problem3_cde():
    K = 5    
    x, y = data_generator(N=500, seed=1234)

    fig, ax = plt.subplots()

    plot_groundtruth(ax)
    plot_data(ax, x,y)

    ax.set_ylabel('f(x)', fontsize=15)
    ax.set_xlabel('x', fontsize=15)
    ax.grid()
    ax.set_ylim([-23, 53])

    phi      = transformation(x, K)
    w_opt_CF = optimize_CF(phi, y)

    
    iteration = 20000
    step_size = 2e-4
    ### run GD
    w_opt_GD, error_GD   = optimize_GD(phi, y, step_size = step_size, iteration = iteration)
    
    
    z = np.linspace(-3.1, 3.1, 1000).reshape([-1,1])
    
    ### plot f(x; w_opt_CF)
    ax.plot(z, f_predict(w_opt_CF, transformation(z, K)), linestyle='-', color='red')
    
    ### plot f(x; w_opt_GD)
    ax.plot(z, f_predict(w_opt_GD, transformation(z, K)), linestyle='-', color='blue')
    

    plt.legend(['Ground Truth', 'Data Points', 'Closed Form', 'GD'], loc=1)
    plt.show()
    plt.close()
      
    
    plt.figure()
    plt.plot(np.arange(iteration), error_GD, alpha = 0.7,  color='blue')

    plt.grid()

    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('MSE', fontsize=15)
    plt.show()
    plt.close()
    
    
### do not touch this...
def run_problem3_g(K):
    x, y = data_generator(N=50, seed=1234)

    fig, ax = plt.subplots()

    plot_groundtruth(ax)
    plot_data(ax, x,y)

    ax.set_ylabel('f(x)', fontsize=15)
    ax.set_xlabel('x', fontsize=15)
    ax.grid()
    ax.set_ylim([-23, 53])

    phi      = transformation(x, K)
    w_opt_CF = optimize_CF(phi, y)
    
    z = np.linspace(-3.1, 3.1, 1000).reshape([-1,1])
    
    ### plot f(x; w_opt_CF)
    ax.plot(z, f_predict(w_opt_CF, transformation(z, K)), linestyle='-', color='red')
    
    plt.legend(['Ground Truth', 'Data Points', 'K={}'.format(K)], loc=1)
    plt.show()
    plt.close()
    

    
if __name__ == "__main__":
    
    run_problem3_cde()
    
    run_problem3_g(K=20) #you should change K to compare polynomial regression with different K's 