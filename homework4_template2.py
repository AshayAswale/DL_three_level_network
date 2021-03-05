import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
import copy
import random

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 1
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10


# def random_rearrange (X_tr, y_tr, seed):
#     np.random.seed(seed)
#     np.random.shuffle(X_tr)
#     np.random.seed(seed)
#     np.random.shuffle(y_tr)

# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

#Checked
def relu(z):
    return np.maximum(0, z)

#Checked
def relu_d(z):
    h_tild=copy.deepcopy(z)
    h_tild[h_tild<=0]=0
    h_tild[h_tild>0]=1
    return h_tild

#Checked
def getYHat(zs):
    exp_z = np.exp(zs)
    exp_z_sums = np.sum(exp_z, axis=0)
    y_hat = (exp_z/exp_z_sums)
    return y_hat

#Checked
def getError(y_te, y_hat):
    cee = (-(np.sum(y_te*np.log(y_hat))) /y_hat.shape[1])
    return cee

def forward_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)

    zs=[]
    hs=[] 
    
    zs.append((np.dot(      Ws[0],   x).T + bs[0]).T) ######## Extra Transpose
    hs.append(relu(zs[0]))
    yhat=getYHat(zs[0]) 

    loss = getError(y, yhat)  #### Possible???
    
    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat
   
def back_prop (x, y, weightsAndBiases, alpha = 0.01):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    Ws, bs = unpack(weightsAndBiases)

    dJdWs = copy.deepcopy(Ws)  # Gradients w.r.t. weights   # Just for dimentions, deepcopy
    dJdbs = copy.deepcopy(bs)  # Gradients w.r.t. biases    # Just for dimentions, deepcopy

    g = (yhat - y)
    g = g*relu_d(zs[0])
    dJdWs[1] = np.dot(g, hs[0].T) 
    dJdbs[1] = np.mean(g, axis=1) 
    g = np.dot(Ws[1].T, g)
    g = g*relu_d(zs[0])
    dJdWs[0] = np.dot(g, x.T) + alpha*Ws[0]/len(x[0])
    dJdbs[0] = np.mean(g, axis=1) 
    
    # Concatenate gradients
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 


def stoch_grad_regression (X_tr, y_tr, weightsAndBiases):

    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]
    X_tr_raw = X_tr
    y_tr_raw = y_tr
    vald_perct = 80
    vald_num = (int)(no_data*vald_perct/100)

    # Step 1, random w and b generation

    # Randomizing the data
    randint = (random.randint(1, 99))
    random_rearrange(X_tr, y_tr, randint) #seed can be any random number

    ###############################
    #### Final Training Tuning ####
    ###############################
    # value = int (input("Enter 1 for hyperparameter tuning\nEnter 2 for training on the tuned hyperparameters\n"))
    
    # if (value == 1):
    #     print("Tuning Hyperparameters!")
    #     n_squig, eps, alpha, epochs = double_cross_validation(X_tr, y_tr)
    # else:
    #     print("Training using pretuned hyperparameters")
    n_squig, eps, alpha, epochs = 160, 0.1, 0.05, 500


    X_tr = X_tr_raw[0:vald_num]
    X_tr_vald = X_tr_raw[vald_num:]
    y_tr = y_tr_raw[0:vald_num]
    y_tr_vald = y_tr_raw[vald_num:]


    no_data = X_tr.shape[0]

    for epoch in range(0, epochs):
        print("Epoch:", epoch)
        print("Validation Error:",test_data(X_tr_vald, y_tr_vald, weightsAndBiases))
        data_remain = True
        n_curr = 0
        n_next = n_squig
        i = 0
        while(data_remain):
            # print(i)
            i+=1
            X_tr_temp = X_tr[n_curr:(min(n_next, no_data))]
            y_tr_temp = y_tr[n_curr:(min(n_next, no_data))]
            n_curr = n_next
            n_next += n_squig

            data_remain = True if n_next<no_data else False
            
            dwdbs = back_prop(X_tr, y_tr, weightsAndBiases, alpha)
            print(np.mean(dwdbs))
            # print(weightsAndBiases)
            # print(dwdbs)
            weightsAndBiases -= eps*dwdbs
            # print(weightsAndBiases)
            # exit()
            
            
            
    return w,b
        

def test_data(X_te, y_te, weightsAndBiases):
    y_te_raw = np.argmax(y_te, axis=1)
    loss, zs, hs, y_hat = forward_prop(X_te, y_te, weightsAndBiases)
    
    y_cat = np.argmax(y_hat, axis=1)
    err_mat = y_cat - y_te_raw
    count = np.count_nonzero(err_mat == 0)
    no_of_data = y_te.shape[0]
    perct = count/no_of_data*100

    return loss, perct


def train (trainX, trainY, weightsAndBiases, testX, testY):
    NUM_EPOCHS = 100
    trajectory = []
    for epoch in range(NUM_EPOCHS):
        # TODO: implement SGD.
        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.
        pass
        
    return weightsAndBiases, trajectory

# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases ():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def plotSGDPath (trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    def toyFunction (x1, x2):
        return np.sin((2 * x1**2 - x2) / 10.)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()


def loadDataset():
    # Load data
    X_tr_raw = (np.load("fashion_mnist_train_images.npy"))
    y_tr_raw = np.load("fashion_mnist_train_labels.npy")
    X_te_raw = (np.load("fashion_mnist_test_images.npy"))
    y_te_raw = np.load("fashion_mnist_test_labels.npy")

    no_data = X_tr_raw.shape[0]

    brightness_value = 256
    X_te = (X_te_raw/brightness_value).T
    X_tr = (X_tr_raw/brightness_value).T
    
    y_tr = (np.zeros([X_tr_raw.shape[0], NUM_OUTPUT])).T
    y_tr_raw = (np.atleast_2d(y_tr_raw))
    np.put_along_axis(y_tr, y_tr_raw, 1, axis=0)
    
    y_te = (np.zeros([X_te_raw.shape[0], NUM_OUTPUT])).T
    y_te_raw = (np.atleast_2d(y_te_raw))
    np.put_along_axis(y_te, y_te_raw, 1, axis=0)
    # print(X_te_raw[:,0:5].shape)
    # print(X_te[:,0:5].shape)
    # exit()
    return X_te, y_te, X_tr, y_tr


if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    # trainX = ...
    # trainY = ...
    # ...

    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()
    trainX, trainY, testX, testY = loadDataset()
    
    a = []

    # Perform gradient check on random training examples
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab)[0], \
                                    lambda wab: back_prop(np.atleast_2d(testX[:,0:5]), np.atleast_2d(testY[:,0:5]), wab), \
                                    weightsAndBiases))
    # stoch_grad_regression(trainX, trainY, weightsAndBiases)

    # weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)
    
    # # Plot the SGD trajectory
    # plotSGDPath(trainX, trainY, ws)
