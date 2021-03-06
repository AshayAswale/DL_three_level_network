import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
import copy
import random
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
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
    # print(zs.shape)
    exp_z_sums = np.sum(exp_z, axis=0)
    y_hat = (exp_z/exp_z_sums)
    # print(y_hat.shape)
    return y_hat

#Checked
def getError(y_te, y_hat):
    cee = (-(np.sum(y_te*np.log(y_hat))) /y_hat.shape[1])
    return cee

def forward_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)

    zs=[]
    hs=[] 
    hs.append(x)
    for i in range(NUM_HIDDEN_LAYERS):
        zs.append((np.dot( Ws[i], hs[i]).T + bs[i]).T) ######## Extra Transpose
        hs.append(relu(zs[i]))
    zs.append((np.dot( Ws[-1], hs[-1]).T + bs[-1]).T) ######## Extra Transpose
    yhat=getYHat(zs[-1]) 

    loss = getError(y, yhat)  #### Possible???
    
    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat
   
def back_prop (x, y, weightsAndBiases, alpha = 0.01):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    # print(loss)
    Ws, bs = unpack(weightsAndBiases)

    dJdWs = []  # Gradients w.r.t. weights  
    dJdbs = []  # Gradients w.r.t. biases   

    g = (yhat - y)
    dJdWs.append( np.dot(g, hs[-1].T) )
    dJdbs.append( np.mean(g, axis=1) )

    for i in range(NUM_HIDDEN_LAYERS-1, -1, -1):
        g = (np.dot(g.T, Ws[i+1])*(relu_d(zs[i]).T)).T
        # print(zs[0])
        dJdWs.append( np.dot(g, hs[i].T) + alpha*Ws[i]/len(x[0]))    
        dJdbs.append( np.mean(g, axis=1) )

    dJdWs.reverse()
    dJdbs.reverse()

    # Concatenate gradients
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 


def double_cross_validation(X_tr, y_tr, X_tr_vald, y_tr_vald):
    no_data = X_tr.shape[1]
    no_features = X_tr.shape[0]

    global NUM_HIDDEN_LAYERS, NUM_HIDDEN

    n_hidden_layers_set = np.array([3,4,5])
    n_hidden_set = np.array([30,40,50])
    eps_set = np.array([0.001, 0.005, 0.01, 0.05])
    n_squig_set = np.array([16,32,64])
    alpha_set = np.array([0.0])
    epochs_set = np.array([50])
    # 16 0.001 0.05 100
    # n_squig_set = np.array([10000, 50000])
    # eps_set = np.array([0.01])
    # alpha_set = np.array([0.01])
    # epochs_set = np.array([5,10])
    

    h_star = [n_squig_set[0], eps_set[0], alpha_set[0], epochs_set[0]]
    A_error = math.inf
    A_accuracy = 0

    i = 0

    for n_hidden_layers in n_hidden_layers_set:
        for n_hidden in n_hidden_set:
            for n_squig in n_squig_set:
                for eps in eps_set:
                    for alpha in alpha_set:
                        for epochs in epochs_set:
                            NUM_HIDDEN_LAYERS = n_hidden_layers
                            NUM_HIDDEN = n_hidden
                            weightsAndBiases = initWeightsAndBiases()
                            # w,b = unpack(weightsAndBiases)
                            # print(w[0].shape)
                            # print(weightsAndBiases.shape)
                            weightsAndBiases = training(X_tr, y_tr, weightsAndBiases, n_squig, eps, alpha, epochs)
                            # print(test_data(X_tr_vald, y_tr_vald, weightsAndBiases))
                            error, perct = test_data(X_tr_vald, y_tr_vald, weightsAndBiases)
                            # print("Hyperparameters: \n","n_hidden_layers: ",n_hidden_layers,"  n_hidden: ",n_hidden,"  n_squig: ",n_squig,"  eps: ",eps,"  alpha: ",alpha,"  epochs: ",epochs)
                            
                            print("Error:", error, " Accuracy:", perct, "h_star:", n_hidden_layers,n_hidden,n_squig,eps,alpha,epochs, "######## case:", i)
                            i += 1
                            if(error<A_error and perct > A_accuracy):
                                A_error = error
                                A_accuracy = perct
                                h_star = [n_squig, eps, alpha, epochs]
                                NUM_HIDDEN = n_hidden
                                NUM_HIDDEN_LAYERS = n_hidden_layers
                                print("## h_star updated ##")
    
    print("######################")
    print("BEST HYPERPARAMETERS")
    print(h_star[0], h_star[1], h_star[2], h_star[3])
    print("######################")
    return h_star[0], h_star[1], h_star[2], h_star[3]
    


def stoch_grad_regression (X_tr, y_tr, weightsAndBiases):

    no_data = X_tr.shape[1]
    no_features = X_tr.shape[0]
    X_tr_raw = X_tr
    y_tr_raw = y_tr
    vald_perct = 80
    vald_num = (int)(no_data*vald_perct/100)

    global NUM_HIDDEN_LAYERS, NUM_HIDDEN

    # Step 1, random w and b generation

    # Randomizing the data
    # randint = (random.randint(1, 99))
    # random_rearrange(X_tr, y_tr, randint) #seed can be any random number


    X_tr = X_tr_raw[:,0:vald_num]
    X_tr_vald = X_tr_raw[:,vald_num:]
    y_tr = y_tr_raw[:,0:vald_num]
    y_tr_vald = y_tr_raw[:,vald_num:]


    ###############################
    #### Final Training Tuning ####
    ###############################
    value = int (input("Enter 1 for hyperparameter tuning\nEnter 2 for training on the tuned hyperparameters\n"))
    
    if (value == 1):
        print("Tuning Hyperparameters!")
        n_squig, eps, alpha, epochs = double_cross_validation(X_tr, y_tr, X_tr_vald, y_tr_vald)
    else:
        print("Training using pretuned hyperparameters")
        #  3 50 32 0.001 0.0 100
        NUM_HIDDEN_LAYERS = 3
        NUM_HIDDEN = 50
        n_squig, eps, alpha, epochs = 32, 0.001, 0.0, 100
    # n_squig, eps, alpha, epochs = 32, 0.001, 0.0, 100

    weightsAndBiases = initWeightsAndBiases()    
    weightsAndBiases, trajectory = training(X_tr, y_tr, weightsAndBiases, n_squig, eps, alpha, epochs)
    print("FINAL Validation", test_data(X_tr_vald, y_tr_vald, weightsAndBiases))
    # print("Validation:",training(X_tr, X_tr_vald, y_tr, y_tr_vald, weightsAndBiases, n_squig, eps, alpha, epochs))
    return weightsAndBiases, trajectory


def training(X_tr, y_tr, weightsAndBiases, n_squig, eps, alpha, epochs):
    no_data = X_tr.shape[1]
    trajectory = []
    for epoch in range(0, epochs):
        data_remain = True
        n_curr = 0
        n_next = n_squig
        i = 0
        # print(test_data(X_tr, y_tr, weightsAndBiases))
        while(data_remain):
            i+=1
            X_tr_temp = X_tr[:,n_curr:(min(n_next, no_data))]
            y_tr_temp = y_tr[:,n_curr:(min(n_next, no_data))]

            n_curr = n_next
            n_next += n_squig

            data_remain = True if n_next<no_data else False
            
            dwdbs = back_prop(X_tr_temp, y_tr_temp, weightsAndBiases, alpha)
            weightsAndBiases -= eps*dwdbs
        trajectory.append(weightsAndBiases)
        print(len(trajectory))
            
    return weightsAndBiases, trajectory
        

def test_data(X_te, y_te, weightsAndBiases):
    y_te_raw = np.argmax(y_te, axis=0)
    loss, zs, hs, y_hat = forward_prop(X_te, y_te, weightsAndBiases)
    
    y_cat = np.argmax(y_hat, axis=0)
    err_mat = y_cat - y_te_raw
    count = np.count_nonzero(err_mat == 0)
    no_of_data = y_te.shape[1]
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
    pca = PCA(n_components=2)
    trajectory = np.array(trajectory)
    transWeightsAndBiases = pca.fit_transform(trajectory)

    def toyFunction (x1, x2):
        invWeightsAndBiases = pca.inverse_transform([x1, x2])
        loss = forward_prop(trainX, trainY, invWeightsAndBiases)[0]
        return loss


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-10, +10, 1)  # Just an example
    axis2 = np.arange(-10, +10, 1)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # # Now superimpose a scatter plot showing the weights during SGD.
    # Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    # Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    # Zaxis = toyFunction(Xaxis, Yaxis)
    # ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()


def loadDataset():
    # Load data
    X_tr_raw = (np.load("fashion_mnist_train_images.npy"))
    y_tr_raw = np.load("fashion_mnist_train_labels.npy")
    X_te_raw = (np.load("fashion_mnist_test_images.npy"))
    y_te_raw = np.load("fashion_mnist_test_labels.npy")

    no_data = X_tr_raw.shape[0]

    brightness_value = 255
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
    # test_data(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weightsAndBiases)
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab)[0], \
                                    lambda wab: back_prop(np.atleast_2d(testX[:,0:5]), np.atleast_2d(testY[:,0:5]), wab), \
                                    weightsAndBiases))
    weightsAndBiases, trajectory = stoch_grad_regression(trainX, trainY, weightsAndBiases)
    cee_err, perct = test_data(testX, testY, weightsAndBiases)
    print("################################")
    print("FINAL TESTING ERROR: ",cee_err, "\nAccuracy: ", perct)
    print("################################")
    

    # weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)
    
    # # Plot the SGD trajectory
    plotSGDPath(trainX,trainY,trajectory)
