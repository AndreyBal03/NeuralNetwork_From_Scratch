import numpy as np

class NeuralLayer:
    def __init__(self, in_feat, out_feat):
        self.W = np.random.rand(in_feat, out_feat) * np.sqrt(2 / in_feat)
        self.B = np.zeros(out_feat)

    def __call__(self, X):
        #Will make the inference
        return X@self.W + self.B


class NeuralNetwork:
    def __init__(self, topology = []):
        #Here I am going to define the structure of the network
        #Using topology as a map for how many neurons are going to be
        #On each layer
        
        self.topology = topology
        self.layer_stack = []

        for i in range(len(topology) - 1):
            #goes to n-1 layer
            self.layer_stack.append(NeuralLayer(topology[i],topology[i+1]))

        self.N = len(self.layer_stack) #Number of layers
        #Activation function (For this calculations sigmoid)

        self.f = lambda x: 1 / (1 + np.exp(-x))

        #Activation function derivative
        self.f_ = lambda x: self.f(x)*(1-self.f(x))
    
    def __call__(self, X):
        #My forward pass will only be pass my data throw all the layers and the activation function
        for layer in self.layer_stack:
            X = self.f(layer(X))
        return X

    def fit(self, X, Y, lr = .003, epochs = 40000):
        #Here are going to be the optimization
        #Using backpropagation and gradient descent
        X_ = X# Only for testing later

        for epoch in range(epochs):
            X = X_
            aL = [X] #This will save all the outputs of each layer
            #The first value is X, because it is our input layer
            for i in self.layer_stack:
                X = self.f(i(X))
                aL.append(X)

            #I will only performe the last layer delta(error input neuron)
            #And its implementation
            indx = len(aL) - 1 #index of the outputs

            delta = aL[indx]*(1-aL[indx])*binary_cross_entropy_der(Y,aL[indx])
            grad_B = np.mean(delta,axis = 0)
            grad_W = aL[indx-1].T@delta

            self.layer_stack[-1].W -= grad_W*lr
            self.layer_stack[-1].B -= grad_B*lr

            # print(grad_W.shape, delta.shape,aL[-2].shape)
            for layer in range(self.N-1):
                indx -=1
                n = self.N-layer-2 #actual layer
                n_ = n+1 #Last layer

                aL_ = aL[indx] #input to the next layer
                aL__ = aL[indx-1] #input to our layer
                
                #new delta
                delta = delta@self.layer_stack[n_].W.T*(aL_*(1-aL_))

                grad_W = aL__.T@delta
                grad_B = np.mean(delta, axis = 0)

                #Updating data
                self.layer_stack[n].W -= grad_W*lr
                self.layer_stack[n].B -= grad_B*lr
            
            if epoch %100 == 0:
                Y_pred = self.__call__(X_)
                print(f"Epoch: {epoch}, BCE: {binary_cross_entropy(Y, Y_pred):.4f}")


def binary_cross_entropy(Y, Y_pred):
    #My error function
    return -np.mean(Y * np.log(Y_pred) + (1-Y)*np.log(1-Y_pred)) 
    
def binary_cross_entropy_der(Y,Y_pred):
    #This is the derivative
    # return -np.mean(Y/Y_pred - (1-Y)/(1-Y_pred))
    # return (Y_pred - Y) / (Y_pred * (1 - Y_pred))
    return Y_pred-Y #This is a simplified version that my research find useful

    
